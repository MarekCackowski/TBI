from config import CONFIG
import logging
import numpy as np
import optuna
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    TabNetClassifier = None

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def ensemble_objective(trial, X_train_numeric, y_train_numeric, cv_strategy, monotone_tuple, binary_indices):
    """ Funkcja celu dla Optuny z dynamiczną selekcją architektury zespołu (NAS) i stałym wsparciem BernoulliNB. """
    
    ensemble_type = trial.suggest_categorical('ensemble_type', ['duo_speed', 'duo_orthogonal', 'trio_trees', 'full_stack'])
    base_pos_weight = float((y_train_numeric == 0).sum() / max(1, (y_train_numeric == 1).sum()))
    
    low_bound = max(1.0, base_pos_weight * 0.8)
    high_bound = max(low_bound + 0.1, base_pos_weight * 1.2)

    scores = []
    for train_idx, val_idx in cv_strategy.split(X_train_numeric, y_train_numeric):
        X_tr, X_va = X_train_numeric[train_idx], X_train_numeric[val_idx]
        y_tr, y_va = y_train_numeric[train_idx], y_train_numeric[val_idx]
        
        # Tworzymy listę każdym foldzie
        estimators = []

        # XGBoost
        if ensemble_type in ['duo_orthogonal', 'trio_trees', 'full_stack']:
            xgb_p = {
                # Wyższa liczba drzew i niski LR, dla stabilności przy budowaniu drzew
                'n_estimators': trial.suggest_int('xgb_n_estimators', 200, 600),
                'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 6),
                # Wymóg większej liczby próbek w liściu zapobiega tworzeniu reguł dla pojedynczych chorych
                'subsample': trial.suggest_float('xgb_subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 0.9),
                'min_child_weight': trial.suggest_int('xgb_min_child_weight', 10, 50),
                # Silniejsza regularyzacja L1/L2 ze względu na silnie skorelowane cechy hemodynamiczne
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-2, 10.0, log=True),
                'scale_pos_weight': trial.suggest_float('xgb_spw', low_bound, high_bound, log=True),
                'random_state': CONFIG["SEED"],
                'verbosity': 0,
                'tree_method': 'hist',
                'device': 'cpu',
                'monotone_constraints': monotone_tuple
            }
            estimators.append(('xgb', xgb.XGBClassifier(**xgb_p)))

        # LightGBM
        if ensemble_type in ['duo_speed', 'trio_trees', 'full_stack']:
            lgbm_p = {
                'n_estimators': trial.suggest_int('lgbm_n_estimators', 200, 600),
                'learning_rate': trial.suggest_float('lgbm_lr', 0.01, 0.15, log=True),
                # Ograniczenie num_leaves dozwolone przez max_depth (2^max_depth)
                'num_leaves': trial.suggest_int('lgbm_leaves', 31, 80),
                # Wymóg większej liczby próbek w liściu zapobiega tworzeniu reguł dla pojedynczych chorych
                'min_child_samples': trial.suggest_int('lgbm_child', 20, 50),
                'scale_pos_weight': trial.suggest_float('lgbm_spw', low_bound, high_bound, log=True),
                'random_state': CONFIG["SEED"],
                'verbosity': -1,
                'device_type': 'cpu'
            }
            estimators.append(('lgbm', LGBMClassifier(**lgbm_p)))

        # CatBoost
        if ensemble_type in ['duo_speed', 'trio_trees', 'full_stack']:
            cat_p = {
                'depth': trial.suggest_int('cat_depth', 4, 6),
                'learning_rate': trial.suggest_float('cat_lr', 0.02, 0.15, log=True),
                # Wymóg większej liczby próbek w liściu zapobiega tworzeniu reguł dla pojedynczych chorych
                'min_data_in_leaf': trial.suggest_int('cat_min_data', 20, 50),
                # Operujemy na karze L2, żeby mocniej karać modele z nadmierną liczbą liści i zbyt dużymi wagami
                'l2_leaf_reg': trial.suggest_float('cat_l2', 2.0, 10.0, log=True),
                'random_seed': CONFIG["SEED"],
                'verbose': 0,
                'task_type': 'CPU'
            }
            estimators.append(('cat', CatBoostClassifier(**cat_p)))

        # TabNet
        if ensemble_type in ['duo_orthogonal', 'full_stack']:
            gamma_val = trial.suggest_float('tabnet_gamma', 1.0, 1.3)
            estimators.append(('tabnet', TabNetClassifier(n_d=16, n_a=16, gamma=gamma_val, verbose=0, seed=CONFIG["SEED"])))

        # Ekspert Czerwonych Flag: BernoulliNB
        if binary_indices:
            nb_expert = Pipeline([
                ('filter_binary', ColumnTransformer(
                    transformers=[('bin_keep', 'passthrough', binary_indices)],
                    remainder='drop'
                )),
                ('nb', BernoulliNB(alpha=1.0, binarize=None, fit_prior=True))
            ])
            estimators.append(('red_flag_nb', nb_expert))

        model = StackingClassifier(estimators=estimators, cv=3, n_jobs=1)

        try:
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y_va, preds))
        except (ValueError, RuntimeError, TypeError) as e:
            logger.warning(f"Próba Optuny odrzucona (błąd modelu): {e}")
            return 0.0

    return np.mean(scores) if scores else 0.0