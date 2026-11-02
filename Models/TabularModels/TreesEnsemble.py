import functools
import os

from TBI import BASE_DIR, SETTINGS, CONFIG
import logging
import joblib
import numpy as np
import optuna
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    TabNetClassifier = None
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import BernoulliNB

# Optymalizacja hiperparametrów
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

logger = logging.getLogger(__name__)


def ensemble_objective(trial, X_train_numeric, y_train_numeric, cv_strategy, monotone_tuple, binary_indices):
    """ Funkcja celu dla Optuny z dynamiczną selekcją architektury zespołu (NAS) i stałym wsparciem BernoulliNB. """
    ensemble_type = trial.suggest_categorical('ensemble_type', ['duo_speed', 'duo_orthogonal', 'trio_trees', 'full_stack'])
    base_pos_weight = float((y_train_numeric == 0).sum() / max(1, (y_train_numeric == 1).sum()))

    # Znacznie mocniejsze uderzenie w fałszywe pominięcia
    low_bound = max(1.5, base_pos_weight * 1.2)
    high_bound = max(low_bound + 0.1, base_pos_weight * 3.0)

    scores = []
    for train_idx, val_idx in cv_strategy.split(X_train_numeric, y_train_numeric):
        X_tr, X_va = X_train_numeric[train_idx], X_train_numeric[val_idx]
        y_tr, y_va = y_train_numeric[train_idx], y_train_numeric[val_idx]

        # Tworzymy listę w każdym foldzie
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
        has_tabnet = False
        if ensemble_type in ['duo_orthogonal', 'full_stack'] and TabNetClassifier is not None:
            has_tabnet = True
            gamma_val = trial.suggest_float('tabnet_gamma', 1.0, 1.3)
            tabnet_epochs = trial.suggest_int('tabnet_max_epochs', 15, 30)
            estimators.append(('tabnet', TabNetClassifier(
                n_d=16,
                n_a=16,
                gamma=gamma_val,
                verbose=0,
                seed=CONFIG["SEED"],
                mask_type='sparsemax',
                optimizer_params=dict(lr=2e-2)
            )))

        # Ekspert Czerwonych Flag: BernoulliNB
        if binary_indices:
            nb_expert = Pipeline([
                ('filter_binary', ColumnTransformer(
                    transformers=[('bin_keep', 'passthrough', binary_indices)],
                    remainder='drop'
                )),
                ('nb', BernoulliNB(alpha=1.0, binarize=0.5, fit_prior=True))
            ])
            estimators.append(('red_flag_nb', nb_expert))

        model = StackingClassifier(estimators=estimators, cv=3, n_jobs=1)

        try:
            # Przekazujemy ograniczoną liczbę epok do TabNeta wewnątrz StackingClassifiera
            fit_kwargs = {}
            if has_tabnet:
                fit_kwargs['tabnet__max_epochs'] = tabnet_epochs
                fit_kwargs['tabnet__patience'] = 5

            model.fit(X_tr, y_tr, **fit_kwargs)
            preds = model.predict_proba(X_va)[:, 1]

            # Zapobieganie błędom ROC AUC w przypadku braku różnorodności etykiet w foldzie
            if len(np.unique(y_va)) > 1:
                scores.append(roc_auc_score(y_va, preds))
        except (ValueError, RuntimeError, TypeError) as e:
            logger.warning(f"Próba Optuny odrzucona (błąd modelu): {e}")
            return 0.0

    return np.mean(scores) if scores else 0.0


def train_optimized_ensemble(X_train_numeric, y_train_numeric, cv_strategy, monotone_tuple, binary_indices, preprocessor_imputed, X_train, y_train):
    """ Przeprowadza optymalizację hiperparametrów i trenuje ostateczny zespół modeli (Konsylium lekarskie). """
    bound_objective = functools.partial(
        ensemble_objective,
        X_train_numeric=X_train_numeric,
        y_train_numeric=y_train_numeric,
        cv_strategy=cv_strategy,
        monotone_tuple=monotone_tuple,
        binary_indices=binary_indices
    )

    os.makedirs(os.path.join(BASE_DIR, 'output_data', 'optuna_study'), exist_ok=True)
    ensemble_study = optuna.create_study(
        direction="maximize",
        study_name="tbi_ensemble_dynamic_architecture",
        storage=SETTINGS.optuna_db,
        load_if_exists=True,
        sampler=TPESampler(seed=SETTINGS.seed),
        pruner=MedianPruner()
    )
    ensemble_study.optimize(
        bound_objective,
        n_trials=CONFIG.get("N_OPTUNA_TRIALS", 15),
        show_progress_bar=True
    )

    bp = ensemble_study.best_params
    best_ensemble_type = bp.get('ensemble_type', 'full_stack')
    logger.info(f"Optuna wyłoniła optymalną architekturę Konsylium: {best_ensemble_type}")

    final_estimators = []
    fit_kwargs = {}

    if best_ensemble_type in ['duo_orthogonal', 'trio_trees', 'full_stack']:
        xgb_p = {
            'n_estimators': bp['xgb_n_estimators'], 'learning_rate': bp['xgb_lr'], 'max_depth': bp['xgb_max_depth'],
            'reg_alpha': bp['xgb_reg_alpha'], 'reg_lambda': bp['xgb_reg_lambda'], 'scale_pos_weight': bp['xgb_spw'],
            'subsample': bp.get('xgb_subsample', 0.8), 'colsample_bytree': bp.get('xgb_colsample_bytree', 0.8),
            'min_child_weight': bp.get('xgb_min_child_weight', 20),
            'random_state': SETTINGS.seed, 'verbosity': 0, 'tree_method': 'hist', 'device': 'cpu', 'monotone_constraints': monotone_tuple
        }
        final_estimators.append(('xgb', xgb.XGBClassifier(**xgb_p)))

    if best_ensemble_type in ['duo_speed', 'trio_trees', 'full_stack']:
        lgbm_p = {
            'n_estimators': bp['lgbm_n_estimators'], 'learning_rate': bp['lgbm_lr'], 'num_leaves': bp['lgbm_leaves'],
            'min_child_samples': bp['lgbm_child'], 'scale_pos_weight': bp['lgbm_spw'],
            'random_state': SETTINGS.seed, 'verbosity': -1, 'device_type': 'cpu'
        }
        final_estimators.append(('lgbm', LGBMClassifier(**lgbm_p)))

    if best_ensemble_type in ['duo_speed', 'trio_trees', 'full_stack']:
        cat_p = {
            'depth': bp['cat_depth'], 'learning_rate': bp['cat_lr'], 'l2_leaf_reg': bp['cat_l2'],
            'min_data_in_leaf': bp.get('cat_min_data', 20),
            'random_seed': SETTINGS.seed, 'verbose': 0, 'task_type': 'CPU', 'train_dir': os.path.join(SETTINGS.models_dir, 'catboost_info')
        }
        final_estimators.append(('cat', CatBoostClassifier(**cat_p)))

    if best_ensemble_type in ['duo_orthogonal', 'full_stack'] and TabNetClassifier is not None:
        tabnet_pipeline = Pipeline([
            ('preprocessor', preprocessor_imputed),
            ('tabnet', TabNetClassifier(n_d=16, n_a=16, gamma=bp['tabnet_gamma'], verbose=0, seed=SETTINGS.seed, mask_type='sparsemax', optimizer_params=dict(lr=2e-2)))
        ])
        final_estimators.append(('tabnet', tabnet_pipeline))
        fit_kwargs['tabnet__tabnet__max_epochs'] = bp.get('tabnet_max_epochs', 20)
        fit_kwargs['tabnet__tabnet__patience'] = 5

    if binary_indices:
        nb_expert = Pipeline([
            ('filter_binary', ColumnTransformer(
                transformers=[('bin_keep', 'passthrough', binary_indices)],
                remainder='drop'
            )),
            ('nb', BernoulliNB(alpha=1.0, binarize=None, fit_prior=True))
        ])
        final_estimators.append(('red_flag_nb', nb_expert))

    ensemble_best = StackingClassifier(estimators=final_estimators, cv=3, n_jobs=1)
    
    # Trenujemy ensemble podając argumenty fit_kwargs dla TabNeta
    ensemble_best.fit(X_train_numeric, y_train_numeric, **fit_kwargs)

    joblib.dump(ensemble_best, os.path.join(SETTINGS.models_dir, 'Optimized_Stacking_Ensemble.pkl'))
    return ensemble_best


def enrich_features_with_cluster_distances(X_train_numeric, n_clusters=4):
    """ Wylicza odległości Mahalanobisa / Euklidesowe od wykrytych centroidów GMM, 
        wzbogacając dane tabelaryczne dla modeli drzewiastych o sygnał fenotypowy. """
    gmm = GaussianMixture(n_components=n_clusters, random_state=CONFIG.get("SEED", 42), n_init=3)
    gmm.fit(X_train_numeric)
    
    # Wyznaczamy odległości do każdego z centroidów klastrów
    distances = gmm.predict_proba(X_train_numeric) # Miękkie przypisanie do klastrów jako cechy
    
    # Łączymy oryginalne dane numeryczne z probabilistycznymi odległościami do fenotypów
    X_enriched = np.hstack((X_train_numeric, distances))
    return X_enriched


def transform_features_with_cluster_distances(X_test_numeric, gmm_model):
    """ Transformuje zbiór testowy/walidacyjny przy użyciu wcześniej wyuczonego modelu GMM,
        zapobiegając wyciekowi danych. """
    distances = gmm_model.predict_proba(X_test_numeric)
    return np.hstack((X_test_numeric, distances))