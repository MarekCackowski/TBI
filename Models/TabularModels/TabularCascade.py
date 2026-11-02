import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.distributions.beta import Beta
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.experimental import enable_halving_search_cv  # Wymagane przed importem HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from imblearn.under_sampling import RandomUnderSampler
from interpret.glassbox import ExplainableBoostingClassifier
from config import CONFIG
from TBI import CustomClassifierWrapper, PhenotypeTournamentRouter

# Konfiguracja loggera oraz urządzenia obliczeniowego
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdaptiveMedicalNoiseAdder(BaseEstimator, TransformerMixin):
    """ Skale Glasgow i APACHE charakteryzują się dużym rozrzutem.
        Agent RL dynamicznie steruje poziomem szumu, by uodpornić model na błędy ludzkie i niedokładności skali. """
    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self.is_fitted_ = False

        # Domyślne mnożniki (zostaną nadpisane przez Agenta RL, jeśli zostanie użyty)
        self.noise_scales = {'gcs': 0.10, 'vitals': 0.02, 'other': 0.05}

    def update_rl_policy(self, scale_gcs, scale_vitals, scale_other):
        """ Metoda dla Agenta RL aktualizująca siłę szumu w locie.
            Mnożymy wyniki z Sigmoidy przez maksymalne sensowne medycznie progi. """
        self.noise_scales = {
            'gcs': float(scale_gcs * 0.20),       # Max 20% odchylenia standardowego dla skal
            'vitals': float(scale_vitals * 0.05), # Max 5% odchylenia dla sztywnej aparatury
            'other': float(scale_other * 0.10)    # Max 10% dla reszty
        }

    def fit(self, X, y=None):
        if self.feature_names is None:
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.is_fitted_ = True
        return self

    def _add_noise(self, X):
        X_copy = np.array(X).copy()

        # Dodanie małej wartości, aby uniknąć dzielenia przez zero, upewniamy się, że poprawnie obliczamy odchylenie standardowe dla każdej kolumny, nawet jeśli zawiera NaN
        stds = np.nanstd(X_copy, axis=0) + 1e-5

        for i, col in enumerate(self.feature_names):
            col_lower = col.lower()
            if 'indicator' in col_lower or 'missing' in col_lower:
                continue

            if any(term in col_lower for term in ['gcs', 'apache', 'verbal', 'motor', 'eyes']):
                dynamic_scale = self.noise_scales['gcs'] * stds[i]
                noise = np.random.normal(0, dynamic_scale, X_copy[:, i].shape)
                X_copy[:, i] += noise

                # Zaokrąglenie do wartości całkowitych dla skal dyskretnych
                X_copy[:, i] = np.round(X_copy[:, i])

                # Fizjologiczne przycięcie widełek
                if 'gcs' in col_lower:
                    X_copy[:, i] = np.clip(X_copy[:, i], 3, 15)
                elif 'eyes' in col_lower:
                    X_copy[:, i] = np.clip(X_copy[:, i], 1, 4)
                elif 'verbal' in col_lower:
                    X_copy[:, i] = np.clip(X_copy[:, i], 1, 5)
                elif 'motor' in col_lower:
                    X_copy[:, i] = np.clip(X_copy[:, i], 1, 6)
                elif 'apache' in col_lower:
                    X_copy[:, i] = np.clip(X_copy[:, i], 0, 71)

            elif any(term in col_lower for term in ['bp', 'rate', 'pulse', 'spo2', 'glucose', 'temp']):
                dynamic_scale = self.noise_scales['vitals'] * stds[i]
                noise = np.random.normal(0, dynamic_scale, X_copy[:, i].shape)
                X_copy[:, i] += noise

                # Zabezpieczenie parametrów życiowych (np. brak ujemnego tętna lub SpO2 > 100)
                if 'spo2' in col_lower:
                    X_copy[:, i] = np.clip(X_copy[:, i], 0, 100)
                else:
                    X_copy[:, i] = np.maximum(X_copy[:, i], 0)

            else:
                dynamic_scale = self.noise_scales['other'] * stds[i]
                noise = np.random.normal(0, dynamic_scale, X_copy[:, i].shape)
                X_copy[:, i] += noise

        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self._add_noise(X)

    def transform(self, X):
        if not getattr(self, 'is_fitted_', False):
            raise NotFittedError("Transformator nie został wyuczony.")
        return np.array(X).copy()

    def get_feature_names_out(self, input_features=None):
        return self.feature_names if input_features is None else input_features


class ColumnDropper(BaseEstimator, TransformerMixin):
    """ Usuwa kolumnę z prawdopodobieństwami modelu 1, jeśli istnieje.
        Używana przy przekazywaniu decyzji obu modeli do meta-modelu, więc niepotrzebnie duplikować. """
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        # Informujemy Scikit-Learn, że model został nauczony
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if self.is_fitted_ is not True:
            raise NotFittedError("Ten transformator nie został wyuczony.")

        # Usuwamy kolumnę prawdopodobieństwa, której model główny nie zna
        if isinstance(X, pd.DataFrame) and 'model_1_prob' in X.columns:
            return X.drop(columns=['model_1_prob'])
        return X

    @staticmethod
    def get_feature_names_out(input_features=None):
        if input_features is None:
            return None
        return np.array([f for f in input_features if f != 'model_1_prob'])


class MedicalCascadePredictor(BaseEstimator, ClassifierMixin):
    """ Kaskada Warunkowa z obsługą strefy niepewności. System analizuje kolejne modalności
        tylko wtedy, gdy poprzednie etapy nie dały jednoznacznej pewności diagnostycznej. """
    # Próg odcięcia przesunięty na 0.35, by złapać więcej wczesnych oznak, a strefa weryfikacji powiększona do 15%
    def __init__(self, preprocessor, ensemble, specialist, judge, anomaly_detector=None, review_threshold=0.15, optimal_threshold=0.35):
        self.preprocessor = preprocessor
        self.ensemble = ensemble
        self.specialist = specialist
        self.judge = judge
        self.anomaly_detector = anomaly_detector
        self.review_threshold = review_threshold  # Margines niepewności wokół progu optymalnego
        self.optimal_threshold = float(np.clip(optimal_threshold, 0.05, 0.95)) # Dynamiczny próg kliniczny z zabezpieczeniem
        if hasattr(preprocessor, 'feature_names_in_'):
            self.feature_names_in_ = preprocessor.feature_names_in_
        else:
            self.feature_names_in_ = None

    def predict_proba(self, X_raw, crnn_probs=None, efficientnet_probs=None, densenet_probs=None, n_tta_runs=4):
        if n_tta_runs < 1:
            raise ValueError("n_tta_runs musi być >= 1.")
        if not isinstance(X_raw, pd.DataFrame):
            if self.feature_names_in_ is not None:
                X_raw = pd.DataFrame(X_raw, columns=self.feature_names_in_)
            else:
                X_raw = pd.DataFrame(X_raw)

        prob_main_accum = np.zeros(len(X_raw))
        estimators = getattr(self.ensemble, 'named_estimators_', None)
        base_pred_accum = {}
        base_pred_counts = {}
        num_cols = [c for c in X_raw.select_dtypes(include='number').columns
                    if 'indicator' not in c.lower() and 'missing' not in c.lower()]

        feature_names_out = None
        for i in range(n_tta_runs):
            X_tta = X_raw.copy()
            if i > 0:
                for col in num_cols:
                    noise = np.random.normal(0, 0.015, size=len(X_tta))
                    X_tta[col] *= (1 + noise)

            X_clean = self.preprocessor.transform(X_tta)
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names_out = [name.split('__')[-1] for name in self.preprocessor.get_feature_names_out()]
            else:
                feature_names_out = None
            last_clean_df = pd.DataFrame(X_clean, columns=feature_names_out, index=X_tta.index)
            prob_main_accum += self.ensemble.predict_proba(last_clean_df.values)[:, 1]

            if estimators is not None:
                for name, est in estimators.items():
                    try:
                        estimator_probs = est.predict_proba(last_clean_df.values)[:, 1]
                        if name not in base_pred_accum:
                            base_pred_accum[name] = np.zeros(len(X_raw))
                            base_pred_counts[name] = 0
                        base_pred_accum[name] += estimator_probs
                        base_pred_counts[name] += 1
                    except (ValueError, NotFittedError, AttributeError) as e:
                        logger.warning(f"Błąd pobierania predykcji z modelu bazowego {name}: {e}")
                    except Exception as e:
                        logger.error(f"Nieoczekiwany błąd w modelu bazowym {name}: {e}")

        prob_main = prob_main_accum / n_tta_runs
        final_probs = prob_main.copy()

        base_preds = [
            base_pred_accum[name] / base_pred_counts[name]
            for name in base_pred_accum
            if base_pred_counts[name] > 0
        ]

        # Łączymy wariancję z bezwzględną strefą niepewności, dla której wymagamy dodatkowej opinii eksperckiej
        is_conformal_uncertain = (prob_main > 0.40) & (prob_main < 0.60)
        
        if len(base_preds) > 1:
            preds_matrix = np.column_stack(base_preds)
            preds_std = np.std(preds_matrix, axis=1)

            # Jeśli modele kłócą się ze sobą lub są niepewne
            is_conformal_uncertain = is_conformal_uncertain | (preds_std > 0.15)

        # Zabezpieczenie: IsolationForest z scikit-learn zwraca -1 dla anomalii (outliers) oraz 1 dla inliers
        if self.anomaly_detector is not None:
            is_anomalous = np.asarray(
                self.anomaly_detector.predict(X_clean) == -1,
                dtype=bool
            )
        else:
            is_anomalous = np.zeros(len(X_raw), dtype=bool)

        needs_stage_2 = is_conformal_uncertain | is_anomalous
        prob_spec = prob_main.copy()

        # Definiujemy strefę niepewności dla skanów, w której wymagamy dodatkowej opinii eksperckiej
        if needs_stage_2.any():
            X_hard = X_raw[needs_stage_2].copy()
            X_hard['model_1_prob'] = prob_main[needs_stage_2]

            prob_spec_calc = self.specialist.predict_proba(X_hard)[:, 1]
            prob_spec[needs_stage_2] = prob_spec_calc

            # Bezpieczne ważenie bazujące na obiektywnej pewności modelów
            conf_main = np.abs(prob_main[needs_stage_2] - 0.5)
            conf_spec = np.abs(prob_spec_calc - 0.5)
            
            # Im bardziej pewny specjalista, tym wyższa jego waga
            weight_spec = conf_spec / (conf_main + conf_spec + 1e-6)
            
            final_probs[needs_stage_2] = (
                (1 - weight_spec) * prob_main[needs_stage_2]
                + weight_spec * prob_spec_calc
            )

        is_uncertain_scans = (prob_spec >= 0.35) & (prob_spec <= 0.65)

        def _validate_prob_input(arr, n_samples):
            if arr is None:
                return np.full(n_samples, 0.5), np.zeros(n_samples, dtype=bool)
            arr = np.asarray(arr, dtype=float).reshape(-1)
            if len(arr) != n_samples:
                raise ValueError(f"Długość tablicy prawdopodobieństw musi wynosić {n_samples}.")
            if not np.all(np.isfinite(arr)):
                raise ValueError("Prawdopodobieństwa nie mogą zawierać NaN ani Inf.")
            if np.any((arr < 0.0) | (arr > 1.0)):
                raise ValueError("Prawdopodobieństwa muszą należeć do zakresu [0, 1].")
            return np.clip(arr, 0.0, 1.0), (arr >= 0.0) & (arr <= 1.0)

        n_samples = len(X_raw)
        crnn_arr, _ = _validate_prob_input(crnn_probs, n_samples)
        eff_arr, _ = _validate_prob_input(efficientnet_probs, n_samples)
        dense_arr, _ = _validate_prob_input(densenet_probs, n_samples)

        has_external_scan = (
            crnn_probs is not None
            or efficientnet_probs is not None
            or densenet_probs is not None
        )
        needs_stage_3 = is_uncertain_scans & has_external_scan

        if needs_stage_3.any():
            meta_list = [
                prob_main[needs_stage_3],
                prob_spec[needs_stage_3],
                crnn_arr[needs_stage_3],
                eff_arr[needs_stage_3],
                dense_arr[needs_stage_3],
            ]

            meta_features = np.column_stack(meta_list)
            base_names_meta = ['Konsylium', 'Eksperci', 'CRNN', 'EfficientNet', 'DenseNet']

            meta_features_df = pd.DataFrame(meta_features, columns=base_names_meta, index=X_raw[needs_stage_3].index)

            expected_features = getattr(self.judge, 'n_features_in_', meta_features_df.shape[1])
            if meta_features_df.shape[1] < expected_features:
                padding = np.full((meta_features_df.shape[0], expected_features - meta_features_df.shape[1]), 0.5)
                padding_df = pd.DataFrame(padding, columns=[f'pad_{i}' for i in range(padding.shape[1])], index=meta_features_df.index)
                meta_features_df = pd.concat([meta_features_df, padding_df], axis=1)

            prob_final_hard = self.judge.predict_proba(meta_features_df)[:, 1]
            final_probs[needs_stage_3] = prob_final_hard

        return np.column_stack((1 - final_probs, final_probs))

    def predict(self, X_raw, crnn_probs=None, efficientnet_probs=None, densenet_probs=None):
        preds = self.predict_proba(X_raw, crnn_probs, efficientnet_probs, densenet_probs)[:, 1]
        return (preds >= self.optimal_threshold).astype(int)

    def predict_with_clinical_review(self, X_raw, crnn_probs=None, efficientnet_probs=None, densenet_probs=None):
        probs = self.predict_proba(X_raw, crnn_probs, efficientnet_probs, densenet_probs)[:, 1]
        decisions = []

        for p in probs:
            if abs(p - self.optimal_threshold) <= self.review_threshold:
                decisions.append("Wymagane sprawdzenie stanu pacjenta")
            elif p >= self.optimal_threshold:
                decisions.append("Stan krytyczny")
            else:
                decisions.append("Stabilny")
        return decisions, probs


class CustomClassifierPipeline(ImbPipeline):
    """ Wrapper omijający błąd biblioteki Imbalanced-Learn związanego z brakiem atrybutu _estimator_type w pipeline. """
    @property
    def _estimator_type(self):
        return "classifier"


def train_specialist(X_spec_train, y_spec_train, spec_prep, cv_strategy=3):
    """ Trenuje modele eksperckie 2giej fazy na strefie niepewności. """
    candidate_specialists = {
        'HistGradientExpert': GridSearchCV(
            estimator=HistGradientBoostingClassifier(random_state=CONFIG["SEED"]),
            param_grid={
                'max_iter': [100, 200],
                'max_depth': [3, 5, 7],
                'l2_regularization': [0.1, 1.0, 5.0],
                'learning_rate': [0.01, 0.05]
            },
            cv=cv_strategy, scoring='roc_auc', n_jobs=1
        ),
        'NonLinearSVM': GridSearchCV(
            estimator=SVC(probability=True, class_weight='balanced', random_state=CONFIG["SEED"]),
            param_grid={'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto', 0.01, 0.1]},
            cv=cv_strategy, scoring='roc_auc', n_jobs=1
        ),
        'NonLinearMLP': HalvingGridSearchCV(
            estimator=MLPClassifier(early_stopping=True, random_state=CONFIG["SEED"]),
            param_grid={
                'hidden_layer_sizes': [(32, 16), (64, 32), (32, 16, 8)],
                'alpha': [0.001, 0.01, 0.1],
                'activation': ['relu', 'tanh']
            },
            factor=3,
            cv=cv_strategy,
            scoring='roc_auc',
            n_jobs=1
        ),
        'ClinicalProfileExpert': VotingClassifier(
            estimators=[
                ('logreg_profile', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=CONFIG["SEED"])),
                ('knn_similarity', KNeighborsClassifier(n_neighbors=7, weights='distance'))
            ],
            voting='soft'
        ),
        'PhenotypeDensityExpert': IsolationForest(
            n_estimators=150, 
            contamination='auto', 
            random_state=CONFIG["SEED"]
        )
    }

    calibrated_spec = CalibratedClassifierCV(
        CustomClassifierWrapper(
            spec_prep,
            RandomUnderSampler(random_state=CONFIG["SEED"]),
            PhenotypeTournamentRouter(candidate_specialists, min_clusters=2, max_clusters=4)
        ),
        method='isotonic', cv=3
    )
    calibrated_spec.fit(X_spec_train, y_spec_train)

    return calibrated_spec


def _unsupervised_anomaly_scorer(estimator, X):
    # Pomocnicza funkcja mapująca IsolationForest na prawdopodobieństwa klasyfikacji
    scores = estimator.score_samples(X)
    # Im mniejszy score, tym większa anomalia -> mapujemy na prawdopodobieństwo kryzysu [0, 1]
    probs = 1.0 / (1.0 + np.exp(scores))
    return np.column_stack((1 - probs, probs))