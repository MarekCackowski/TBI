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

# Konfiguracja loggera oraz urządzenia obliczeniowego
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RLAutoAugmentAgent(nn.Module):
    """ Agent z pamięcią akcji: stan to teraz [strata, anatomia, symetria, p_affine, p_elastic, p_blur, p_jitter, p_erase]. 
        Po to, żeby agent mógł uczyć się, które augmentacje są skuteczne w dłuższej perspektywie, a które nie. Jego optymalizacja
        składa się z dwóch nagród: poprawy F1 (task_reward) i zachowania anatomii (struct_reward). W ten sposób agent uczy się
        kompromisu między agresywnymi augmentacjami a zachowaniem biologii w sposób umożliwiający poprawne rozpoznawanie. 
        W przyszłości warto myśleć o wprowadzeniu rywalizującego agenta, który będzie starał się punktowo poprawić jakość ciała modzelowatego,
        które jest dość niewyraźne na skanach CT, a agent główny będzie musiał balansować, żeby zniekształcenia mimo wszystko były akceptowalne. """
    def __init__(self, state_dim=8, action_dim=5, lr=1e-3, struct_lambda=0.2):
        super().__init__()
        self.struct_lambda = struct_lambda

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim * 2) 
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        
        self.last_log_prob = None
        self.last_entropy = None
        self.last_state = None

        # Pamięć startowa (początkowo wyzerowana)
        self.last_actions_memory = np.zeros(action_dim, dtype=np.float32)

    def select_augmentations(self, current_loss, struct_score=1.0, symmetry_score=1.0, deterministic=False):
        # Stan rozszerzony o to, jakich akcji agent użył w poprzedniej epoce
        metrics = [current_loss, struct_score, symmetry_score]
        state_list = metrics + self.last_actions_memory.tolist()
        
        state = torch.tensor([state_list], dtype=torch.float32, device=DEVICE)
        self.last_state = state
        
        params = self.actor(state)
        
        alpha = func.softplus(params[:, :5]) + 1.0
        beta = func.softplus(params[:, 5:]) + 1.0
        dist = Beta(alpha, beta)
        
        if deterministic:
            actions = alpha / (alpha + beta)
        else:
            actions = dist.sample()
            self.last_log_prob = dist.log_prob(actions).sum(dim=-1)
            self.last_entropy = dist.entropy().sum(dim=-1)
            
        actions_np = actions.detach().cpu().numpy()[0]
        
        # Zapisanie akcji do pamięci na potrzeby kolejnego kroku
        self.last_actions_memory = actions_np
        
        return actions_np


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
        stds = np.std(X_copy, axis=0) + 1e-5
        
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
        self.is_fitted_ = None

    def fit(self, X, y=None):
        # Informujemy Scikit-Learn, że model został nauczony
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not hasattr(self, 'is_fitted_'):
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
    def __init__(self, preprocessor, ensemble, specialist, judge, anomaly_detector=None, review_threshold=0.05, optimal_threshold=0.5):
        self.preprocessor = preprocessor
        self.ensemble = ensemble
        self.specialist = specialist
        self.judge = judge
        self.anomaly_detector = anomaly_detector 
        self.review_threshold = review_threshold  # Margines niepewności wokół progu optymalnego
        self.optimal_threshold = optimal_threshold # Dynamiczny próg kliniczny
        self.feature_names_in = preprocessor.feature_names_in_

    def predict_proba(self, X_raw, crnn_probs=None, efficientnet_probs=None, densenet_probs=None, n_tta_runs=4):
        if not isinstance(X_raw, pd.DataFrame):
            X_raw = pd.DataFrame(X_raw, columns=self.feature_names_in)

        prob_main_accum = np.zeros(len(X_raw))
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
            feature_names_out = [name.split('__')[-1] for name in self.preprocessor.get_feature_names_out()]
            X_clean_df = pd.DataFrame(X_clean, columns=feature_names_out, index=X_tta.index)
            prob_main_accum += self.ensemble.predict_proba(X_clean_df.values)[:, 1]
            
        prob_main = prob_main_accum / n_tta_runs
        final_probs = prob_main.copy()

        base_preds = []
        estimators = getattr(self.ensemble, 'named_estimators_', None)
        if estimators is not None:
            for name, est in estimators.items():
                try:
                    base_preds.append(est.predict_proba(X_clean_df.values)[:, 1])
                except (ValueError, NotFittedError, AttributeError) as e:
                    logger.warning(f"Błąd pobierania predykcji z modelu bazowego {name}: {e}")
                    pass
                except Exception as e:
                    logger.error(f"Nieoczekiwany błąd w modelu bazowym {name}: {e}")
                    pass
        
        # Jeśli modele w Konsylium nie zgadzają się ze sobą, wariancja jest wysoka
        if len(base_preds) > 1:
            preds_matrix = np.column_stack(base_preds)
            preds_std = np.std(preds_matrix, axis=1)
            
            # Pacjent niepewny, jeśli modele kłócą się ze sobą powyżej dynamicznego odchylenia 0.15
            is_conformal_uncertain = (preds_std > 0.15) 
        else:
            is_conformal_uncertain = (prob_main > 0.40) & (prob_main < 0.60)

        if self.anomaly_detector is not None:
            X_clean_base = pd.DataFrame(self.preprocessor.transform(X_raw), columns=feature_names_out, index=X_raw.index)
            is_anomalous = (self.anomaly_detector.predict(X_clean_base) == 1)
        else:
            is_anomalous = pd.Series(False, index=X_raw.index)

        needs_stage_2 = is_conformal_uncertain | is_anomalous
        prob_spec = prob_main.copy()

        if needs_stage_2.any():
            X_hard = X_raw[needs_stage_2].copy()
            X_hard['model_1_prob'] = prob_main[needs_stage_2]
            
            prob_spec_calc = self.specialist.predict_proba(X_hard)[:, 1]
            prob_spec[needs_stage_2] = prob_spec_calc
            
            diff = prob_main[needs_stage_2] - 0.5
            weight = 1 / (1 + np.exp(-10 * diff)) 
            final_probs[needs_stage_2] = (1 - weight) * prob_main[needs_stage_2] + weight * prob_spec_calc

        is_uncertain_scans = (prob_spec > 0.40) & (prob_spec < 0.60)
        needs_stage_3 = needs_stage_2 & is_uncertain_scans

        if needs_stage_3.any():
            if crnn_probs is None: crnn_probs = np.full_like(prob_main, 0.5)
            if efficientnet_probs is None: efficientnet_probs = np.full_like(prob_main, 0.5)
            if densenet_probs is None: densenet_probs = np.full_like(prob_main, 0.5)

            meta_list = [
                prob_main[needs_stage_3], 
                prob_spec[needs_stage_3], 
                crnn_probs[needs_stage_3], 
                efficientnet_probs[needs_stage_3],
                densenet_probs[needs_stage_3]
            ]
            
            meta_features = np.column_stack(meta_list)
            base_names_meta = ['Konsylium', 'Eksperci', 'CRNN', 'EfficientNet', 'DenseNet']
            
            # Przekazanie DataFrame pozwala EBM na poprawne mapowanie nazw cech
            meta_features_df = pd.DataFrame(meta_features, columns=base_names_meta, index=X_raw[needs_stage_3].index)
            
            expected_features = getattr(self.judge, 'n_features_in_', meta_features_df.shape[1])
            if meta_features_df.shape[1] < expected_features:
                padding = np.full((meta_features_df.shape[0], expected_features - meta_features_df.shape[1]), 0.5)
                padding_df = pd.DataFrame(padding, columns=[f'pad_{i}' for i in range(padding.shape[1])], index=meta_features_df.index)
                meta_features_df = pd.concat([meta_features_df, padding_df], axis=1)
                
            # EBM przyjmuje surowe prawdopodobieństwa i optymalizuje wagi
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
            # Szara strefa jest teraz centrowana wokół optymalnego progu klinicznego
            if abs(p - self.optimal_threshold) <= self.review_threshold:
                decisions.append("Wymagane sprawdzenie stanu pacjenta")
            elif p >= self.optimal_threshold:
                decisions.append("Stan krytyczny")
            else:
                decisions.append("Stabilny")
        return decisions, probs


# Wrapper omijający błąd biblioteki Imbalanced-Learn
class CustomClassifierPipeline(ImbPipeline):
    @property
    def _estimator_type(self):
        return "classifier"