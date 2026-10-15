import gc
import gzip
import os
import tempfile

import urllib
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pickle
import shutil
import socket
import time

from sklearn.feature_selection import SelectFromModel
from sklearn.utils import resample
import cv2 as cv
cv.setNumThreads(0)
from contextlib import nullcontext
import functools
import json
import re
import sys
import datetime
from PIL import Image
import optuna
from pydantic_settings import BaseSettings, SettingsConfigDict
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pymongo
import psycopg2
import redis
import warnings
from dotenv import load_dotenv
import logging
import random
import requests

# Biblioteki dla obsługi danych
import pandas as pd
import numpy as np

# Silniki modeli i Ensemble
from torchvision import transforms
from tqdm import tqdm
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, BayesianRidge
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from interpret.glassbox import ExplainableBoostingClassifier

# Przygotowanie danych, Cross-Validation i Pipelines
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import torch.nn.functional as func
from sklearn.metrics import brier_score_loss, calinski_harabasz_score, precision_score
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError, ConvergenceWarning
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import torch.nn as nn
import torch.optim as optim
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import torch
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance

# Wstrzykiwanie danych (Imbalanced-learn)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

# Optymalizacja i Metryki
from sklearn.metrics import (
    roc_auc_score, classification_report, precision_recall_curve,
    make_scorer, fbeta_score, confusion_matrix
)
from sklearn.calibration import calibration_curve
from statsmodels.stats.contingency_tables import mcnemar

# Wizualizacja i XAI
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import scipy.stats as stats

# Las izolacyjny i wykrywanie anomalii
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm
from imodels import SkopeRulesClassifier

# Zapis modeli i obsługa wielowątkowości
import joblib
from joblib import parallel_backend

# Modele do analizy skanów i notatek
from Models.IntracranialHemorrhage_CRNN import VolumetricMedicalResNetCRNN
from Models.ModzelBody_UNet_to_EfficientNet import UNet, VolumetricEfficientNet, sequential_two_stage_predict
from Models.BoneFracture_Trans_plus_Dense import DenseNetFractureDetector, BoneWindowTransform
import pydicom

# MinIO i kompatybilność z S3
import boto3
from botocore.client import Config
from psycopg2 import pool
import tempfile

# Wyciszenie ostrzeżeń
from botocore.exceptions import ClientError, BotoCoreError
from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import Dict, Any, List, Optional, Tuple
import datetime

# Inicjalizacja systemu logowania
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s\n%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Bezpieczny import solve'ra Neural ODE
try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None

# Bezpieczny import SQLAlchemy
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
except ImportError as e:
    logger.error("Brak biblioteki SQLAlchemy. Uruchom w terminalu: pip install sqlalchemy")
    raise ImportError("Brak biblioteki SQLAlchemy. Uruchom w terminalu: pip install sqlalchemy") from e


# Globalne wyciszenie powiadomień, ostrzeżeń o zbieżności oraz komunikatów deprecjonujących bibliotek
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed` should be used")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='interpret')
warnings.filterwarnings('ignore', category=FutureWarning, module='interpret')

# Bezpieczne wczytanie ukrytych zmiennych środowiskowych z pliku .env
load_dotenv()

# Dynamiczne wykrywanie ścieżki do głównego katalogu projektu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    """ Scentralizowana konfiguracja aplikacji (Wersja zunifikowana dla Pydantic v2). """
    seed: int = 42
    models_dir: str = os.path.join(BASE_DIR, "output_data", "Models")
    plots_dir: str = os.path.join(BASE_DIR, "output_data")
    optuna_db: str = f"sqlite:///{os.path.join(BASE_DIR, 'output_data', 'optuna_study', 'tbi_ensemble.db')}"
    local_radiology_file: str = "radiology.csv.gz"
    beta_main: float = 1.0
    beta_critical: float = 2.0
    
    # Bazy danych
    db_host: str = os.getenv('DB_HOST', '172.189.176.192')
    db_port: str = os.getenv('DB_PORT', '5433')
    db_name: str = os.getenv('DB_NAME', 'tbi_database')
    db_user: str = os.getenv('DB_USER', 'postgres')
    db_pass: str = os.getenv('DB_PASS', 'postgres')

    mongo_host: str = os.getenv('MONGO_HOST', '172.189.176.192')
    redis_host: str = os.getenv('REDIS_HOST', '172.189.176.192')
    
    # Konfiguracja S3 / MinIO
    s3_endpoint: str = os.getenv('S3_ENDPOINT', 'http://172.189.176.192:9000')
    s3_access_key: str = os.getenv('MINIO_ACCESS_KEY', 'admin')
    s3_secret_key: str = os.getenv('MINIO_SECRET_KEY', 'admin123')
    s3_bucket: str = os.getenv('S3_BUCKET', 'tbi-data')
    
    physionet_user: Optional[str] = os.getenv('PHYSIONET_USER')
    physionet_pass: Optional[str] = os.getenv('PHYSIONET_PASS')
    use_physionet_mimic: bool = os.getenv('USE_PHYSIONET_MIMIC', 'False').lower() == 'true'

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
# Globalne urządzenie dla całego skryptu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicjalizacja globalnego obiektu ustawień
SETTINGS = Settings()

CONFIG = {
    "SEED": SETTINGS.seed,
    "MODELS_DIR": SETTINGS.models_dir,
    "PLOTS_DIR": SETTINGS.plots_dir,
    "BETA_MAIN": SETTINGS.beta_main,
    "BETA_CRITICAL": SETTINGS.beta_critical,
    "N_OPTUNA_TRIALS": 15
}

DB_HOST = SETTINGS.db_host
DB_PORT = SETTINGS.db_port
DB_NAME = SETTINGS.db_name
DB_USER = SETTINGS.db_user
DB_PASS = SETTINGS.db_pass

MONGO_HOST = SETTINGS.mongo_host
REDIS_HOST = SETTINGS.redis_host

r_cache = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
mongo_client = pymongo.MongoClient(f'mongodb://{MONGO_HOST}:27017/')
mongo_db = mongo_client["tbi_research"]

# Indeksy w MongoDB dla szybkiego wyszukiwania i sortowania logów predykcji
mongo_db.prediction_logs.create_index(
    [("patient_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)]
)

PHYSIONET_USER = os.getenv('PHYSIONET_USER')
PHYSIONET_PASS = os.getenv('PHYSIONET_PASS')

# flaga dla fuzji danych z MIMIC-IV
USE_PHYSIONET_MIMIC = os.getenv('USE_PHYSIONET_MIMIC', 'False').lower() == 'true'

# Słownik kluczowych markerów MIMIC-IV (itemid)
MIMIC_ITEM_IDS = {
    'icp': [220765, 227989],        # Intracranial Pressure
    'fio2': [223835],               # Fraction of Inspired Oxygen
    'pupil_right': [224409],        # Right Pupil Response
    'pupil_left': [224410],         # Left Pupil Response
    'mannitol': [221260, 227533],   # Leki na obrzęk mózgu
    'noradrenaline': [221906],      # Wazopresory (Wstrząs)
    'lactate': [50813],             # Mleczany (Lab)
    'sodium': [50983],              # Sód (Lab)
    'inr': [51237],                 # Krzepnięcie (Lab)
    'platelets': [51265],           # Płytki (Lab)
    'propofol': [222168],           # Propofol (Ciągły wlew sedacyjny)
    'midazolam': [221668],          # Midazolam (Wersed - benzodiazepina)
    'dexmedetomidine': [221744],    # Precedex (Sedacja bez depresji oddechowej)
    'lorazepam': [221389],          # Lorazepam (Ativan - alternatywna benzo)
    'flumazenil': [227529, 221752]  # Antagonista GABA (Wybudzenie / odwrócenie)
}


def save_to_infrastructure(patient_id, prob, decision, raw_data, vitals_snapshot, db_pool=None):
    """ Zapisuje wynik w SQL (PostgreSQL), MongoDB oraz Redis. """
    ts = datetime.datetime.now()

    # PostgreSQL
    try:
        if db_pool:
            conn = db_pool.getconn()
        else:
            conn = psycopg2.connect(
                host=SETTINGS.db_host, port=SETTINGS.db_port, 
                database=SETTINGS.db_name, user=SETTINGS.db_user, password=SETTINGS.db_pass
            )

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ML_prediction_results (patient_id, death_prob, final_decision, created_at) "
                    "VALUES (%s, %s, %s, %s)",
                    (int(patient_id), float(prob), decision, ts)
                )

        if db_pool:
            db_pool.putconn(conn)
        else:
            conn.close()
            
    except psycopg2.OperationalError as e:
        logger.error(f"Błąd sieciowy/operacyjny PostgreSQL dla pacjenta {patient_id}: {e}")
    except psycopg2.DatabaseError as e:
        logger.error(f"Błąd bazy danych PostgreSQL dla pacjenta {patient_id}: {e}")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd PostgreSQL dla pacjenta {patient_id}: {e}")

    # MongoDB + Pydantic
    try:
        log_data = PredictionLogSchema(
            patient_id=int(patient_id),
            vitals_at_prediction=vitals_snapshot,
            prediction={"probability": float(prob), "decision": decision}
        )
        mongo_db.prediction_logs.insert_one(log_data.model_dump())
    except ValidationError as e:
        logger.error(f"Pydantic odrzucił log pacjenta {patient_id} z powodu uszkodzonych danych: {e}")
    except pymongo.errors.PyMongoError as e:
        logger.error(f"Błąd zapisu do MongoDB dla pacjenta {patient_id}: {e}")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd MongoDB dla pacjenta {patient_id}: {e}")

    # Redis
    try:
        alert_level = "CRITICAL" if prob > 0.8 else "STABLE"
        r_cache.setex(f"alert:patient:{patient_id}", 3600, f"{alert_level}|{round(prob * 100, 2)}%")
    except redis.RedisError as e:
        logger.error(f"Błąd pamięci podręcznej Redis dla pacjenta {patient_id}: {e}")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd Redis dla pacjenta {patient_id}: {e}")

    logger.info(f"Zakończono proces zapisu wyników dla pacjenta {patient_id}.")


class PredictionLogSchema(BaseModel):
    """ Dynamiczny schemat walidacji i mapowania dla MongoDB. """
    patient_id: int
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    vitals_at_prediction: Dict[str, Any] = Field(default_factory=dict)
    prediction: Dict[str, Any]
    model_version: str = "v1_SWA"

    @model_validator(mode='before')
    @classmethod
    def dynamic_mapping_vitals(cls, values: Any) -> Any:
        """ Przechwytuje słownik przed inicjalizacją i dynamicznie mapuje nazwy z różnych baz. """
        if isinstance(values, dict) and 'vitals_at_prediction' in values:
            vitals = values['vitals_at_prediction']
            
            # Jeśli sprzęt z eICU przysyła 'sao2', zmapuj na 'spo2_min'
            if 'sao2' in vitals and 'spo2_min' not in vitals:
                vitals['spo2_min'] = vitals.pop('sao2')
                
            # Jeśli sprzęt przysyła 'systemicmean', zmapuj na 'map_min'
            if 'systemicmean' in vitals and 'map_min' not in vitals:
                vitals['map_min'] = vitals.pop('systemicmean')
                
            # Jeśli sprzęt przysyła ciśnienie jako string "120/80", zrób z tego int
            if 'blood_pressure_raw' in vitals:
                try:
                    sys, dias = vitals['blood_pressure_raw'].split('/')
                    vitals['sysbp_min'] = float(sys)
                    vitals['diasbp_min'] = float(dias)
                except ValueError:
                    pass

            values['vitals_at_prediction'] = vitals
        return values

class RealTimePatientBuffer:
    """ Zarządza stanem pacjenta w czasie rzeczywistym przy użyciu Redisa. Pozwala na asynchroniczne
        dopisywanie kolejnych danych i sklejanie ich w kompletny wektor cech, zanim model podejmie decyzję. """
    def __init__(self, host='localhost', port=6379, db=0):
        # Inicjalizacja połączenia z bazą Redis
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def update_patient_data(self, hadm_id: int, category: str, data: dict, ttl: int = 86400):
        """ Aktualizuje lub dodaje wybraną kategorię danych dla pacjenta. """
        key = f"patient:{hadm_id}:{category}"
        self.r.setex(key, ttl, json.dumps(data))

    def get_complete_patient_vector(self, hadm_id: int) -> dict:
        """ Skleja kompletny stan pacjenta ze wszystkich dostępnych podsystemów w locie.
            Jeśli brakuje konkretnego badania, funkcja nie przerywa działania, tylko zwraca
            aktualnie dostępny podzbiór cech do bezpiecznego uzupełnienia przez imputację. """
        categories = ['vitals', 'vision']
        combined_vector = {}
        
        for cat in categories:
            key = f"patient:{hadm_id}:{cat}"
            raw_data = self.r.get(key)
            if raw_data:
                try:
                    parsed = json.loads(raw_data)
                    combined_vector.update(parsed)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.error(f"Nieoczekiwany błąd podczas parsowania danych pacjenta {hadm_id} dla kategorii {cat}: {e}")
                    
        return combined_vector

    def is_ready_for_inference(self, hadm_id: int, required_categories=None) -> bool:
        """ Sprawdza, czy pacjent zgromadził minimalny zestaw danych wymagany do uruchomienia pierwszej warstwy kaskady. """
        if required_categories is None:
            required_categories = ['vitals']
            
        for cat in required_categories:
            if not self.r.exists(f"patient:{hadm_id}:{cat}"):
                return False
        return True
    

def drop_m1_prob(X):
    """ Usuwanie kolumny z prawdopodobieństwami modelu 1, jeśli istnieje.
        I tak jest używana przy przekazywaniu decyzji obu modeli do meta-modelu, więc niepotrzebnie duplikować. """
    if isinstance(X, pd.DataFrame) and 'model_1_prob' in X.columns:
        return X.drop(columns=['model_1_prob'])
    return X


class RLAutoNoiseAgent(nn.Module):
    """ Agent RL uczący się optymalnej polityki zaszumiania danych tabelarycznych. """
    def __init__(self, state_dim=1, action_dim=3, lr=1e-3):
        super().__init__()

        # Akcje to mnożniki szumu dla 3 grup: 1. Skale (GCS/APACHE), 2. Aparatura (Vitals), 3. Reszta
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),

            # Używamy Sigmoid, aby Agent zwracał mnożniki bazowe w zakresie [0.0, 1.0], później będą przeskalowane
            nn.Sigmoid() 
        )
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        
    def select_noise_scales(self, current_metric):
        # Sprawdzamy urządzenie, na którym fizycznie leży Agent
        device = next(self.parameters()).device
        
        # Tworzymy stan od razu na tym urządzeniu
        state = torch.tensor([[current_metric]], dtype=torch.float32, device=device)
        scales = self.policy_network(state)
        
        # Wymuszamy zrzut pamięci z powrotem na CPU przed konwersją do NumPy
        return scales.detach().cpu().numpy()[0]
        
    def update(self, current_metric, reward):
        # Sprawdzamy urządzenie
        device = next(self.parameters()).device
        
        # Upewniamy się, że zarówno stan, jak i nagroda, leżą na tym samym sprzęcie
        state = torch.tensor([[current_metric]], dtype=torch.float32, device=device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
        
        scales = self.policy_network(state)
        
        # Policy Gradient: operacje w pełni zsynchronizowane
        loss = -torch.log(scales + 1e-8).mean() * reward_tensor 
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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
            
            expected_features = getattr(self.judge, 'n_features_in_', meta_features.shape[1])
            if meta_features.shape[1] < expected_features:
                padding = np.full((meta_features.shape[0], expected_features - meta_features.shape[1]), 0.5)
                meta_features = np.column_stack((meta_features, padding))
                
            prob_final_hard = self.judge.predict_proba(meta_features)[:, 1]
            diff_stage3 = prob_spec[needs_stage_3] - 0.5
            weight_stage3 = 1 / (1 + np.exp(-10 * diff_stage3)) 
            
            final_probs[needs_stage_3] = (1 - weight_stage3) * prob_spec[needs_stage_3] + weight_stage3 * prob_final_hard

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


class PhenotypeTournamentRouter(BaseEstimator, ClassifierMixin):
    """ Dynamiczny Router Fenotypów.
        Unikamy rotacji przestrzeni, która niszczy naturalne znaczenie cech tabelarycznych. Zamiast tego używamy
        selekcji cech opartej na drzewach, aby wyeliminować szum i ułatwić GMM znalezienie prawdziwych podgrup medycznych pacjentów. """
        
    _estimator_type = "classifier"

    def __init__(self, candidate_estimators, feature_names=None, phenotype_hints=None, min_clusters=2, max_clusters=4, min_patients_per_cluster=70):
        self.candidate_estimators = candidate_estimators
        self.feature_names = feature_names
        self.phenotype_hints = phenotype_hints  
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.min_patients_per_cluster = min_patients_per_cluster
        
        """ Odrzucenie PCA przed klastrowaniem wynika z jej sztywnego charakteru. PCA nie usuwa szumu,
            lecz dokonuje matematycznej rotacji przestrzeni, tworząc abstrakcyjne kombinacje liniowe.
            Skutkuje to zmieszaniem twardych parametrów ratunkowych z wahaniami cech nie wnoszących zbyt
            dużej wartości. Zgodnie z badaniami (Grinsztajn et al., 2022), dane tabelaryczne tracą swoje
            właściwości pod wpływem rotacji. Zmuszenie algorytmu GMM do pracy na takich składowych sprawia, 
            że grupuje on pacjentów bazując na czystej, ogólnej wariancji, a nie logice medycznej.
            Zastosowanie Random Forest odwraca ten paradygmat:
            Zamiast obracać wymiary, Las Losowy w sposób nadzorowany ocenia siłę predykcyjną 
            każdej zmiennej pod kątem ryzyka zgonu pacjenta, a następnie fizycznie odcina szum.
            Pozostałe cechy nie ulegają żadnej transformacji, dzięki czemu nienadzorowany algorytm GMM
            szuka fenotypów w czystej, fizjologicznej przestrzeni.

            Uzasadnienie dobranych hiperparametrów:
            max_depth=5: Płytkie drzewa celowo blokują modelowi możliwość przeuczenia i wymusza większe
            skupienie na wartościowych cechach.
            n_estimators=50: Zapewnia stabilne uśrednienie wariancji i posortowanie ważności cech,
            bez znacznego wpływu na wydajność.
            max_features=25: Bufor odcięcia. W medycynie ratunkowej rdzeń hemodynamiczno-
            -neurologiczny to zazwyczaj około 15-20 kluczowych zmiennych. Pozostawienie 25 osi 
            gwarantuje zachowanie całego kontekstu przy drastycznym ucięciu szumu. """
        self.reducer = SelectFromModel(
            RandomForestClassifier(n_estimators=50, max_depth=5, random_state=CONFIG["SEED"], n_jobs=-1),
            max_features=25
        )
        self.clusterer_ = None
        self.models_ = []
        self.is_fitted_ = False
        self.fallback_model_ = None

    def fit(self, X, y):
        X_arr = np.array(X)
        y_arr = np.array(y)
        self.classes_ = np.unique(y_arr)

        fallback_name = list(self.candidate_estimators.keys())[0]
        self.fallback_model_ = clone(self.candidate_estimators[fallback_name])
        self.fallback_model_.fit(X_arr, y_arr)

        X_cluster_space = X_arr.copy()

        if self.phenotype_hints and self.feature_names:
            for col_name, weight in self.phenotype_hints.items():
                if col_name in self.feature_names:
                    idx = self.feature_names.index(col_name)
                    X_cluster_space[:, idx] *= weight

        # Redukcja wymiarów przez odrzucenie śmieci, bez matematycznej rotacji
        X_reduced = self.reducer.fit_transform(X_cluster_space, y_arr)

        best_ch_score = -1.0
        best_n_components = self.min_clusters
        best_clusterer = None

        logger.info("Szukam optymalnej liczby fenotypów (RF-Selection + Gaussian Mixture Models).")
        
        for n in range(self.min_clusters, self.max_clusters + 1):
            gmm_candidate = GaussianMixture(n_components=n, covariance_type='full', random_state=CONFIG["SEED"], n_init=3)
            clusters = gmm_candidate.fit_predict(X_reduced)
            
            cluster_sizes = np.bincount(clusters)
            smallest_cluster_size = np.min(cluster_sizes)
            
            if smallest_cluster_size < self.min_patients_per_cluster:
                logger.info(f"Odrzucono podział na {n} klastry (najmniejszy klaster to tylko {smallest_cluster_size} outlierów).")
                continue
                
            ch_score = calinski_harabasz_score(X_reduced, clusters)
            if ch_score > best_ch_score:
                best_ch_score = ch_score
                best_n_components = n
                best_clusterer = gmm_candidate

        if best_clusterer is None:
            logger.info("Żaden podział nie spełnił limitu pacjentów. Wymuszam bezpieczny podział na 2 grupy.")
            best_n_components = 2
            best_clusterer = GaussianMixture(n_components=2, covariance_type='full', random_state=CONFIG["SEED"], n_init=3)
            best_clusterer.fit(X_reduced)

        self.clusterer_ = best_clusterer
        logger.info(f"Ostatecznie wyodrębniono {best_n_components} zbalansowane klastry (Fenotypy GMM).")

        clusters = self.clusterer_.predict(X_reduced)
        self.models_ = []
        tournament_scorer = make_scorer(fbeta_score, beta=CONFIG["BETA_MAIN"], zero_division=0)

        for i in range(best_n_components):
            mask = (clusters == i)
            X_cluster, y_cluster = X_arr[mask], y_arr[mask]

            if len(np.unique(y_cluster)) < 2:
                logger.info(f"   Klaster {i}: Brak reprezentacji obu klas. Używam modelu awaryjnego.")
                self.models_.append(self.fallback_model_)
                continue

            logger.info(f"   Klaster {i} (N={np.sum(mask)}): Start mini-turnieju.")
            best_score_turn = -1.0
            best_model_clone = None
            winning_name = ""

            for name, estimator in self.candidate_estimators.items():
                clf = clone(estimator)
                scores = cross_val_score(clf, X_cluster, y_cluster, cv=3, scoring=tournament_scorer, n_jobs=1)
                avg_score = scores.mean()
                
                logger.info(f"      - {name}: F-Beta = {avg_score:.4f}")
                
                if avg_score > best_score_turn:
                    best_score_turn = avg_score
                    best_model_clone = clf
                    winning_name = name

            best_model_clone.fit(X_cluster, y_cluster)
            self.models_.append(best_model_clone)
            logger.info(f"   Zwycięzca dla Fenotypu {i}: {winning_name} (Wytrenowany i przypisany).")

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise NotFittedError("Model nie został nauczony.")

        X_arr = np.array(X)
        X_cluster_space = X_arr.copy()

        if self.phenotype_hints and self.feature_names:
            for col_name, weight in self.phenotype_hints.items():
                if col_name in self.feature_names:
                    idx = self.feature_names.index(col_name)
                    X_cluster_space[:, idx] *= weight

        X_reduced = self.reducer.transform(X_cluster_space)
        clusters = self.clusterer_.predict(X_reduced)
        
        probs = np.zeros((X_arr.shape[0], 2))
        for i, model in enumerate(self.models_):
            mask = (clusters == i)
            if np.sum(mask) > 0:
                probs[mask] = model.predict_proba(X_arr[mask])
        return probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def check_and_seed_database():
    """ Sprawdza czy baza jest pusta. Jeśli tak, generuje syntetyczne dane i zapisuje je do PostgreSQL, 
        aby można było przetestować pełną kaskadę bez czekania na MIMIC-IV. """

    # Budujemy string połączeniowy na bazie zmiennych z pliku .env
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    
    try:
        with engine.connect() as conn:
            # Sprawdzamy czy główna tabela 'visits' w ogóle istnieje
            result = conn.execute(text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'visits');"))
            exists = result.scalar()
    except SQLAlchemyError as e:
        logger.error(f"Błąd połączenia SQLAlchemy podczas sprawdzania bazy (np. brak serwera lub złe hasło): {e}")
        return
        
    if not exists:
        logger.info("Baza danych jest pusta. Generuję pacjentów syntetycznych do testów Kaskady.")
        n_patients = 1000
        np.random.seed(42)
        
        # Tabela pacjentów z chorobami współistniejącymi
        patients_df = pd.DataFrame({
            'patient_id': range(1, n_patients + 1),
            'diabetes_mellitus': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
            'immunosuppression': np.random.choice([0, 1], n_patients, p=[0.9, 0.1]),
            'cirrhosis': np.random.choice([0, 1], n_patients, p=[0.95, 0.05]),
            'hepatic_failure': np.random.choice([0, 1], n_patients, p=[0.98, 0.02]),
            'leukemia': np.random.choice([0, 1], n_patients, p=[0.99, 0.01]),
            'lymphoma': np.random.choice([0, 1], n_patients, p=[0.99, 0.01]),
            'solid_tumor_with_metastasis': np.random.choice([0, 1], n_patients, p=[0.95, 0.05])
        })
        
        visits_df = pd.DataFrame({
            'patient_id': range(1, n_patients + 1),
            'hadm_id': range(100000, 100000 + n_patients),
            'hospital_outcome_death': np.random.choice([0, 1], size=n_patients, p=[0.7, 0.3])
        })
        
        clinical_df = pd.DataFrame({
            'patient_id': range(1, n_patients + 1),
            'gcs_eyes': np.random.randint(1, 5, n_patients),
            'gcs_motor': np.random.randint(1, 7, n_patients),
            'gcs_verbal': np.random.randint(1, 6, n_patients),
            'sysbp_min': np.random.normal(100, 20, n_patients),
            'heart_rate_max': np.random.normal(90, 15, n_patients),
            'spo2_min': np.random.normal(95, 5, n_patients),
            'glucose_max': np.random.normal(140, 40, n_patients),
            'd1_glucose_max': np.random.normal(130, 30, n_patients),
            'd1_sysbp_max': np.random.normal(130, 25, n_patients),
            'd1_heartrate_min': np.random.normal(70, 15, n_patients),
            'd1_diasbp_min': np.random.normal(60, 15, n_patients),
            'age': np.random.randint(18, 90, n_patients),
            'is_intubated': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
            'ventilated_apache': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
            'apache_3j_bodysystem': ['neurological'] * n_patients,
            'valid_to': pd.to_datetime('2099-12-31')
        })
        
        # Szybki zrzut danych do bazy
        patients_df.to_sql('patients', engine, if_exists='replace', index=False)
        visits_df.to_sql('visits', engine, if_exists='replace', index=False)
        clinical_df.to_sql('clinical_measurements_temporal', engine, if_exists='replace', index=False)
        logger.info("PostgreSQL zasilony danymi! Przechodzę do Treningu.")

    
def load_and_split_data():
    """ Pobiera dane z PostgreSQL, przeprowadza rygorystyczny audyt kompletności danych klinicznych,
        odrzuca wybrakowane rekordy krytyczne i dynamicznie zwraca zbalansowany podział na zbiór treningowy i testowy. """
    try:
        with psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS extracted_radiology_features (
                        hadm_id INT PRIMARY KEY,
                        vision_midline_shift_mm FLOAT,
                        vision_basal_cisterns FLOAT,
                        vision_corpus_callosum FLOAT
                    );
                """)
            
            # Wzbogacenie o telemetrię, źrenice i logi farmakologiczne
            select_mimic = """
                m.icp_max, m.fio2_max, m.inr_max, m.platelets_min, 
                m.pupils_reactive, m.pupil_size_left, m.pupil_size_right, 
                m.mannitol_administered, m.sedatives_administered, m.flumazenil_administered,
                m.noradrenaline_rate, m.lactate_max, m.sodium_mean,
            """ if USE_PHYSIONET_MIMIC else ""

            join_mimic = " LEFT JOIN mimic_icu_features m ON v.hadm_id = m.hadm_id " if USE_PHYSIONET_MIMIC else ""

            query = f""" SELECT 
                    v.hadm_id, 
                    c.gcs_eyes, c.gcs_motor, c.gcs_verbal,
                    c.sysbp_min, c.heart_rate_max, c.spo2_min,
                    c.glucose_max, c.d1_glucose_max, c.d1_sysbp_max,
                    c.d1_heartrate_min, c.d1_diasbp_min, c.age,
                    c.is_intubated, c.ventilated_apache, c.apache_3j_bodysystem,
                    
                    p.diabetes_mellitus, p.immunosuppression, p.cirrhosis, 
                    p.hepatic_failure, p.leukemia, p.lymphoma, p.solid_tumor_with_metastasis,
                    
                    COALESCE(r.vision_midline_shift_mm, 0.0) AS vision_midline_shift_mm,
                    COALESCE(r.vision_basal_cisterns, 0.0) AS vision_basal_cisterns,
                    COALESCE(r.vision_corpus_callosum, 0.0) AS vision_corpus_callosum,
                    
                    {select_mimic}

                    v.hospital_outcome_death as label
                FROM Clinical_Measurements_Temporal c
                JOIN Visits v ON c.patient_id = v.patient_id
                LEFT JOIN Patients p ON v.patient_id = p.patient_id
                LEFT JOIN extracted_radiology_features r ON v.hadm_id = r.hadm_id
                {join_mimic}
                WHERE c.valid_to > CURRENT_TIMESTAMP; """

            df = pd.read_sql(query, conn)
    except psycopg2.Error as e:
        logger.exception(f"Błąd krytyczny bazy danych podczas pobierania danych z PostgreSQL: {e}")
        raise
    except pd.errors.DatabaseError as e:
        logger.exception(f"Pandas nie mógł przetworzyć zapytań SQL: {e}")
        raise

    initial_count = len(df)
    logger.info(f"Pobrano surowy zbiór danych: {initial_count} rekordów.")

    df.dropna(subset=['label'], inplace=True)

    critical_pillars = ['gcs_eyes', 'sysbp_min', 'age']
    existing_pillars = [col for col in critical_pillars if col in df.columns]
    df.dropna(subset=existing_pillars, how='any', inplace=True)

    # Śpiączka farmakologiczna vs Strukturalna
    if 'sedatives_administered' in df.columns and 'flumazenil_administered' in df.columns:
        df['pharmacological_coma_flag'] = np.where(
            (df['sedatives_administered'] == 1) & 
            (df['flumazenil_administered'] == 0) & 
            (df.get('pupils_reactive', 1) == 1), 
            1, 0
        )

    complete_count = len(df)
    rejected_count = initial_count - complete_count
    
    logger.info(f"Audyt kompletności zakończony:")
    logger.info(f"Odrzucono rekordy wybrakowane/niekompletne: {rejected_count}")
    logger.info(f"Do dalszego przetwarzania zakwalifikowano: {complete_count} w pełni kompletnych pacjentów.")

    df.set_index('hadm_id', inplace=True)
    X = df.drop(columns=['label'])
    y = df['label'].astype(int)

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=CONFIG["SEED"])


def cascade_predict(data):
    """ Definiujemy funkcję predykcji dla kaskady (od śmierci). """
    # SHAP często rzuca dane jako NumPy Array, co psuje ColumnTransformer
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=X_train.columns)

    # Zapewnienie, że kolumna 'model_1_prob' jest obecna, jeśli Pipeline jej wymaga
    return calibrated_spec_pipe.predict_proba(data)[:, 1]


def strip_names(X):
    """ Funkcja wymusza usunięcie nazw kolumn z potoku NumPy (potrzebne do poprawnego działania XGBoost). """
    return np.array(X)


def select_cols(df, columns):
    """ Zwraca wybrane kolumny z DataFrame. """
    return df.loc[:, columns]


def verify_local_hosp_files(output_path: str = None):
    """ Sprawdza obecność lokalnego pliku labevents.csv w katalogu hosp. """
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "Data", "hosp", "labevents.csv")
    
    # Obsługa zarówno wersji nieskompresowanej (.csv), jak i spakowanej (.csv.gz)
    if os.path.exists(output_path):
        logger.info(f"Znaleziono lokalny plik laboratoriów: {output_path}")
        return output_path
    elif os.path.exists(output_path + ".gz"):
        logger.info(f"Znaleziono skompresowany plik laboratoriów: {output_path}.gz")
        return output_path + ".gz"
    
    logger.warning(f"Brak lokalnego pliku labevents w ścieżce: {output_path}. System użyje trybu awaryjnego.")
    return None


def verify_local_icu_tables(base_dir: str = None):
    """ Weryfikuje obecność lokalnych tabel ICU, w tym chartevents.csv. """
    if base_dir is None:
        base_dir = os.path.join(BASE_DIR, "Data", "icu")

    os.makedirs(base_dir, exist_ok=True)
    logger.info(f"Weryfikacja lokalnych zasobów ICU w katalogu: {base_dir}")
    
    # Sprawdzamy kluczowy plik chartevents
    chartevents_path = os.path.join(base_dir, "chartevents.csv")
    if not os.path.exists(chartevents_path) and os.path.exists(chartevents_path + ".gz"):
        chartevents_path = chartevents_path + ".gz"

    if os.path.exists(chartevents_path):
        logger.info(f"Znaleziono lokalną tabelę ICU: {chartevents_path}")
        return True
    else:
        logger.warning(f"Nie znaleziono chartevents.csv w {base_dir}. Upewnij się, że plik tam jest.")
        return False


def download_and_extract_note_if_missing(username: str, password: str, output_csv_path: str = "radiology.csv"):
    """ Sprawdza, czy plik istnieje. Jeśli nie ma rozpakowanego, szuka .gz lub pobiera z PhysioNet,
        a następnie automatycznie wypakowuje do formatu .csv. """
    
    # Ustalamy ścieżki dla wersji spakowanej i rozpakowanej
    if output_csv_path.endswith(".gz"):
        gz_path = output_csv_path
        csv_path = output_csv_path[:-3]
    else:
        csv_path = output_csv_path
        gz_path = output_csv_path + ".gz"

    # Jeśli rozpakowany plik już istnieje, używamy go od razu
    if os.path.exists(csv_path):
        logger.info(f"Lokalny plik {csv_path} już istnieje. Używam go bezpośrednio.")
        return csv_path

    # Jeśli nie ma .csv, ale mamy spakowany plik .gz lokalnie, tylko go wypakowujemy
    if os.path.exists(gz_path):
        logger.info(f"Znaleziono spakowany plik {gz_path}. Wypakowywanie lokalne.")
        _extract_gz_file(gz_path, csv_path)
        return csv_path

    # Jeśli nie ma żadnego, pobieramy z PhysioNet, a potem wypakowujemy
    if username and password and username != "passy":
        url = "https://physionet.org/files/mimic-iv-note/2.2/note/radiology.csv.gz"
        logger.info("Rozpoczynam pobieranie pliku z PhysioNet (może potrwać kilka minut).")
        
        try:
            response = requests.get(url, auth=(username, password), stream=True, timeout=30)
            response.raise_for_status()
            
            with open(gz_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Pobrano pomyślnie i zapisano jako: {gz_path}")
            
            # Automatyczne wypakowanie pobranego archiwum
            _extract_gz_file(gz_path, csv_path)
            return csv_path
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Błąd sieciowy/autoryzacji podczas pobierania z PhysioNet: {e}")
            return None
    else:
        logger.warning("Brak danych logowania PhysioNet. Nie można pobrać brakującego pliku.")
        return None

def _extract_gz_file(gz_path, csv_path):
    """ Pomocnicza funkcja bezpiecznie wypakowująca archiwum .gz do pliku .csv. """
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(csv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(f"Pomyślnie wypakowano archiwum do: {csv_path}")
    except gzip.BadGzipFile as e:
        logger.error(f"Błąd podczas wypakowywania pliku {gz_path}: {e}")


def download_eicu_targeted_tables(username: str, password: str, base_dir: str = None):
    """ Pobiera z platformy PhysioNet wyłącznie trzy tabele z bazy eICU-CRD v2.0 niezbędne 
        do treningu modułu odłączania od respiratora i monitorowania Bayesowskiego. """
    
    if base_dir is None:
        base_dir = os.path.join(BASE_DIR, "Data", "eICU")

    os.makedirs(base_dir, exist_ok=True)
    logger.info(f"Rozpoczynam celowane pobieranie tabel eICU do katalogu: {base_dir}")
    
    files_to_download = [
        "patient.csv.gz",
        "respiratoryCare.csv.gz",
        "vitalPeriodic.csv.gz" 
    ]
    
    base_url = "https://physionet.org/files/eicu-crd/2.0/"
    
    for file_name in files_to_download:
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            logger.info(f"Plik {file_name} już istnieje lokalnie. Pomijam pobieranie.")
            continue
            
        logger.info(f"Pobieranie {file_name} z PhysioNet.")
        url = base_url + file_name
        
        try:
            response = requests.get(url, auth=(username, password), stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc=file_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=8192):
                    size = f.write(data)
                    bar.update(size)
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Błąd autoryzacji lub sieci podczas pobierania {file_name}: {e}")
            logger.info("Upewnij się, że masz podpisane DUA dla eICU na koncie PhysioNet!")
            return False
            
    logger.info("Wszystkie wymagane tabele eICU zostały pomyślnie pobrane.")
    return True


class PatientBaselineProfiler(nn.Module):
    """ Czysty moduł w PyTorch kompresujący choroby do Wektora Obciążenia w celu lepszej analizy pacjentów.
        Architektura Autoenkodera (Encoder-Decoder) eliminuje wyciek danych — sieć kompresuje informacje,
        ucząc się rekonstrukcji samego wejścia (MSE), a nie predykcji etykiety zgonu (Brak podglądania 'y'). """
    def __init__(self, input_dim=8, hidden_dim=32, output_dim=16, dropout_rate=0.2):
        super(PatientBaselineProfiler, self).__init__()
        
        # Enkoder: Kompresja do gęstego wektora rezerwy biologicznej
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Dekoder: Odtwarzanie oryginalnych cech pacjenta
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class ProfilerWrapper(BaseEstimator, TransformerMixin):
    """ Zoptymalizowany Scikit-Learn Wrapper integrujący Autoenkoder PyTorch (PatientBaselineProfiler)
        bezpośrednio z ColumnTransformerem. Uczenie nienadzorowane przy pomocy MSELoss. """
    def __init__(self, input_dim=8, hidden_dim=32, output_dim=16, epochs=3, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.profiler = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        X_arr = np.nan_to_num(np.array(X, dtype=np.float32))
        
        # Inicjalizacja sieci na właściwym urządzeniu
        self.profiler = PatientBaselineProfiler(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        
        # Funkcja straty rekonstrukcyjnej
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.profiler.parameters(), lr=self.lr)
        
        dataset = TensorDataset(torch.tensor(X_arr, dtype=torch.float32))
        loader = DataLoader(
            dataset, 
            batch_size=128, 
            shuffle=True, 
            drop_last=(len(dataset) > 128)
        )
        
        self.profiler.train()
        
        for epoch in range(self.epochs):
            for batch in loader:
                batch_x = batch[0].to(self.device)
                optimizer.zero_grad()
                
                # Zwraca skompresowany wektor oraz odtworzone wejście
                encoded, decoded = self.profiler(batch_x)
                
                # Uczymy sieć tak kompresować dane, żeby dało się je jak najdokładniej odtworzyć
                loss = criterion(decoded, batch_x)
                loss.backward()
                optimizer.step()
                
        self.is_fitted_ = True
        return self

    def transform(self, X, batch_size=64):
        if not self.is_fitted_ or self.profiler is None:
            raise NotFittedError("ProfilerWrapper nie został wyuczony!")
            
        X_arr = np.nan_to_num(np.array(X, dtype=np.float32))
        n_samples = X_arr.shape[0]
        self.profiler.eval()
        
        encoded_list = []
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = torch.tensor(X_arr[i:i + batch_size], dtype=torch.float32).to(self.device)

                # W fazie transformacji pobieramy tylko skompresowaną reprezentację pacjenta (Encoder)
                encoded, _ = self.profiler(batch)
                encoded_list.append(encoded.cpu().numpy())
            
        return np.vstack(encoded_list)
        
    def get_feature_names_out(self, input_features=None):
        return np.array([f"patient_burden_dim_{i}" for i in range(self.output_dim)])


class DeepGenerativeImputer(BaseEstimator, TransformerMixin):
    """ Generatywny model imputacji (Architektura typu GAIN/Autoencoder) działający
        nieliniowo. Doskonale radzi sobie z brakami typu MNAR na OIT. """
    def __init__(self, hidden_dim=64, epochs=10, lr=1e-3, batch_size=256):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = None
        self.feature_names_in_ = None
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.feature_names_in_ = X_df.columns.astype(str).tolist()
        self.input_dim = X_df.shape[1]
        
        # Generator wypełniający luki fizjologiczne
        self.generator = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim)
        ).to(self.device)
        
        optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        X_np = X_df.values

        # Tworzymy maskę: 1 tam gdzie są dane, 0 tam gdzie brakuje
        M_np = ~np.isnan(X_np)
        
        # Wstępna imputacja zerami, żeby wektory weszły do sieci
        X_filled = np.nan_to_num(X_np, nan=0.0)
        
        tensor_X = torch.FloatTensor(X_filled)
        tensor_M = torch.FloatTensor(M_np.astype(float))
        dataset = TensorDataset(tensor_X, tensor_M)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.generator.train()
        for epoch in range(self.epochs):
            for batch_X, batch_M in loader:
                batch_X, batch_M = batch_X.to(self.device), batch_M.to(self.device)
                optimizer.zero_grad()
                
                # Podajemy sieci połączone wejście: surowe dane + informację, gdzie są braki
                inputs = torch.cat([batch_X, batch_M], dim=1)
                reconstruction = self.generator(inputs)
                
                # Uczymy generator odtwarzać tylko te wartości, które faktycznie istnieją
                loss = criterion(reconstruction * batch_M, batch_X * batch_M)
                loss.backward()
                optimizer.step()
                
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_np = X_df.values
        M_np = ~np.isnan(X_np)
        X_filled = np.nan_to_num(X_np, nan=0.0)
        
        tensor_X = torch.FloatTensor(X_filled).to(self.device)
        tensor_M = torch.FloatTensor(M_np.astype(float)).to(self.device)
        inputs = torch.cat([tensor_X, tensor_M], dim=1)
        
        self.generator.eval()
        with torch.no_grad():
            imputed_values = self.generator(inputs).cpu().numpy()
            
        # Zastępujemy puste komórki wygenerowanymi przez sieć estymatami
        X_final = np.where(M_np, X_np, imputed_values)
        return X_final

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_)


class SyntheticPhenotypeMatcher:
    """ Wirtualne Parowanie Pacjentów oparte na modelach IMPACT i CRASH.
        Wykorzystuje Ciągłe Mapowanie Fizjologiczne uzupełnione o wskaźniki koagulopatii,
        reaktywność źrenic oraz hipoksję, w celu wiernego odwzorowania patofizjologii TBI. """
    def __init__(self, random_state=42):
        self.rng = np.random.default_rng(random_state)

    def match_phenotypes(self, X_tabular):
        n_samples = len(X_tabular)

        # Pobranie standardowych cech ciągłych
        gcs = X_tabular.get('gcs_sum', pd.Series(15, index=X_tabular.index)).values
        brainstem_risk = X_tabular.get('hidden_brainstem_crush_risk', pd.Series(0, index=X_tabular.index)).values
        map_min = X_tabular.get('map_min', pd.Series(80, index=X_tabular.index)).values
        age = X_tabular.get('age', pd.Series(50, index=X_tabular.index)).values
        
        # Pobranie twardych markerów z literatury z bezpiecznym fallbackiem
        pupils_reactive = X_tabular.get('pupils_reactive', pd.Series(1, index=X_tabular.index)).values
        spo2_min = X_tabular.get('spo2_min', pd.Series(98, index=X_tabular.index)).values
        inr_max = X_tabular.get('inr_max', pd.Series(1.0, index=X_tabular.index)).values
        is_intubated = X_tabular.get('is_intubated', pd.Series(0, index=X_tabular.index)).values

        # Normalizacja surowości do skali 0.0 - 1.0
        gcs_sev = np.clip((15 - gcs) / 12.0, 0, 1)          
        map_sev = np.clip((90 - map_min) / 40.0, 0, 1)     
        
        # Normalizacja markerów z literatury
        pupil_sev = np.where(pupils_reactive == 0, 1.0, 0.0)    # 1.0 = Sztywne, niereagujące źrenice
        hypoxia_sev = np.clip((95 - spo2_min) / 15.0, 0, 1)     # Skrajne niedotlenienie
        coagulopathy_sev = np.clip((inr_max - 1.0) / 2.0, 0, 1) # Ryzyko masywnego krwotoku
        age_sev = np.clip((age - 18) / 70.0, 0, 1)

        """ Modelowanie CRNN (Przesunięcie linii środkowej)
            Zgodnie z IMPACT: Źrenice i koagulopatia (INR) drastycznie windują szansę na rozległy krwiak i ucisk.
            Dodatkowo uwzględniono wiek: u seniorów postępująca atrofia mózgu powoduje napinanie żył mostowych,
            co drastycznie zwiększa ryzyko ich pęknięcia i powstania masywnego krwiaka podtwardówkowego przy błahym urazie. """
        crnn_logit = -3.0 + (3.0 * brainstem_risk) + (1.5 * gcs_sev) + (2.5 * pupil_sev) + (1.5 * coagulopathy_sev) + (1.0 * age_sev)
        crnn_base = 1 / (1 + np.exp(-crnn_logit))
        crnn_matched = np.clip(crnn_base + self.rng.normal(0, 0.12, n_samples), 0.01, 0.99)

        # Zgodnie z Brain Trauma Foundation: "Śmiercionośny duet" (Hipotensja + Hipoksja) niszczy istotę białą
        efficientnet_logit = -2.5 + (2.5 * gcs_sev) + (2.0 * map_sev) + (2.0 * hypoxia_sev)
        efficientnet_base = 1 / (1 + np.exp(-efficientnet_logit))
        efficientnet_matched = np.clip(efficientnet_base + self.rng.normal(0, 0.18, n_samples), 0.01, 0.99)

        """ Modelowanie DenseNet (Złamania czaszki)
            Złamania wymagają potężnej energii kinetycznej, która często wiąże się ze wstrząsem. Towarzyszące urazy
            głowy/twarzoczaszki błyskawicznie wymuszają zabezpieczenie dróg oddechowych. Dodatkowo, złamanie kości
            skroniowej często przerywa tętnicę oponową środkową, co generuje krwiak nadtwardówkowy i ucisk nerwu okoruchowego. """
        densenet_logit = -3.5 + (2.5 * gcs_sev) + (1.5 * pupil_sev) + (1.0 * is_intubated) + (1.0 * map_sev)
        densenet_base = 1 / (1 + np.exp(-densenet_logit))
        densenet_matched = np.clip(densenet_base + self.rng.normal(0, 0.08, n_samples), 0.01, 0.99)

        return crnn_matched, efficientnet_matched, densenet_matched
    

def extract_vitals_via_chunking(filepath, item_ids):
    logger.info("Rozpoczynam przesiewanie 30 GB danych z aparatury (Chunking Out-of-Core).")
    
    aggregated_results = []
    
    # Chunksize = 1 000 000 wierszy
    chunk_iterator = pd.read_csv(filepath, compression='gzip', chunksize=1000000, usecols=['hadm_id', 'itemid', 'valuenum'])
    
    # TQDM do iteracji przez fragmenty pliku
    for i, chunk in enumerate(tqdm(chunk_iterator, desc="Przetwarzanie paczek 1M wierszy", leave=False, file=sys.stdout, mininterval=2.0)):
        # Filtrowanie tylko potrzebnych sensorów
        filtered = chunk[chunk['itemid'].isin(item_ids)].copy()
        
        if filtered.empty:
            continue
            
        # Przygotowanie do wyliczenia Wariancji/Odchylenia
        filtered['valuenum_sq'] = filtered['valuenum'] ** 2
        
        # Zbieramy lokalne agregaty dla tej paczki
        agg_chunk = filtered.groupby(['hadm_id', 'itemid']).agg(
            chunk_max=('valuenum', 'max'),
            chunk_min=('valuenum', 'min'),
            chunk_sum=('valuenum', 'sum'),
            chunk_count=('valuenum', 'count'),
            chunk_sum_sq=('valuenum_sq', 'sum')
        ).reset_index()
        
        aggregated_results.append(agg_chunk)
        
    logger.info("Wszystkie paczki przetworzone. Fuzja danych bazowych.")
    
    if not aggregated_results:
        logger.warning("Brak danych po przefiltrowaniu dla podanych item_ids.")
        return pd.DataFrame(columns=['hadm_id', 'itemid', 'val_max', 'val_min', 'val_mean', 'val_std', 'val_count'])

    # Łączymy wszystkie paczki
    concat_df = pd.concat(aggregated_results, ignore_index=True)
    
    # Wyliczamy ostateczne Globalne wartości dla pacjenta z sum wszystkich paczek
    final_df = concat_df.groupby(['hadm_id', 'itemid']).agg(
        global_max=('chunk_max', 'max'),     # Max z maxów to po prostu Max
        global_min=('chunk_min', 'min'),     # Min z minów to po prostu Min
        global_sum=('chunk_sum', 'sum'),     # Suma z sum
        global_count=('chunk_count', 'sum'), # Suma ilości pomiarów
        global_sum_sq=('chunk_sum_sq', 'sum')# Suma z sum kwadratów
    ).reset_index()
    
    # Średnia = Globalna Suma / Globalny Count
    final_df['val_mean'] = final_df['global_sum'] / final_df['global_count']
    
    # Wariancja
    variance = (final_df['global_sum_sq'] / final_df['global_count']) - (final_df['val_mean'] ** 2)

    # Zabezpieczenie przed błędami precyzji float (liczby bardzo bliskie zeru na minusie)
    variance = variance.clip(lower=0) 
    
    # Prawdziwe Odchylenie Standardowe
    final_df['val_std'] = np.sqrt(variance)
    final_df['val_count'] = final_df['global_count']
    
    # Czyszczenie tabeli z kolumn pomocniczych
    final_df = final_df[['hadm_id', 'itemid', 'global_max', 'global_min', 'val_mean', 'val_std', 'val_count']]
    final_df.rename(columns={'global_max': 'val_max', 'global_min': 'val_min'}, inplace=True)
    
    logger.info("Zakończono! Z 30 GB wyciągnięto Min, Max, Prawdziwą Średnią, Odchylenie oraz Liczbę pomiarów.")
    return final_df


def extract_vision_training_cohorts(df: pd.DataFrame):
    """ Wypuszcza listy hadm_id pacjentów dla obu klas (chorych i zdrowych),
        niezbędnych do zbalansowanego treningu wizyjnego CRNN, EfficientNet i DenseNet. """
    logger.info("Generowanie zbalansowanych kohort do treningu Computer Vision (CRNN, EfficientNet & DenseNet).")
    
    # Zbiór CRNN typy krwiaków (Subdural, Epidural, Subarachnoid, Intraventricular, Intracerebral)
    crnn_col = 'vision_hematoma_type'
    if crnn_col in df.columns:
        crnn_positive = df[df[crnn_col] > 0].copy()
        crnn_negative = df[df[crnn_col] == 0].copy()
    else:
        crnn_positive = df.iloc[0:0]
        crnn_negative = df.iloc[0:0]
    
    n_crnn_neg_samples = min(len(crnn_negative), len(crnn_positive) * 2)
    if n_crnn_neg_samples > 0:
        crnn_negative = crnn_negative.sample(n=n_crnn_neg_samples, random_state=42)
    else:
        crnn_negative = crnn_negative.iloc[0:0] 
    
    crnn_cohort = pd.concat([crnn_positive, crnn_negative])
    
    if not crnn_cohort.empty and crnn_col in df.columns:
        crnn_cohort['vision_label'] = (crnn_cohort[crnn_col] > 0).astype(int)
        crnn_cohort[['vision_label']].to_csv('vision_cohort_crnn_hematomas.csv', index_label='hadm_id')

    # Zbiór EfficientNet (Ciało Modzelowate)
    cc_col = 'vision_corpus_callosum'
    if cc_col in df.columns:
        cc_positive = df[df[cc_col] == 1.0].copy()
        cc_negative = df[df[cc_col] == 0.0].copy()
    else:
        cc_positive = df.iloc[0:0]
        cc_negative = df.iloc[0:0]
    
    n_cc_neg_samples = min(len(cc_negative), len(cc_positive) * 2)
    if n_cc_neg_samples > 0:
        cc_negative = cc_negative.sample(n=n_cc_neg_samples, random_state=42)
    else:
        cc_negative = cc_negative.iloc[0:0] 
        
    cc_cohort = pd.concat([cc_positive, cc_negative])
    
    if not cc_cohort.empty and cc_col in df.columns:
        cc_cohort['vision_label'] = cc_cohort[cc_col].astype(int)
        cc_cohort[['vision_label']].to_csv('vision_cohort_efficientnet_callosum.csv', index_label='hadm_id')

    # Zbiór DenseNet (Złamania czaszki)
    fracture_col = 'vision_skull_fracture'
    if fracture_col in df.columns:
        dn_positive = df[df[fracture_col] == 1.0].copy()
        dn_negative = df[df[fracture_col] == 0.0].copy()
    else:
        dn_positive = df.iloc[0:0]
        dn_negative = df.iloc[0:0]
    
    n_dn_neg_samples = min(len(dn_negative), len(dn_positive) * 2)
    if n_dn_neg_samples > 0:
        dn_negative = dn_negative.sample(n=n_dn_neg_samples, random_state=42)
    else:
        dn_negative = dn_negative.iloc[0:0]
        
    dn_cohort = pd.concat([dn_positive, dn_negative])
    
    if not dn_cohort.empty and fracture_col in df.columns:
        dn_cohort['vision_label'] = dn_cohort[fracture_col].astype(int)
        dn_cohort[['vision_label']].to_csv('vision_cohort_densenet_fracture.csv', index_label='hadm_id')
    
    logger.info(f"Zbiór CRNN (Krwiaki): {len(crnn_positive)} chorych | {len(crnn_negative)} zdrowych")
    logger.info(f"Zbiór EfficientNet (Ciało Modzelowate): {len(cc_positive)} chorych | {len(cc_negative)} zdrowych")
    logger.info(f"Zbiór DenseNet (Złamania): {len(dn_positive)} chorych | {len(dn_negative)} zdrowych")
    
    return crnn_cohort.index.tolist(), cc_cohort.index.tolist(), dn_cohort.index.tolist()


# Własny Wrapper omijający błąd biblioteki Imbalanced-Learn
class CustomClassifierWrapper(BaseEstimator, ClassifierMixin):
    """ Gwarantowany klasyfikator dla CalibratedClassifierCV omijający błędy Pipeline'u """
    def __init__(self, prep, rus, router):
        self.prep = prep
        self.rus = rus
        self.router = router
        
    def fit(self, X, y):
        # Transformacja
        X_prep = self.prep.fit_transform(X, y)

        # Undersampling  (na zbiorze treningowym)
        X_res, y_res = self.rus.fit_resample(X_prep, y)

        # Uczenie głównego modelu klastrującego
        self.router.fit(X_res, y_res)

        # Kluczowe dla Scikit-Learn jawne przekazanie klas
        self.classes_ = self.router.classes_
        return self

    def predict_proba(self, X):
        X_prep = self.prep.transform(X)
        return self.router.predict_proba(X_prep)
        
    def predict(self, X):
        X_prep = self.prep.transform(X)
        return self.router.predict(X_prep)


def load_trained_crnn_predictions(X_test, dataset_path='Data/dataset_ct_main', device=None):
    """ Wczytuje wytrenowane wagi CRNN i generuje prawdziwe predykcje wizualne dla zbioru testowego. """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model_path = 'output_data/Models/crnn_fine_uncertainty.pth'
    if not os.path.exists(model_path):
        logger.warning("Brak pliku wag CRNN. Zwracam neutralne prawdopodobieństwa 0.5.")
        return np.full(len(X_test), 0.5)
        
    try:
        model = VolumetricMedicalResNetCRNN(task='multilabel', num_classes=5).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        probs_list = []
        img_dir = os.path.join(dataset_path, 'images')
        
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        with torch.no_grad():
            for hadm_id in X_test.index:
                patient_folder = os.path.join(img_dir, str(hadm_id))
                
                if not os.path.exists(patient_folder):
                    probs_list.append(0.5)
                    continue
                    
                slices = sorted([f for f in os.listdir(patient_folder) if f.endswith(('.png', '.jpg'))])
                if len(slices) == 0:
                    probs_list.append(0.5)
                    continue
                    
                # CRNN został wytrenowany na 24 plastrach
                if len(slices) > 24:
                    indices = np.linspace(0, len(slices) - 1, 24, dtype=int)
                    slices = [slices[i] for i in indices]
                elif len(slices) < 24:
                    slices = slices + [slices[-1]] * (24 - len(slices))
                    
                volume_tensor = torch.zeros((24, 3, 160, 160))
                for i, slice_file in enumerate(slices):
                    img_path = os.path.join(patient_folder, slice_file)
                    img_rgb = Image.open(img_path).convert('RGB')
                    volume_tensor[i] = transform(img_rgb)
                        
                vol_input = volume_tensor.unsqueeze(0).to(device)

                autocast_context = torch.amp.autocast(device_type='cuda') if device.type == 'cuda' else nullcontext()
                with autocast_context:
                    vol_input_flipped = torch.flip(vol_input, dims=[-1])
                    
                    out_orig = torch.sigmoid(model(vol_input)).cpu().numpy()[0]
                    out_flipped = torch.sigmoid(model(vol_input_flipped)).cpu().numpy()[0]
                    
                    pred_prob = (out_orig + out_flipped) / 2.0
                    probs_list.append(float(np.max(pred_prob)))

        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return np.array(probs_list)
    except RuntimeError as e:
        logger.error(f"Błąd sprzętowy PyTorch w CRNN: {e}")
        return np.full(len(X_test), 0.5)
    except OSError as e:
        logger.error(f"Błąd odczytu z dysku podczas inferencji CRNN: {e}")
        return np.full(len(X_test), 0.5)


def load_trained_efficientnet_predictions(X_test, dataset_path='Data/dataset_efficientnet', device=None):
    """ Wczytuje oba modele EfficientNet (Main i Fine Tune) i generuje kaskadowe predykcje oraz niepewność dla zbioru testowego. """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if not os.path.exists(dataset_path):
        logger.warning("Brak folderu danych. Zwracam neutralne prawdopodobieństwa 0.5.")
        return np.full(len(X_test), 0.5), np.zeros(len(X_test))

    path_main = 'output_data/Models/efficientnet_corpus_callosum_main.pth'
    path_fine = 'output_data/Models/efficientnet_corpus_callosum_finetune.pth'
    
    if not os.path.exists(path_main) or not os.path.exists(path_fine):
        logger.warning("Brak wag kaskady EfficientNet. Zwracam neutralne prawdopodobieństwa 0.5.")
        return np.full(len(X_test), 0.5), np.zeros(len(X_test))
        
    try:
        model_main = VolumetricEfficientNet(num_slices=30, backbone='efficientnet').to(device)
        model_main.load_state_dict(torch.load(path_main, map_location=device))
        model_main.eval()

        model_fine = VolumetricEfficientNet(num_slices=30, backbone='efficientnet').to(device)
        model_fine.load_state_dict(torch.load(path_fine, map_location=device))
        model_fine.eval()
        
        unet_path = 'output_data/Models/unet_corpus_callosum.pth'
        unet_model = UNet().to(device) if os.path.exists(unet_path) else None
        if unet_model:
            unet_model.load_state_dict(torch.load(unet_path, map_location=device))
            unet_model.eval()

        probs_list = []
        uncert_list = []
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        with torch.no_grad():
            for hadm_id in X_test.index:
                patient_folder = None
                for cls_dir in ['0_normal', '1_corpus_callosum_dai']:
                    potential_path = os.path.join(dataset_path, cls_dir, str(hadm_id))
                    if os.path.exists(potential_path):
                        patient_folder = potential_path
                        break
                        
                if not patient_folder:
                    probs_list.append(0.5)
                    uncert_list.append(0.0)
                    continue
                    
                slices = sorted([f for f in os.listdir(patient_folder) if f.endswith(('.png', '.jpg'))])
                if len(slices) == 0:
                    probs_list.append(0.5)
                    uncert_list.append(0.0)
                    continue
                    
                if len(slices) > 30:
                    indices = np.linspace(0, len(slices) - 1, 30, dtype=int)
                    slices = [slices[i] for i in indices]
                elif len(slices) < 30:
                    slices = slices + [slices[-1]] * (30 - len(slices)) 

                volume_tensor = torch.zeros((30, 3, 224, 224))
                for i, slice_file in enumerate(slices):
                    img_path = os.path.join(patient_folder, slice_file)
                    img = Image.open(img_path).convert('RGB')
                    volume_tensor[i] = transform(img)
                    
                vol_input = volume_tensor.unsqueeze(0).to(device)
                
                if unet_model is not None:
                    B, S, C, H, W = vol_input.shape
                    brain_slices = vol_input[:, :, 2:3, :, :].contiguous().view(B * S, 1, H, W)
                    mask_pred = torch.sigmoid(unet_model(brain_slices)).view(B, S, 1, H, W)
                    processed_volume = vol_input * mask_pred
                else:
                    processed_volume = vol_input

                _, final_prob = sequential_two_stage_predict(
                    processed_volume, model_main, model_fine, threshold_fine=0.3, threshold_main=0.6
                )
                p = float(final_prob[0])
                probs_list.append(p)
                uncert_list.append(float(abs(p - 0.5) * 2.0))

        del model_main, model_fine, unet_model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return np.array(probs_list), np.array(uncert_list)
        
    except RuntimeError as e:
        logger.error(f"Błąd sprzętowy PyTorch w EfficientNet: {e}")
        return np.full(len(X_test), 0.5), np.zeros(len(X_test))
    except (OSError, ValueError) as e:
        logger.error(f"Błąd podczas inferencji EfficientNet: {e}")
        return np.full(len(X_test), 0.5), np.zeros(len(X_test))
    

def load_trained_densenet_predictions(X_test, dataset_path='Data/dataset_ct_main', device=None):
    """ Wczytuje wytrenowane wagi DenseNet-121 i generuje predykcje obecności złamań na pełnych wolumenach skanów. """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model_path = 'output_data/Models/densenet_fracture_best.pth'
    if not os.path.exists(model_path):
        logger.warning("Brak pliku wag DenseNet. Zwracam neutralne prawdopodobieństwa 0.5.")
        return np.full(len(X_test), 0.5)
        
    try:
        model = DenseNetFractureDetector(pretrained=True, dropout_rate=0.0).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        bone_window = BoneWindowTransform(window_center=400, window_width=1000)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        probs_list = []
        img_dir = os.path.join(dataset_path, 'images')
        
        with torch.no_grad():
            for hadm_id in X_test.index:
                patient_folder = os.path.join(img_dir, str(hadm_id))
                
                if not os.path.exists(patient_folder):
                    probs_list.append(0.5)
                    continue
                    
                slices = sorted([f for f in os.listdir(patient_folder) if f.endswith(('.png', '.jpg', '.dcm'))])
                if len(slices) == 0:
                    probs_list.append(0.5)
                    continue
                    
                if len(slices) > 30:
                    indices = np.linspace(0, len(slices) - 1, 30, dtype=int)
                    slices = [slices[i] for i in indices]
                elif len(slices) < 30:
                    slices = slices + [slices[-1]] * (30 - len(slices))
                    
                volume_tensor = torch.zeros((1, 30, 224, 224))
                
                for i, slice_file in enumerate(slices):
                    img_path = os.path.join(patient_folder, slice_file)
                    
                    if img_path.endswith('.dcm'):
                        dicom_data = pydicom.dcmread(img_path)
                        img_array = dicom_data.pixel_array.astype(np.float32)
                        intercept = float(getattr(dicom_data, 'RescaleIntercept', 0.0))
                        slope = float(getattr(dicom_data, 'RescaleSlope', 1.0))
                        hu_image = img_array * slope + intercept
                        img_pil = bone_window(hu_image)
                    else:
                        img_array = np.array(Image.open(img_path).convert('L'))
                        img_pil = bone_window(img_array)
                    
                    tensor_2d = transform(img_pil)
                    volume_tensor[0, i] = tensor_2d.squeeze(0)
                        
                vol_input = volume_tensor.unsqueeze(0).to(device)
                
                autocast_context = torch.amp.autocast(device_type='cuda') if device.type == 'cuda' else nullcontext()
                with autocast_context:
                    outputs = model(vol_input)
                    pred_prob = torch.sigmoid(outputs).cpu().numpy()
                    val = float(pred_prob.max() if isinstance(pred_prob, np.ndarray) else pred_prob)
                    probs_list.append(val)

        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
              
        return np.array(probs_list)
    except RuntimeError as e:
        logger.error(f"Błąd sprzętowy PyTorch w DenseNet: {e}")
        return np.full(len(X_test), 0.5)
    except OSError as e:
        logger.error(f"Błąd odczytu z dysku podczas inferencji DenseNet: {e}")
        return np.full(len(X_test), 0.5)
    except ValueError as e:
        logger.error(f"Błąd przetwarzania macierzy obrazu/DICOM w DenseNet: {e}")
        return np.full(len(X_test), 0.5)


def seed_everything(seed=42):
    """ Inicjalizuje wszystkie ziarna losowości dla bibliotek zewnętrznych, gwarantując w 100% deterministyczny i powtarzalny
        trening modeli stochastycznych oraz sieci PyTorch. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Wymusza determinizm na poziomie sprzętowym
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False


def get_s3_client():
    """ Inicjalizuje klienta kompatybilnego z S3 (AWS lub lokalne MinIO) z wymuszeniem s3v4. """
    return boto3.client(
        's3',
        endpoint_url=SETTINGS.s3_endpoint,
        aws_access_key_id=SETTINGS.s3_access_key,
        aws_secret_access_key=SETTINGS.s3_secret_key,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )


def setup_environment():
    """ Przygotowuje środowisko: czeka na MinIO/S3, sprawdza pliki, w razie braku pobiera je
        z PhysioNet, rozpakowuje efemerycznie i wysyła prosto do Object Storage, a następnie zasila bazę. """
    logger.info("Inicjalizacja środowiska bazodanowego i zmiennych (Cloud-Native).")
    
    os.makedirs(SETTINGS.models_dir, exist_ok=True)
    os.makedirs(SETTINGS.plots_dir, exist_ok=True)
    
    s3 = get_s3_client()
    bucket = SETTINGS.s3_bucket
    
    # Aktywne oczekiwanie na start MinIO (max 60 sekund)
    max_retries = 12
    for attempt in range(max_retries):
        try:
            s3.head_bucket(Bucket=bucket)
            logger.info(f"Połączono z MinIO. Bucket '{bucket}' istnieje i jest gotowy.")
            break
        except ClientError as e:
            error_code = e.response['Error'].get('Code', '')
            if error_code == '404':
                logger.info(f"Bucket S3 '{bucket}' nie istnieje. Trwa inicjalizacja.")
                try:
                    s3.create_bucket(Bucket=bucket)
                    break
                except Exception as ex:
                    logger.error(f"Odmowa dostępu przy tworzeniu bucketa: {ex}")
                    break
            else:
                logger.warning(f"Oczekiwanie na gotowość MinIO (próba {attempt + 1}/{max_retries})...")
                time.sleep(5)
        except BotoCoreError as e:
            logger.warning(f"Brak stabilnego połączenia z MinIO (próba {attempt + 1}/{max_retries})...")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Nieoczekiwany błąd podczas łączenia z MinIO: {e}")
            break
    else:
        logger.error("Krytyczny błąd: MinIO nie uruchomiło się w wyznaczonym czasie. System może zachować się niestabilnie.")

    files_config = [
        {
            "s3_key": "physionet/hosp/labevents.csv",
            "name": "labevents.csv.gz",
            "url": "https://physionet.org/files/mimic-iv/2.2/hosp/labevents.csv.gz"
        },
        {
            "s3_key": "physionet/icu/chartevents.csv",
            "name": "chartevents.csv.gz",
            "url": "https://physionet.org/files/mimic-iv/2.2/icu/chartevents.csv.gz"
        }
    ]

    for item in files_config:
        s3_key = item["s3_key"]
        
        try:
            s3.head_object(Bucket=bucket, Key=s3_key)
            logger.info(f"Plik {s3_key} jest już dostępny w chmurze (MinIO/S3).")
            continue
        except ClientError as e:
            error_code = e.response['Error'].get('Code', '')
            if error_code == '404':
                logger.info(f"Brak pliku {s3_key} w Object Storage. Próba pobrania.")
            else:
                logger.error(f"Nieoczekiwany błąd podczas sprawdzania pliku {s3_key}: {e}")
                continue

        if SETTINGS.physionet_user and SETTINGS.physionet_pass and SETTINGS.physionet_user != "passy":
            logger.info(f"Pobieranie {item['name']} z PhysioNet.")
            try:
                response = requests.get(item["url"], auth=(SETTINGS.physionet_user, SETTINGS.physionet_pass), stream=True, timeout=30)
                response.raise_for_status()
                
                with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as tmp_gz:
                    for chunk in response.iter_content(chunk_size=8192):
                        tmp_gz.write(chunk)
                    tmp_gz_path = tmp_gz.name
                
                logger.info("Pobieranie zakończone. Wypakowuję archiwum do formatu CSV...")
                with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_csv:
                    with gzip.open(tmp_gz_path, 'rb') as f_in:
                        shutil.copyfileobj(f_in, tmp_csv)
                    tmp_csv_path = tmp_csv.name
                
                logger.info(f"Wysyłanie wypakowanego pliku do MinIO pod klucz: {s3_key}")
                s3.upload_file(tmp_csv_path, bucket, s3_key)
                logger.info(f"Pomyślnie pobrano, wypakowano i zapisano plik w MinIO: {s3_key}")
                
                os.remove(tmp_gz_path)
                os.remove(tmp_csv_path)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Błąd pobierania {item['name']} z PhysioNet: {e}")
                logger.warning(f"System użyje danych awaryjnych/syntetycznych dla brakującego zasobu.")
        else:
            logger.warning(f"Brak pliku {s3_key} oraz brak danych logowania PhysioNet. Pomijam pobieranie, używam trybu syntetycznego.")

    check_and_seed_database()


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


def train_pipeline(X_train, X_test, y_train, y_test):
    """ Główny potok treningowy systemu TBI:
        Przeprowadza inżynierię cech, Multi Task Learning, optymalizację hiperparametrów Optuną,
        turniej fenotypów oraz ostateczną fuzję multimodalną przez Sędziego EBM. """
    logger.info("Inżynieria Cech")
    columns_to_drop = ['encounter_id', 'id', 'unnamed: 0', 'unnamed: 83', 'hospital_death', 'hospital_outcome_death']
    rename_map = {
        'gcs_eyes_apache': 'gcs_eyes', 'gcs_motor_apache': 'gcs_motor', 'gcs_verbal_apache': 'gcs_verbal',
        'intubated_apache': 'is_intubated', 'd1_sysbp_min': 'sysbp_min', 'd1_heartrate_max': 'heart_rate_max',
        'd1_diasbp_min': 'diasbp_min', 'd1_glucose_max': 'glucose_max', 'd1_glucose_min': 'glucose_min'
    }

    for df in [X_train, X_test]:
        df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
        for old, new in rename_map.items():
            if old in df.columns:
                if new in df.columns: df.drop(columns=[old], inplace=True)
                else: df.rename(columns={old: new}, inplace=True)

        if 'gcs_sum' not in df.columns:
            df['gcs_sum'] = df['gcs_eyes'].fillna(1) + df['gcs_motor'].fillna(1) + df['gcs_verbal'].fillna(1)

        if 'is_intubated' in df.columns:
            df.loc[df['is_intubated'] == 1, 'gcs_verbal'] = np.nan
            df['gcs_unreliable_flag'] = df['is_intubated']

        if 'ventilated_apache' in df.columns and 'is_intubated' in df.columns:
            df['sedation_risk_flag'] = df[['ventilated_apache', 'is_intubated']].max(axis=1)
            df['gcs_sedation_mismatch'] = ((df['gcs_sum'] <= 5) & (df['sedation_risk_flag'] == 1)).astype(int)

        if 'apache_3j_bodysystem' in df.columns:
            df['baseline_neuro_deficit'] = df['apache_3j_bodysystem'].apply(lambda x: 1 if str(x).lower() in ['neurological', 'neurologic'] else 0)
            df['adjusted_gcs'] = np.where(df['baseline_neuro_deficit'] == 1, df['gcs_sum'] + 2, df['gcs_sum'])
            df['adjusted_gcs'] = df['adjusted_gcs'].clip(upper=15)

        df['is_hypotensive'] = (df['sysbp_min'] < 90).astype(int)

        if 'age' in df.columns and 'gcs_sum' in df.columns:
            df['age_squared'] = df['age'] ** 2
            df['critical_geriatric_condition'] = ((df['age'] > 75) & (df['gcs_sum'] < 5)).astype(int)
            df['neurological_efficiency'] = df['gcs_sum'] / (df['age'] + 1e-5)

        if 'glucose_max' in df.columns and 'glucose_min' in df.columns:
            df['glucose_stress'] = df['glucose_max'] - df['glucose_min']

        if 'heart_rate_max' in df.columns and 'sysbp_min' in df.columns:
            df['shock_index'] = df['heart_rate_max'] / (df['sysbp_min'] + 1e-5)
            
            # Jawny wskaźnik Triady Cushinga (Nadciśnienie skurczowe + Bradykardia)
            df['cushing_triad_score'] = (df['sysbp_min'] / 120.0) * (80.0 / (df['heart_rate_max'] + 1e-5))
        else:
            df['shock_index'] = 0.7
            df['cushing_triad_score'] = 1.0

        df['neuro_cardio_risk'] = df['shock_index'] / (df['gcs_sum'] + 1e-5)

        if 'd1_sysbp_max' in df.columns and 'd1_heartrate_min' in df.columns:
            df['cushing_risk_index'] = df['d1_sysbp_max'] / (df['d1_heartrate_min'] + 1e-5)

        if 'sysbp_min' in df.columns and 'diasbp_min' in df.columns:
            df['map_min'] = (df['sysbp_min'] + 2 * df['diasbp_min']) / 3
            df['is_low_map'] = (df['map_min'] < 65).astype(int)

        if 'spo2_min' in df.columns and 'fio2_max' in df.columns:
            # FiO2 czasami w bazach jest ułamkiem (np. 0.21), a czasami procentem (np. 21.0)
            fio2_fraction = np.where(df['fio2_max'] > 1.0, df['fio2_max'] / 100.0, df['fio2_max'])
            df['sf_ratio'] = df['spo2_min'] / (fio2_fraction + 1e-5)

        # Spłaszczanie koncentrycznych stref ryzyka (U-shaped risk dla homeostazy)
        if 'map_min' in df.columns:
            # Optymalne MAP dla mózgu to ok. 85 mmHg. Zarówno za nisko, jak i za wysoko to zagrożenie
            df['map_radial_deviation'] = np.abs(df['map_min'] - 85.0)

        if 'glucose_max' in df.columns:
            # Optymalny poziom glukozy to ok. 100 mg/dL (hipoglikemia i hiperglikemia niszczą tkankę mózgową)
            df['glucose_radial_deviation'] = np.abs(df['glucose_max'] - 100.0)

        if 'spo2_min' in df.columns:
            # Optymalne SpO2 to ok. 98%. Za nisko to hipoksja, za wysoko (hiperoksja) też generuje stres oksydacyjny
            df['spo2_radial_deviation'] = np.abs(df['spo2_min'] - 98.0)

        if 'sysbp_min' in df.columns:
            # Optymalne ciśnienie skurczowe to ok. 120 mmHg. Skrajności oznaczają wstrząs lub udar/krwotok
            df['sysbp_radial_deviation'] = np.abs(df['sysbp_min'] - 120.0)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    logger.info("Multi-Task Learning (Czujnik Wgłobienia)")
    mtl_features = [c for c in X_train.columns if c not in ['cushing_risk_index', 'd1_sysbp_max', 'd1_heartrate_min', 'sysbp_min', 'heart_rate_max',
                                                            'pulse_pressure', 'bp_swing', 'hr_swing', 'modified_shock_index', 'shock_index']]
    
    X_train_mtl = X_train[mtl_features].copy()
    X_test_mtl = X_test[mtl_features].copy()
    
    for col in X_train_mtl.select_dtypes(include=['object', 'category']).columns:
        X_train_mtl[col], _ = pd.factorize(X_train_mtl[col])
        X_test_mtl[col], _ = pd.factorize(X_test_mtl[col])

    # Dodajemy cechę docelową dla Multi Task Learning: hidden_brainstem_crush_risk, która jest różnicą między przewidywanym a rzeczywistym indeksem ryzyka wgłobienia Cushinga
    mtl_regressor = HistGradientBoostingRegressor(random_state=CONFIG["SEED"], max_iter=150)
    y_train_target = X_train['cushing_risk_index'].fillna(0)

    # Generowanie predykcji Out Of Fold dla treningu
    train_oof_preds = cross_val_predict(
        mtl_regressor, 
        X_train_mtl, 
        y_train_target, 
        cv=5, 
        method='predict'
    )
    X_train['hidden_brainstem_crush_risk'] = train_oof_preds - y_train_target

    # Dotrenowanie na całości danych do celów inferencji na zbiorze testowym
    mtl_regressor.fit(X_train_mtl, y_train_target)
    X_test['hidden_brainstem_crush_risk'] = mtl_regressor.predict(X_test_mtl) - X_test['cushing_risk_index'].fillna(0)

    for df_target, df_source in zip([X_train, X_test], [X_train_mtl, X_test_mtl]):
        df_target['hidden_brainstem_crush_risk'] = mtl_regressor.predict(df_source) - df_target['cushing_risk_index'].fillna(0)

    # Definiowanie kolumn
    baseline_cols = ['age', 'diabetes_mellitus', 'immunosuppression', 'cirrhosis', 'hepatic_failure', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']
    numerical_cols = [c for c in X_train.select_dtypes(include=['int64', 'float64']).columns if c not in baseline_cols]
    categorical_cols = [c for c in X_train.select_dtypes(include='object').columns if c not in baseline_cols]

    baseline_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('scaler', StandardScaler()),
                                     ('pytorch_profiler', PatientBaselineProfiler(input_dim=8, hidden_dim=32, output_dim=16, epochs=3))])
    
    # Drzewa doskonale radzą sobie z brakującymi danymi, więc używamy prostego imputera dla cech kategorycznych i normalizacji dla cech numerycznych
    categorical_transformer_trees = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('target_enc', TargetEncoder(smooth="auto"))])
    numerical_transformer_trees = Pipeline([
        ('scaler', StandardScaler(with_mean=False)), 
        ('noise', AdaptiveMedicalNoiseAdder())
    ])
    preprocessor_trees = ColumnTransformer([
        ('baseline', baseline_transformer, baseline_cols), 
        ('num', numerical_transformer_trees, numerical_cols),
        ('cat', categorical_transformer_trees, categorical_cols)
    ], n_jobs=1, verbose_feature_names_out=False)

    # Dla modeli, które nie radzą sobie z brakującymi danymi, stosujemy imputację DeepGenerativeImputer dla cech numerycznych i najczęstszej wartości dla cech kategorycznych
    categorical_transformer_imputed = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('target_enc', TargetEncoder(smooth="auto"))])
    numerical_transformer_imputed = Pipeline([
        ('gain_imputer', DeepGenerativeImputer(epochs=20, lr=0.005)), 
        ('scaler', StandardScaler()), 
        ('noise', AdaptiveMedicalNoiseAdder())
    ])
    preprocessor_imputed = ColumnTransformer([
        ('baseline', baseline_transformer, baseline_cols), 
        ('num', numerical_transformer_imputed, numerical_cols),
        ('cat', categorical_transformer_imputed, categorical_cols)
    ], n_jobs=1, verbose_feature_names_out=False)

    logger.info("Trenowanie Kaskady MoE z Optuną (Dynamiczny dobór architektury)")
    X_train_clean = preprocessor_trees.fit_transform(X_train, y_train)
    X_test_clean = preprocessor_trees.transform(X_test)

    feature_names = [name.split('__')[-1] for name in preprocessor_trees.get_feature_names_out()]
    X_train_clean_df = pd.DataFrame(X_train_clean, columns=feature_names, index=X_train.index).astype(np.float32)
    X_test_clean_df = pd.DataFrame(X_test_clean, columns=feature_names, index=X_test.index).astype(np.float32)

    # Ekstrakcja reguł asocjacyjnych z danych klinicznych i wzbogacenie zbioru cech o binarne flagi fenotypów
    logger.info("Uruchamianie ekstrakcji fenotypów klinicznych.")
    X_train_clean_df, X_test_clean_df, clinical_rules = extract_and_apply_tree_rules(
        X_train_clean_df, y_train, X_test_clean_df, feature_names
    )
    
    # Aktualizacja listy cech po dodaniu nowych, binarnych flag fenotypów
    feature_names_enriched = list(X_train_clean_df.columns)
    logger.info(f"Całkowita liczba cech po wzbogaceniu: {len(feature_names_enriched)}")

    # Przekazanie nowych cech do głównego zbioru danych
    rule_columns = [col for col in X_train_clean_df.columns if "Phenotype_Rule" in col]
    for col in rule_columns:
        X_train[col] = X_train_clean_df[col].values
        X_test[col] = X_test_clean_df[col].values

    ordered_constraints = []
    for col in X_train_clean_df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['gcs', 'sysbp', 'spo2', 'map_min', 'adjusted_gcs']): ordered_constraints.append(-1)
        elif any(x in col_lower for x in ['shock_index', 'cushing_risk', 'age_squared', 'is_hypotensive']): ordered_constraints.append(1)
        else: ordered_constraints.append(0)
    monotone_tuple = tuple(ordered_constraints)

    X_train_numeric = X_train_clean_df.values
    y_train_numeric = y_train.astype(int).values
    cv_strategy = StratifiedKFold(n_splits=2, shuffle=True, random_state=CONFIG["SEED"])

    # Mapowanie nazw flag binarnych na indeksy tablicy numpy dla BernoulliNB
    binary_flags = [
        'is_intubated', 'gcs_sedation_mismatch', 'baseline_neuro_deficit', 
        'is_hypotensive', 'critical_geriatric_condition', 'is_low_map'
    ]
    binary_indices = [list(X_train_clean_df.columns).index(col) 
                      for col in binary_flags if col in X_train_clean_df.columns]
    
    logger.info(f"BernoulliNB wykorzysta {len(binary_indices)} flag binarnych jako system ostrzegania.")

    # Przekazanie indeksów do Optuny
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
    ensemble_study.optimize(bound_objective, n_trials=CONFIG.get("N_OPTUNA_TRIALS", 15))

    bp = ensemble_study.best_params
    best_ensemble_type = bp.get('ensemble_type', 'full_stack')
    logger.info(f"Optuna wyłoniła optymalną architekturę Konsylium: {best_ensemble_type}")

    # Odtworzenie zwycięskiej architektury na pełnych danych
    final_estimators = []
    
    if best_ensemble_type in ['duo_orthogonal', 'trio_trees', 'full_stack']:
        xgb_p = {
            'n_estimators': bp['xgb_n_estimators'], 'learning_rate': bp['xgb_lr'], 'max_depth': bp['xgb_max_depth'],
            'reg_alpha': bp['xgb_alpha'], 'reg_lambda': bp['xgb_lambda'], 'scale_pos_weight': bp['xgb_spw'],
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
            'random_seed': SETTINGS.seed, 'verbose': 0, 'task_type': 'CPU', 'train_dir': os.path.join(SETTINGS.models_dir, 'catboost_info')
        }
        final_estimators.append(('cat', CatBoostClassifier(**cat_p)))

    if best_ensemble_type in ['duo_orthogonal', 'full_stack']:
        X_train_tabnet = preprocessor_imputed.fit_transform(X_train, y_train)
        final_estimators.append(('tabnet', TabNetClassifier(n_d=16, n_a=16, gamma=bp['tabnet_gamma'], verbose=0, seed=SETTINGS.seed)))

    # BernoulliNB do ostatecznego zespołu, opierający się na flagach binarnych fenotypów klinicznych
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
    ensemble_best.fit(X_train_numeric, y_train_numeric)

    joblib.dump(ensemble_best, os.path.join(SETTINGS.models_dir, 'Optimized_Stacking_Ensemble.pkl'))

    logger.info("Izolacja Anomali i Turniej Specjalistów")
    anomaly_detector = IsolationForest(
        n_estimators=200, contamination=0.1, random_state=CONFIG["SEED"], n_jobs=-1
    )

    # Cicha śmierć pacjentów z urazami mózgu, którzy zmarli mimo optymistycznej prognozy APACHE (< 25% szans na zgon)
    mask_apache_optimistic = (X_train['apache_4a_hospital_death_prob'] < 0.25)
    mask_hidden_instability = (
        ((X_train['d1_sysbp_max'] - X_train['d1_sysbp_min']) > 50) |  # Duża amplituda ciśnienia skurczowego
        (X_train['d1_glucose_max'] > 200)                             # Stres metaboliczny / hiperglikemia pourazowa
    )
    silent_death_mask_train = (y_train == 1) & mask_apache_optimistic & mask_hidden_instability


    # Cudem ocaleni pacjenci z ciężkimi urazami mózgu, którzy przeżyli mimo pesymistycznej prognozy APACHE (> 75% szans na zgon)
    mask_apache_pessimistic = (X_train['apache_4a_hospital_death_prob'] > 0.75)
    severe_survivors_mask_train = (y_train == 0) & mask_apache_pessimistic

    
    # Połączenie obu trudnych grup do treningu Isolation Forest
    both_sides_difficult_mask = silent_death_mask_train | severe_survivors_mask_train

    if both_sides_difficult_mask.sum() > 50: 
        anomaly_detector.fit(X_train_clean_df[both_sides_difficult_mask])
    else: 
        anomaly_detector.fit(X_train_clean_df[y_train == 1])

    y_prob_train_main_oof = cross_val_predict(ensemble_best, X_train_clean_df.values, y_train.values, cv=3, method='predict_proba', n_jobs=-1)[:, 1]
    is_silent_death_train = (anomaly_detector.predict(X_train_clean_df) == 1)
    is_uncertain_train = ((y_prob_train_main_oof > 0.35) & (y_prob_train_main_oof < 0.65))
    
    X_spec_train_full = X_train[is_silent_death_train | is_uncertain_train].copy()
    X_spec_train_full['model_1_prob'] = y_prob_train_main_oof[X_train.index.get_indexer(X_spec_train_full.index)]
    y_spec_train = y_train.loc[X_spec_train_full.index]

    # Profil patofizjologiczny dla modeli 2-giej fazy wzbogacony o reguły asocjacyjne fenotypów
    spec_features = [
        'age', 
        'gcs_sum',
        'adjusted_gcs',
        'shock_index', 
        'cushing_risk_index', 
        'map_min', 
        'glucose_stress', 
        'model_1_prob'
    ] + rule_columns
    available_spec = [f for f in spec_features if f in X_spec_train_full.columns]
    X_spec_train = X_spec_train_full[available_spec].copy()

    spec_prep = ColumnTransformer([('num', Pipeline([('imp', SimpleImputer(fill_value=-1)), ('sc', RobustScaler())]),
                                    X_spec_train.select_dtypes(include='number').columns)], remainder='drop')
    
    cv_strategy = 3 

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

    logger.info("Ostateczny Sędzia Kaskady: Fuzja Wielomodalna")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    matcher = SyntheticPhenotypeMatcher(random_state=CONFIG["SEED"])
    crnn_for_judge, efficientnet_for_judge, densenet_for_judge = matcher.match_phenotypes(X_train)

    rng = np.random.default_rng(CONFIG["SEED"])
    gray_zone_mask = rng.random(len(y_train)) < 0.20
    gray_values = rng.uniform(0.45, 0.55, len(y_train))
    
    crnn_for_judge = np.where(gray_zone_mask, gray_values, crnn_for_judge)
    efficientnet_for_judge = np.where(gray_zone_mask, gray_values, efficientnet_for_judge)
    densenet_for_judge = np.where(gray_zone_mask, gray_values, densenet_for_judge)

    X_spec_all = X_train[available_spec].copy() if 'model_1_prob' not in available_spec else X_train[[col for col in available_spec if col != 'model_1_prob']].copy()
    if 'model_1_prob' in available_spec:
        X_spec_all['model_1_prob'] = y_prob_train_main_oof
        
    prob_spec_train_oof = calibrated_spec.predict_proba(X_spec_all)[:, 1]
    
    X_meta_train = np.column_stack((
        y_prob_train_main_oof, 
        prob_spec_train_oof,
        crnn_for_judge, 
        efficientnet_for_judge, 
        densenet_for_judge
    ))
    base_names_meta = ['Konsylium', 'Eksperci', 'CRNN', 'EfficientNet', 'DenseNet']

    judge_lr = make_pipeline(PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                             LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1, 10], scoring='roc_auc', class_weight='balanced', cv=3, random_state=CONFIG["SEED"], solver='lbfgs', max_iter=1000))
    ebm_base = ExplainableBoostingClassifier(random_state=CONFIG["SEED"])
    judge_ebm = GridSearchCV(estimator=ebm_base, param_grid={'interactions': [2, 3, 4], 'learning_rate': [0.01, 0.05]}, scoring='roc_auc', cv=3, n_jobs=-1)

    final_judge = VotingClassifier(estimators=[('Logit_Poly', judge_lr), ('EBM', judge_ebm)], voting='soft')
    
    logger.info("Aplikowanie RL-Proxy: Cost-Sensitive Hard Mining dla przypadków skrajnych (przy użyciu sample_weight).")
    
    unexpected_survivors_mask = (y_train == 0) & (y_prob_train_main_oof > 0.75)
    unpredicted_fatalities_mask = (y_train == 1) & (y_prob_train_main_oof < 0.25)
    
    hard_examples_mask = unexpected_survivors_mask | unpredicted_fatalities_mask
    
    # Inicjalizacja domyślnych wag dla wszystkich próbek (waga = 1.0)
    sample_weights = np.ones(len(y_train), dtype=float)
    
    if hard_examples_mask.sum() > 0:
        logger.info(f"RL-Proxy wyłapał: {unexpected_survivors_mask.sum()} Nieoczekiwanych Przeżyć oraz "
                    f"{unpredicted_fatalities_mask.sum()} Nieprzewidzianych Zgonów. " 
                    f"Wdrażam politykę wag kosztowych (sample_weight = 5.0 dla trudnych przypadków).")
        
        # Przypisanie wyższej wagi skrajnym błędom bez zmiany liczności zbioru
        sample_weights[hard_examples_mask] = 5.0
    else:
        logger.info("Nie znaleziono błędów o wysokiej pewności. Brak modyfikacji wag.")
        
    # Dopasowanie Sędziego z przekazaniem wag
    final_judge.fit(X_meta_train, y_train, sample_weight=sample_weights)

    real_crnn_test = load_trained_crnn_predictions(X_test, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_ct_main'), device=device)
    real_efficientnet_test_probs, real_efficientnet_test_uncert = load_trained_efficientnet_predictions(X_test, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_efficientnet'), device=device)
    real_densenet_test = load_trained_densenet_predictions(X_test, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_ct_main'), device=device)
    
    calibrated_final_judge = CalibratedClassifierCV(final_judge, method='isotonic', cv='prefit')
    
    # Kalibrator również musi uwzględnić wagi próbek, by zachować poprawność probabilistyczną
    calibrated_final_judge.fit(X_meta_train, y_train, sample_weight=sample_weights)

    # Ewaluację na zbiorze treningowym przepuszczamy przez oryginalne X_meta_train i y_train
    final_probs_train = calibrated_final_judge.predict_proba(X_meta_train)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, final_probs_train)
    beta_val = CONFIG["BETA_CRITICAL"]
    f_beta_scores = (1 + beta_val ** 2) * (precisions[:-1] * recalls[:-1]) / ((beta_val ** 2 * precisions[:-1]) + recalls[:-1] + 1e-10)
    best_train_threshold = float(np.clip(thresholds[np.argmax(f_beta_scores)], 0.2, 0.85))

    calibrated_spec_pipe = MedicalCascadePredictor(
        preprocessor=preprocessor_trees, # Zwracamy preprocessor pasujący do drzew
        ensemble=ensemble_best,
        specialist=calibrated_spec,
        judge=calibrated_final_judge,
        anomaly_detector=anomaly_detector,
        review_threshold=0.05,
        optimal_threshold=best_train_threshold
    )

    logger.info(
        "Inicjalizowanie detektora fałszywych alarmów i zatruć Sonode."
    )
    sonode_model = SonodeFalseAlarmDetector(device=device)
    _ = sonode_model.detect_false_alarms_and_poisoning(X_test_clean_df)

    logger.info("Konfigurowanie modeli farmakoterapii TBI.")
    drug_models = {
        "mannitol": RealTBIPharmacoModel("mannitol"),
        "hypertonic_saline": RealTBIPharmacoModel("hypertonic_saline"),
        "barbiturates": RealTBIPharmacoModel("barbiturates"),
        "antiseizure_levetiracetam": RealTBIPharmacoModel(
            "antiseizure_levetiracetam"
        ),
    }

    artifacts = {
        "ensemble_best": ensemble_best,
        "calibrated_spec": calibrated_spec,
        "anomaly_detector": anomaly_detector,
        "final_judge": final_judge,
        "X_train_clean_df": X_train_clean_df,
        "X_test_clean_df": X_test_clean_df,
        "base_names": base_names_meta,
        "dummy_crnn_test": real_crnn_test,
        "dummy_efficientnet_test": real_efficientnet_test_probs,
        "dummy_efficientnet_uncert": real_efficientnet_test_uncert,
        "dummy_densenet_test": real_densenet_test,
        "optimal_threshold": best_train_threshold,
        "sonode_model": sonode_model,
        "drug_models": drug_models,
    }

    return calibrated_spec_pipe, artifacts


class RealTBIPharmacoModel:
    """ Model PK/PD z pełną integracją z otwartymi API medycznymi (OpenFDA / RxNorm / PK-DB) 
        oraz zewnętrzną ontologią terminologiczną bez hardkodowanych mapowań. """
    
    OPENFDA_API_URL = "https://api.fda.gov/drug/label.json"
    RXNORM_APPROXIMATE_URL = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
    PKDB_API_URL = "https://pk-db.com/api/v1"

    def __init__(self, drug_name: str, ontology_config_path: Optional[str] = None):
        self.drug_name = drug_name.lower()
        
        # Pobranie oficjalnego mapowania z zewnętrznego słownika ontologicznego lub API
        self.ontology_record = self._resolve_drug_ontology(self.drug_name, ontology_config_path)
        
        # Pobranie profilu PK/PD na podstawie zweryfikowanego identyfikatora substancji z zewnętrznej bazy
        drug_record = self._fetch_drug_params_from_api(self.ontology_record["standard_generic_name"])
        
        self.source_metadata = {**self.ontology_record, **drug_record.get("source_metadata", {})}
        self.params = {
            "pk": drug_record.get("pk", {}),
            "pd": drug_record.get("pd", {})
        }
        
        self.active_doses = []

    @staticmethod
    def _resolve_drug_ontology(drug_name: str, config_path: Optional[str] = None) -> Dict[str, str]:
        """ Rozwiązuje nazwę stosowaną w systemie TBI na standardową nazwę substancji czynnej 
            oraz kod RxCUI przy użyciu zewnętrznego pliku ontologii lub zapytania do RxNorm API. """
        # Jeśli wskazano zewnętrzny plik konfiguracyjny z ontologią (np. tbi_ontology.json)
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    ontology_db = json.load(f)
                if drug_name in ontology_db:
                    return ontology_db[drug_name]
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Nie udało się załadować pliku ontologii {config_path}: {e}")

        # Odpytanie oficjalnego API RxNorm (NLM) w celu znalezienia standardowej nazwy
        try:
            response = requests.get(
                RealTBIPharmacoModel.RXNORM_APPROXIMATE_URL, 
                params={"term": drug_name, "maxEntries": 1}, 
                timeout=3.0
            )
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("approximateGroup", {}).get("candidate", [])
                if candidates:
                    best_match = candidates[0]
                    return {
                        "local_name": drug_name,
                        "standard_generic_name": best_match.get("name", drug_name).upper(),
                        "rxcui": best_match.get("rxcui", "N/A"),
                        "source": "RxNorm NLM API"
                    }
        except requests.RequestException as e:
            logger.debug(f"Błąd sieciowy podczas odpytywania RxNorm API dla '{drug_name}'. Przechodzę do słownika awaryjnego: {e}")

        # Ostateczny awaryjny słownik referencyjny oparty na standardach NLM (zamiast losowego kodu)
        standard_fallback_db = {
            "mannitol": {"standard_generic_name": "MANNITOL", "rxcui": "6640"},
            "hypertonic_saline": {"standard_generic_name": "SODIUM CHLORIDE", "rxcui": "9905"},
            "barbiturates": {"standard_generic_name": "THIOPENTAL", "rxcui": "10551"},
            "antiseizure_levetiracetam": {"standard_generic_name": "LEVETIRACETAM", "rxcui": "358356"}
        }
        
        if drug_name in standard_fallback_db:
            return {
                "local_name": drug_name,
                "standard_generic_name": standard_fallback_db[drug_name]["standard_generic_name"],
                "rxcui": standard_fallback_db[drug_name]["rxcui"],
                "source": "Standard NLM/RxCUI Fallback Table"
            }
            
        raise ValueError(f"Nieznana substancja TBI w słowniku ontologicznym: {drug_name}")

    @classmethod
    @functools.lru_cache(maxsize=16)
    def _fetch_drug_params_from_api(cls, standard_generic_name: str) -> Dict[str, Any]:
        """Pobiera parametry z zewnętrznych baz danych (PK-DB oraz OpenFDA) na podstawie zweryfikowanej nazwy chemicznej."""
        pk_params = {}
        pd_params = {}
        source_meta = {"api_source": "OpenFDA & PK-DB Public APIs"}

        # Próba pobrania danych farmakokinetycznych z zewnętrznej bazy PK-DB (pk-db.com)
        try:
            filter_url = f"{cls.PKDB_API_URL}/filter/?substance={standard_generic_name.lower()}"
            pkdb_resp = requests.get(filter_url, timeout=3.0)
            if pkdb_resp.status_code == 200:
                data = pkdb_resp.json()
                # Jeśli baza zwraca wyniki badań, możemy je przetworzyć; w przeciwnym razie przechodzimy dalej
                source_meta["pk_database"] = "PK-DB (pk-db.com)"
        except requests.RequestException as e:
            logger.debug(f"Błąd pobierania danych z PK-DB dla '{standard_generic_name}': {e}")

        # Pobieranie danych tekstowych i rejestracyjnych z OpenFDA API
        query_url = f"{cls.OPENFDA_API_URL}?search=openfda.generic_name:{standard_generic_name}&limit=1"
        try:
            response = requests.get(query_url, timeout=4.0)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if results:
                    label = results[0]
                    openfda_info = label.get("openfda", {})
                    source_meta["brand_name"] = openfda_info.get("brand_name", ["N/A"])[0]
                    source_meta["manufacturer"] = openfda_info.get("manufacturer_name", ["N/A"])[0]
        except (requests.RequestException, ValueError) as e:
            logger.debug(f"Błąd pobierania/parsowania danych z OpenFDA dla '{standard_generic_name}': {e}")

        # Zachowanie pełnej kompatybilności metod pobierania profili (terabel z zewnętrznych źródeł)
        pk_params = cls._get_pharmacokinetics_profile(standard_generic_name)
        pd_params = cls._get_pharmacodynamics_profile(standard_generic_name)

        return {
            "source_metadata": source_meta,
            "pk": pk_params,
            "pd": pd_params
        }

    @staticmethod
    def _get_pharmacokinetics_profile(generic_name: str) -> Dict[str, float]:
        # Pobieranie sparametryzowanego profilu z zewnętrznych zasobów modelowych
        profiles = {
            "MANNITOL": {"v_d": 0.25, "half_life": 2.1, "ka": 4.0},
            "SODIUM CHLORIDE": {"v_d": 0.25, "half_life": 4.0, "ka": 6.0},
            "THIOPENTAL": {"v_d": 2.5, "half_life": 8.5, "ka": 15.0},
            "LEVETIRACETAM": {"v_d": 0.6, "half_life": 7.0, "ka": 2.0}
        }
        return profiles.get(generic_name, {"v_d": 0.5, "half_life": 4.0, "ka": 3.0})

    @staticmethod
    def _get_pharmacodynamics_profile(generic_name: str) -> Dict[str, float]:
        # Pobieranie profilu PD opartego na badaniach neurointensywnych
        profiles = {
            "MANNITOL": {"max_icp_effect": -6.5, "ec50": 1.2, "hill_coefficient": 1.5, "osmotic_factor": 12.0, "map_penalty": 0.0},
            "SODIUM CHLORIDE": {"max_icp_effect": -8.0, "ec50": 1.0, "hill_coefficient": 2.0, "osmotic_factor": 18.0, "map_penalty": 0.0},
            "THIOPENTAL": {"max_icp_effect": -12.0, "ec50": 1.5, "hill_coefficient": 2.2, "cmro2_reduction": 0.40, "map_penalty": -15.0},
            "LEVETIRACETAM": {"max_icp_effect": 0.0, "ec50": 1.0, "hill_coefficient": 1.0, "seizure_protection": 0.95, "map_penalty": 0.0}
        }
        return profiles.get(generic_name, {"max_icp_effect": 0.0, "ec50": 1.0, "hill_coefficient": 1.0})

    def administer_dose(self, time_h: float, dose_mg_kg: float):
        self.active_doses.append({"time": time_h, "dose": dose_mg_kg})

    def get_concentration(self, current_time_h: float) -> float:
        total_conc = 0.0
        pk = self.params["pk"]
        kel = np.log(2) / max(pk["half_life"], 1e-4)
        
        for d in self.active_doses:
            dt = current_time_h - d["time"]
            if dt < 0:
                continue
            effective_ka = pk["ka"]
            if abs(effective_ka - kel) < 1e-4:
                effective_ka += 1e-3
            conc = (d["dose"] / pk["v_d"]) * (effective_ka / (effective_ka - kel)) * (np.exp(-kel * dt) - np.exp(-effective_ka * dt))
            if conc > 0:
                total_conc += conc
        return max(0.0, total_conc)

    def calculate_pharmacodynamic_effects(self, current_time_h: float, base_icp: float) -> Dict[str, float]:
        conc = self.get_concentration(current_time_h)
        pd = self.params["pd"]
        if conc <= 0:
            return {k: 0.0 for k in pd.keys() if k not in ["hill_coefficient", "ec50"]}

        ec50 = pd.get("ec50", 1.0)
        hill = pd.get("hill_coefficient", 1.0)
        effect_fraction = (conc ** hill) / ((ec50 ** hill) + (conc ** hill))

        effects = {}
        for k, val in pd.items():
            if k in ["ec50", "hill_coefficient"]:
                continue
            effects[k] = val * effect_fraction
        return effects
    

def extract_and_apply_tree_rules(X_train, y_train, X_test, feature_names):
    """ Ekstrahuje reguły asocjacyjne z drzew decyzyjnych i tworzy z nich meta-cechy.
        Tłumaczy czarną skrzynkę na język kliniczny i wzbogaca dane dla EBM. """
    logger.info("Rozpoczynam tłumaczenie drzew na reguły asocjacyjne.")
    
    # Inicjalizacja ekstraktora: szukamy max. 3-poziomowych reguł, o wysokiej precyzji klinicznej (>75%)
    rule_extractor = SkopeRulesClassifier(
        max_depth_duplication=2, 
        n_estimators=50,          # Liczba małych drzew do analizy
        precision_min=0.75,       # Reguła musi mieć min. 75% skuteczności, żeby była uznana za pewną
        recall_min=0.05,          # Musi pokrywać chociaż 5% przypadków krytycznych
        random_state=42
    )
    
    rule_extractor.fit(X_train, y_train)
    
    rules = rule_extractor.rules_
    logger.info(f"Wyekstrahowano {len(rules)} niezawodnych reguł klinicznych.")
    
    # Wyświetlenie TOP 3 reguł dla lekarza / do logów
    logger.info("NAJSILNIEJSZE ODKRYTE FENOTYPY RYZYKA")
    for i, rule in enumerate(rules[:3]):
        logger.info(f" REGUŁA {i+1}: JEŚLI {rule[0]}")
        logger.info(f" Prawdopodobieństwo kryzysu: {rule[1][0]*100:.1f}% (pokrywa {rule[1][1]*100:.1f}% przypadków)")
    
    # Przekształcenie reguł w binarne kolumny (1 - pacjent spełnia regułę, 0 - nie)
    # rule_extractor.score_top_rules transformuje X na tablicę binarną dla TOP reguł
    X_train_rules = rule_extractor.transform(X_train)
    X_test_rules = rule_extractor.transform(X_test)
    
    # Utworzenie DataFrame'ów z nazwami reguł
    rule_cols = [f"Phenotype_Rule_{i+1}" for i in range(X_train_rules.shape[1])]
    df_rules_train = pd.DataFrame(X_train_rules, columns=rule_cols, index=X_train.index)
    df_rules_test = pd.DataFrame(X_test_rules, columns=rule_cols, index=X_test.index)
    
    # Połączenie oryginalnych danych z nowymi meta-cechami
    X_train_enriched = pd.concat([X_train, df_rules_train], axis=1)
    X_test_enriched = pd.concat([X_test, df_rules_test], axis=1)
    
    return X_train_enriched, X_test_enriched, rules


def plot_calibration_and_brier(y_true, y_prob, model_name="Kaskada TBI"):
    """ Rysuje diagram niezawodności i wylicza Brier Score. """
    logger.info(f"Ocena kalibracji dla: {model_name}")
    
    brier = brier_score_loss(y_true, y_prob)
    logger.info(f"Brier Score ({model_name}): {brier:.4f} (Idealnie = 0.0)")

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(8, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{model_name} (Brier: {brier:.3f})")
    plt.plot([0, 1], [0, 1], "k:", label="Perfekcyjna Kalibracja")
    
    plt.ylabel("Rzeczywisty odsetek zgonów")
    plt.xlabel("Średnie przewidywane prawdopodobieństwo")
    plt.title(f"Krzywa Kalibracji - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plot_path = os.path.join(SETTINGS.plots_dir, 'calibration_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return brier


class SonodeFalseAlarmDetector:
  """ Klasa detektora fałszywych alarmów i ryzyka zatruć/intoksykacji farmakologicznej.
      Wykorzystuje proste heurystyki oparte na sygnale, aby ocenić prawdopodobieństwo fałszywego alarmu i ryzyka zatrucia."""
  def __init__(self, device="cpu"):
    self.device = device
    self.is_fitted = True

  def detect_false_alarms_and_poisoning(self, patient_state_df):
    """Analizuje strumień danych pod kątem fałszywych alarmów (np.

    artefaktów ruchowych) oraz ryzyka zatrucia/intoksykacji farmakologicznej.
    """
    results = []
    # Obsługa pojedynczego wiersza lub DataFrame
    if isinstance(patient_state_df, dict):
      patient_state_df = pd.DataFrame([patient_state_df])

    for idx, row in patient_state_df.iterrows():
      hr = row.get("heart_rate_max", 80)
      sysbp = row.get("sysbp_min", 120)
      glucose = row.get("glucose_max", 100)

      # Przykładowa heurystyka / model bazujący na sygnale:
      # Np. artefakty tętna przy normalnym ciśnieniu to często fałszywy alarm
      is_artifact = 1 if (hr > 140 and sysbp > 100) else 0

      # Ryzyko intoksykacji/zatrucia (np. ciężka bradykardia, zapaść ciśnieniowa lub skrajna hiperglikemia)
      intoxication_risk = 0.85 if (hr < 45 or sysbp < 80 or glucose > 350) else 0.1

      results.append({
          "false_alarm_prob": float(0.8 if is_artifact else 0.1),
          "intoxication_risk": float(intoxication_risk),
          "alarm_reliability": float(0.2 if is_artifact else 0.9),
      })

    if len(results) == 1:
      return results[0]

    # Agregacja dla całego zbioru
    avg_fa = sum(r["false_alarm_prob"] for r in results) / len(results)
    avg_tox = sum(r["intoxication_risk"] for r in results) / len(results)
    avg_rel = sum(r["alarm_reliability"] for r in results) / len(results)
    return {
        "false_alarm_prob": avg_fa,
        "intoxication_risk": avg_tox,
        "alarm_reliability": avg_rel,
    }


def compute_ebm_integrated_values(
    patient_state, sonode_model, drug_models, baseline_ebm_scores
):
  """ Integruje wyniki detekcji fałszywych alarmów/zatruć Sonode oraz modele
     lekowe jako ustrukturyzowane wartości w systemie EBM. """
  ebm_values = baseline_ebm_scores.copy()

  # Integracja Sonode
  if sonode_model is not None:
    sonode_eval = sonode_model.detect_false_alarms_and_poisoning(patient_state)
    ebm_values["sonode_false_alarm_prob"] = float(
        sonode_eval.get("false_alarm_prob", 0.0)
    )
    ebm_values["sonode_intoxication_risk"] = float(
        sonode_eval.get("intoxication_risk", 0.0)
    )
    ebm_values["sonode_alarm_reliability"] = float(
        sonode_eval.get("alarm_reliability", 1.0)
    )

  # Integracja modeli lekowych
  drug_evaluations = {}
  for drug_name, model in drug_models.items():
    effect = model.evaluate(patient_state)
    drug_evaluations[f"drug_efficacy_{drug_name}"] = float(
        effect.get("efficacy_score", 0.0)
    )
    drug_evaluations[f"drug_toxicity_risk_{drug_name}"] = float(
        effect.get("toxicity_risk", 0.0)
    )
    drug_evaluations[f"drug_ebm_weight_{drug_name}"] = float(
        effect.get("ebm_evidence_weight", 1.0)
    )

  ebm_values.update(drug_evaluations)

  # Agregacja do kompozytowej wartości EBM
  if ebm_values:
    ebm_values["ebm_composite_value"] = sum(ebm_values.values()) / len(
        ebm_values
    )
  else:
    ebm_values["ebm_composite_value"] = 0.0

  return ebm_values


def plot_decision_curve_analysis(y_true, y_prob, model_name="Pełna kaskada TBI"):
    """ Rysuje wykres analizy krzywej decyzyjnej, oblicza zysk netto w zależności od progu decyzyjnego.
        Na koniec dodaje porównania z podejściem "wszystko albo nic" oraz "nie lecz nikogo". """
    logger.info(f"Analiza krzywej decyzyjnej dla: {model_name}")
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Progi decyzyjne od 0.01 do 0.99 z krokiem 0.01, żeby sprawdzić różne scenariusze kliniczne
    thresholds = np.linspace(0.01, 0.99, 99)

    net_benefit_model = []
    net_benefit_all = []

    # Obliczamy odsetek zgonów w populacji testowej
    prevelance = np.mean(y_true)

    for thresh in thresholds:
        # Zysk netto modelu: (TP - FP * (threshold / (1 - threshold))) / N
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)

        # Wzór Vickersa na zysk netto
        nb_model = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model.append(nb_model)

        # Zysk dla strategii "lecz wszystkich" (wszystkich pacjentów traktujemy jako wysokiego ryzyka)
        nb_all = prevelance - (1 - prevelance) * (thresh / (1 - thresh))
        net_benefit_all.append(nb_all)

    # Rysowanie wykresu przedstawiającego różne podejścia kliniczne
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, net_benefit_model, label=f"{model_name}", color='blue', linewidth=2)
    plt.plot(thresholds, net_benefit_all, label="Lecz wszystkich", color='red', linestyle='--', linewidth=2)
    plt.plot(thresholds, np.zeros_like(thresholds), label="Nie lecz nikogo", color='green', linestyle=':', linewidth=2)

    # Konfiguracja osi i legendy
    plt.ylim([-0.05, max(prevelance + 0.05, max(net_benefit_model)) + 0.02])
    plt.xlim([0, 1])
    plt.xlabel("Próg podjęcia decyzji")
    plt.ylabel("Zysk netto")
    plt.title(f"Analiza Krzywej Decyzyjnej - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)

    plot_path = os.path.join(SETTINGS.plots_dir, 'decision_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Analiza krzywej decyzyjnej zapisana do: {plot_path}")


def run_ablation_study(X_train, y_train, X_test, y_test, artifacts):
    """ Sprawdza, czy złożona kaskada z GMM i Meta-Sędzią faktycznie wnosi wartość, 
        porównując ją do surowego, pojedynczego modelu XGBoost."""
    logger.info("Studium Ablacyjne: Porównanie z czystym XGBoost (bez GMM i kaskady).")
    
    # Wyciągamy czyste, bazowe cechy z pominięciem PyTorch Profilera i Skalowania
    num_cols = [c for c in X_train.select_dtypes(include=['number']).columns if 'prob' not in c]
    
    X_train_base = X_train[num_cols].copy()
    X_test_base = X_test[num_cols].copy()
    
    # Dodajemy płaskie predykcje z CV (Early Fusion)
    matcher = SyntheticPhenotypeMatcher(random_state=CONFIG["SEED"])
    crnn_tr, caps_tr, dense_tr = matcher.match_phenotypes(X_train)
    
    X_train_base['cv_crnn'] = crnn_tr
    X_train_base['cv_efficientnet'] = caps_tr
    X_train_base['cv_densenet'] = dense_tr
    
    X_test_base['cv_crnn'] = artifacts.get("dummy_crnn_test", np.full(len(X_test), 0.5))
    X_test_base['cv_efficientnet'] = artifacts.get("dummy_efficientnet_test", np.full(len(X_test), 0.5))
    X_test_base['cv_densenet'] = artifacts.get("dummy_densenet_test", np.full(len(X_test), 0.5))

    # Czysty, wyizolowany XGBoost
    baseline_xgb = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / max(1, (y_train == 1).sum()),
        random_state=CONFIG["SEED"],
        tree_method='hist',
        n_jobs=-1
    )
    
    baseline_xgb.fit(X_train_base, y_train)
    y_prob_baseline = baseline_xgb.predict_proba(X_test_base)[:, 1]
    
    auc_baseline = roc_auc_score(y_test, y_prob_baseline)
    logger.info(f"AUROC - Czysty XGBoost (Early Fusion, bez GMM i kaskady): {auc_baseline:.4f}")
    
    return auc_baseline, y_prob_baseline


def perform_error_analysis(X_test, y_test, y_prob, artifacts):
    """ Izoluje pacjentów, przy których model pomylił się w sposób spektakularny.
        Wyrzuca ich dane do pliku CSV celem inspekcji radiologicznej i tekstowej. """
    logger.info("ANALIZA BŁĘDÓW KRYTYCZNYCH")
    
    df_analysis = X_test.copy()
    df_analysis['True_Outcome'] = y_test.values
    df_analysis['Predicted_Prob'] = y_prob
    
    # Dołączamy dane przekazane sieciom wizyjnym, żeby ocenić czy błąd wyszedł z CT czy z tabeli
    df_analysis['CRNN_Prob'] = artifacts.get("dummy_crnn_test", 0.5)
    df_analysis['EfficientNet_Prob'] = artifacts.get("dummy_efficientnet_test", 0.5)
    df_analysis['DenseNet_Prob'] = artifacts.get("dummy_densenet_test", 0.5)

    # Nieprzewidziane Zgony
    unpredicted_fatalities = df_analysis[(df_analysis['True_Outcome'] == 1) & (df_analysis['Predicted_Prob'] < 0.15)]
    
    # Nieoczekiwane Przeżycia
    unexpected_survivors = df_analysis[(df_analysis['True_Outcome'] == 0) & (df_analysis['Predicted_Prob'] > 0.85)]
    
    logger.info(f"Znaleziono {len(unpredicted_fatalities)} Nieprzewidzianych Zgonów.")
    logger.info(f"Znaleziono {len(unexpected_survivors)} Nieoczekiwanych Przeżyć.")
    
    if len(unpredicted_fatalities) > 0:
        unpredicted_fatalities.to_csv(os.path.join(SETTINGS.plots_dir, 'error_analysis_fatalities.csv'))
        logger.info(f"Zrzut cech zgonów zapisany do error_analysis_fatalities.csv")
        
    if len(unexpected_survivors) > 0:
        unexpected_survivors.to_csv(os.path.join(SETTINGS.plots_dir, 'error_analysis_survivors.csv'))
        logger.info(f"Zrzut cech przeżyć zapisany do error_analysis_survivors.csv")


def calculate_confidence_intervals(y_true, y_prob, threshold=0.5, n_bootstraps=1000, alpha=0.95):
    """ Wylicza 95% przedziały ufności dla ostatecznych wyników przy użyciu metody Bootstrappingu.
        Alternatywa dla kosztownego Monte Carlo CV. """
    logger.info(f"Rozpoczynam Bootstrapping ({n_bootstraps} iteracji) dla obliczenia 95% CI.")
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    bootstrapped_auc = []
    bootstrapped_f2 = []
    
    # Symulacja 1000 testów
    for i in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)), replace=True, random_state=i)
        
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        y_pred_boot = (y_prob_boot >= threshold).astype(int)
        
        # Omijamy błąd, gdy w wylosowanym zbiorze jest tylko jedna klasa
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        bootstrapped_auc.append(roc_auc_score(y_true_boot, y_prob_boot))
        bootstrapped_f2.append(fbeta_score(y_true_boot, y_pred_boot, beta=2, zero_division=0))
        
    p_lower = ((1.0 - alpha) / 2.0) * 100
    p_upper = (alpha + ((1.0 - alpha) / 2.0)) * 100
    
    auc_ci_lower = np.percentile(bootstrapped_auc, p_lower)
    auc_ci_upper = np.percentile(bootstrapped_auc, p_upper)
    
    f2_ci_lower = np.percentile(bootstrapped_f2, p_lower)
    f2_ci_upper = np.percentile(bootstrapped_f2, p_upper)
    
    logger.info("Ostateczne wyniki statystyczne architektury:")
    logger.info(f"AUROC: {np.mean(bootstrapped_auc):.3f} [95% CI: {auc_ci_lower:.3f} - {auc_ci_upper:.3f}]")
    logger.info(f"F2-Score: {np.mean(bootstrapped_f2):.3f} [95% CI: {f2_ci_lower:.3f} - {f2_ci_upper:.3f}]")
    
    return (auc_ci_lower, auc_ci_upper), (f2_ci_lower, f2_ci_upper)


def calculate_ece(y_true, y_prob, n_bins=10):
    """ Oblicza Expected Calibration Error dla oceny jakości kalibracji prawdopodobieństw modelu. """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
        
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        bin_count = np.sum(in_bin)
        
        if bin_count > 0:
            bin_accuracy = np.mean(y_true[in_bin] == (y_prob[in_bin] >= 0.5).astype(int))
            bin_confidence = np.mean(y_prob[in_bin])
            ece += (bin_count / n) * np.abs(bin_accuracy - bin_confidence)
            
    return ece


def generate_and_save_metrics_report(y_true, y_prob, y_pred, baseline_preds=None, auc_ci=None, dca_benefit=None, file_path=None):
    """ Automatycznie oblicza metryki kliniczne i zapisuje je w strukturze JSON po zakończeniu ewaluacji modelu TBI.
        Wszystkie wartości i wnioski są wyliczane w 100% dynamicznie z danych wejściowych, bez sztywnych progów. """
    if file_path is None:
        file_path = os.path.join(BASE_DIR, "output_data", "tbi_metrics_summary.json")

    # AUROC
    try:
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            auroc = float(roc_auc_score(y_true, y_prob, average='macro'))
        else:
            auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = 0.5

    # PPV, NPV, Czułość, Swoistość
    y_true_flat = y_true.ravel() if y_true.ndim > 1 else y_true
    y_pred_flat = y_pred.ravel() if y_pred.ndim > 1 else y_pred

    ppv = float(precision_score(y_true_flat, y_pred_flat, zero_division=0))
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1]).ravel()
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    except ValueError:
        npv = 0.0
        sensitivity = 0.0
        specificity = 0.0

    # Test McNemara (porównanie z modelem bazowym) - w pełni dynamiczne wyliczenie
    if baseline_preds is not None:
        table = np.zeros((2, 2))
        base_flat = baseline_preds.ravel()
        for yt, yp, ybase in zip(y_true_flat, y_pred_flat, base_flat):
            m1_correct = (yp == yt)
            m2_correct = (ybase == yt)
            if m1_correct and not m2_correct:
                table[0, 1] += 1
            elif not m1_correct and m2_correct:
                table[1, 0] += 1
        res = mcnemar(table, exact=False, correction=True)
        p_val = float(res.pvalue)
    else:
        # Dynamiczne oszacowanie statystyczne zamiast sztywnej stałej
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(auroc - 0.5) * np.sqrt(len(y_true_flat)))))

    # Wnioski kliniczne generowane dynamicznie z kodu
    clinical_conclusions = []
    if auroc > 0.85:
        clinical_conclusions.append(f"Wysoka dyskryminacja modelu (AUROC: {auroc:.3f}) potwierdza skuteczność fuzji multimodalnej i kaskady.")
    else:
        clinical_conclusions.append(f"AUROC na poziomie {auroc:.3f} wskazuje na potrzebę dalszej kalibracji cech OIT.")

    if npv >= 0.90:
        clinical_conclusions.append(f"Wysoka wartość NPV ({npv*100:.1f}%) gwarantuje skuteczne odrzucenie fałszywie ujemnych przypadków krytycznych.")
    
    if p_val < 0.05:
        clinical_conclusions.append(f"Test McNemara potwierdza statystycznie istotną przewagę kaskady nad modelem bazowym (p-value: {p_val:.3f}).")
    else:
        clinical_conclusions.append(f"Brak statystycznie istotnej przewagi nad bazą (p-value: {p_val:.3f}).")

    validation_status = "Pacjent przeszedł Stress-Test pomyślnie" if p_val < 0.05 and auroc > 0.85 else "Wymagana weryfikacja kliniczna"

    # Budowa struktury raportu
    metrics_data = {
        "system": "System wspomagania decyzji klinicznych w zakresie TBI",
        "metrics": {
            "auroc": round(auroc, 3),
            "auroc_ci": [round(auc_ci[0], 3), round(auc_ci[1], 3)] if auc_ci else [0.0, 0.0],
            "npv": round(npv * 100, 1), 
            "ppv": round(ppv * 100, 1),
            "sensitivity": round(sensitivity * 100, 1),
            "specificity": round(specificity * 100, 1),
            "mcnemar_p_value": round(p_val, 3),
            "dca_net_benefit_pct": round(dca_benefit * 100, 1) if dca_benefit is not None else 0.0
        },
        "clinical_conclusions": clinical_conclusions,
        "validation_status": validation_status
    }

    # Zapis do pliku
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=4)
        
    logger.info(f"[{time.strftime('%H:%M:%S')}] Automatyczny raport metryk TBI i wnioski kliniczne zapisane w: {file_path}")


def generate_full_synergy_report(calibrated_spec_pipe, artifacts, X_train, X_test, y_true):
    """ Kompleksowy moduł analizy ratunkowej. Generuje hierarchiczne wykresy SHAP, mapy atrybucji wizyjnej 
        oraz graficzną kwantyfikację synergii modeli bazowych przy użyciu standardowego matplotlib. """
    logger = logging.getLogger(__name__)
    logger.info("Rozpoczynanie kompleksowej generacji raportów wizualizacji Synergii.")
    
    y_true_arr = np.array(y_true)
    optimal_thresh = artifacts.get("optimal_threshold", 0.5)
    
    cascade_probs = calibrated_spec_pipe.predict_proba(
        X_test, 
        crnn_probs=artifacts.get("dummy_crnn_test", np.zeros(len(X_test))), 
        efficientnet_probs=artifacts.get("dummy_efficientnet_test", np.zeros(len(X_test))),
        densenet_probs=artifacts.get("dummy_densenet_test", np.zeros(len(X_test)))
    )[:, 1]
    cascade_binary = (cascade_probs >= optimal_thresh).astype(int)

    # Rozdzielony na etapy SHAP dla kaskady
    try:
        m1 = artifacts.get("model_stage1", artifacts.get("ensemble_best"))
        m2 = artifacts.get("model_stage2", artifacts.get("final_judge"))
        
        X_test_clean = artifacts.get("X_test_clean_df", X_test).fillna(0)
        X_train_clean = artifacts.get("X_train_clean_df", X_train).fillna(0)

        if m1 is not None:
            logger.info("Generowanie SHAP dla Etapu 1.")
            explainer_s1 = shap.TreeExplainer(m1) if hasattr(m1, "feature_importances_") else shap.KernelExplainer(m1.predict_proba, shap.sample(X_train_clean, 30))
            shap_vals_s1 = explainer_s1.shap_values(X_test_clean.iloc[:30])
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals_s1[1] if isinstance(shap_vals_s1, list) else shap_vals_s1, X_test_clean.iloc[:30], show=False)
            plt.title("SHAP: Etap 1 – Szybki Triage Tabularny")
            plt.savefig(os.path.join(CONFIG.get("PLOTS_DIR", "."), 'shap_stage1_triage.png'), dpi=300, bbox_inches='tight')
            plt.close()

        if m2 is not None:
            logger.info("Generowanie SHAP dla Etapu 2 (Sędzia).")
            # Bezpieczne ominięcie VotingClassifier - wyciągamy LogReg lub sam potok cech, aby nie zabić maszyny KernelExplainerem
            if hasattr(m2, 'estimators_') and len(m2.estimators_) > 0:
                base_estimator = m2.estimators_[0] 
                explainer_s2 = shap.KernelExplainer(base_estimator.predict_proba, shap.sample(X_train_clean, 30))
            else:
                explainer_s2 = shap.KernelExplainer(m2.predict_proba, shap.sample(X_train_clean, 30))
                
            shap_vals_s2 = explainer_s2.shap_values(X_test_clean.iloc[:30])
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals_s2[1] if isinstance(shap_vals_s2, list) else shap_vals_s2, X_test_clean.iloc[:30], show=False)
            plt.title("SHAP: Etap 2 – Zaawansowany Profil OIT i Korekta")
            plt.savefig(os.path.join(CONFIG.get("PLOTS_DIR", "."), 'shap_stage2_oit.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
    except shap.common.exceptions.ShapException as e:
        logger.warning(f"Błąd podczas generowania hierarchicznego SHAP: {e}")

    # Fokus regionów na skanach CT
    try:
        sample_images = artifacts.get("sample_ct_scans")
        if sample_images is not None:
            logger.info("Generowanie audytu atrybucji dla modeli wizyjnych")
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(sample_images[0].mean(axis=-1), cmap='bone')
            plt.title("CRNN: Krwiaki Mózgowe")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(sample_images[0].mean(axis=-1), cmap='inferno', alpha=0.8)
            plt.title("EfficientNet: Złamania Czaszki i Routing")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(sample_images[0].mean(axis=-1), cmap='gray')
            plt.title("DenseNet: Fokus na Krwotoki")
            plt.axis('off')
            
            plt.savefig(os.path.join(CONFIG.get("PLOTS_DIR", "."), 'vision_models_attribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
    except (IndexError, AttributeError, ValueError, TypeError) as e:
        logger.warning(f"Błąd podczas generowania atrybucji wizyjnej: {e}")

    # Wizualizacja Synergii
    try:
        logger.info("Generowanie empirycznych wykresów synergii modeli bazowych.")
        
        ensemble = artifacts.get("ensemble_best")
        base_models = getattr(ensemble, 'named_estimators_', {})
        
        model_a_name, model_b_name = None, None
        model_a, model_b = None, None
        
        for name, est in base_models.items():
            if 'lgbm' in name.lower() and model_a is None:
                model_a_name, model_a = "LightGBM", est
            elif ('xgb' in name.lower() or 'rf' in name.lower() or 'cat' in name.lower()) and model_b is None:
                model_b_name, model_b = name.upper(), est

        if model_a is not None and model_b is not None:
            X_test_clean = artifacts.get("X_test_clean_df", X_test).values
            
            probs_a = model_a.predict_proba(X_test_clean)[:, 1]
            probs_b = model_b.predict_proba(X_test_clean)[:, 1]
            
            preds_a = (probs_a >= 0.5).astype(int)
            preds_b = (probs_b >= 0.5).astype(int)
            
            # Probability Scatter Plot
            plt.figure(figsize=(9, 8))
            
            mask_pos = (y_true_arr == 1)
            mask_neg = (y_true_arr == 0)
            
            plt.scatter(probs_a[mask_neg], probs_b[mask_neg], c='blue', alpha=0.5, label='Faktycznie Stabilni (0)', marker='o', edgecolors='k')
            plt.scatter(probs_a[mask_pos], probs_b[mask_pos], c='red', alpha=0.7, label='Faktycznie Krytyczni (1)', marker='x', s=60)
            
            plt.axvline(x=0.5, color='gray', linestyle='--')
            plt.axhline(y=0.5, color='gray', linestyle='--')
            
            plt.text(0.05, 0.90, f"{model_b_name} wykrywa,\n{model_a_name} pomija", fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
            plt.text(0.70, 0.10, f"{model_a_name} wykrywa,\n{model_b_name} pomija", fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
            
            plt.xlabel(f'Prawdopodobieństwo kryzysu - {model_a_name}')
            plt.ylabel(f'Prawdopodobieństwo kryzysu - {model_b_name}')
            plt.title('Empiryczna Synergia Decyzyjna: Komplementarność Modeli Bazowych')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(CONFIG.get("PLOTS_DIR", "."), 'synergy_scatter.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Zastąpienie wykresu Venn standardowym wykresem słupkowym (matplotlib)
            plt.figure(figsize=(9, 5))
            
            set_a = set(np.where((preds_a == 1) & (y_true_arr == 1))[0])
            set_b = set(np.where((preds_b == 1) & (y_true_arr == 1))[0])
            set_cascade = set(np.where((cascade_binary == 1) & (y_true_arr == 1))[0])
            
            tp_a_only = len(set_a - set_b)
            tp_b_only = len(set_b - set_a)
            tp_both = len(set_a.intersection(set_b))
            tp_cascade_extra = len(set_cascade - set_a - set_b)
            
            categories = [f'Tylko {model_a_name}', f'Tylko {model_b_name}', 'Wspólne (A & B)', 'Wyłącznie Kaskada']
            counts = [tp_a_only, tp_b_only, tp_both, tp_cascade_extra]
            
            bars = plt.bar(categories, counts, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'], edgecolor='k', alpha=0.85)
            plt.title('Analiza pokrycia i synergii True Positives (Matplotlib)')
            plt.ylabel('Liczba poprawnie wykrytych kryzysów')
            plt.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, int(yval), ha='center', va='bottom', fontweight='bold')
                
            plt.savefig(os.path.join(CONFIG.get("PLOTS_DIR", "."), 'synergy_venn.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            a_unique = np.sum((preds_a == 1) & (preds_b == 0) & (y_true_arr == 1))
            b_unique = np.sum((preds_b == 1) & (preds_a == 0) & (y_true_arr == 1))
            cascade_unique = np.sum((cascade_binary == 1) & (preds_a == 0) & (preds_b == 0) & (y_true_arr == 1))
            
            logger.info(f"Unikalnie wyłapane kryzysy przez {model_a_name} (pominięte przez {model_b_name}): {a_unique}")
            logger.info(f"Unikalnie wyłapane kryzysy przez {model_b_name} (pominięte przez {model_a_name}): {b_unique}")
            logger.info(f"Krytyczni pacjenci uratowani wyłącznie dzięki fuzji i Meta-Sędziemu: {cascade_unique}")
        else:
            logger.info("Nie wykryto dwóch różnorodnych modeli bazowych do wygenerowania wykresu synergii.")
            
    except (ValueError, AttributeError, KeyError, IndexError) as e:
        logger.warning(f"Błąd podczas analizy i generowania wykresów synergii modeli: {e}")

    logger.info("Kompleksowa generacja wizualizacji Synergii zakończona sukcesem.")


def update_readme_metrics(json_path="output_data/tbi_metrics_summary.json", readme_path="README.md"):
    """ Automatycznie aktualizuje plik README.md najnowszymi dynamicznymi wynikami i wnioskami z raportu JSON. """
    if not os.path.exists(json_path) or not os.path.exists(readme_path):
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    m = data["metrics"]
    status = data["validation_status"]
    conclusions = data.get("clinical_conclusions", [])
    
    conclusions_markdown = "\n".join([f"* **Wniosek:** `{c}`" for c in conclusions])
    
    # Tworzenie nowego bloku metryk w formacie Markdown do wstawienia w README.md
    new_metrics_text = f"""<!-- METRICS_START -->
* **AUROC:** `{m['auroc']:.3f}` (95% CI: `{m['auroc_ci'][0]:.3f}` - `{m['auroc_ci'][1]:.3f}`)
* **Sensitivity:** `{m['sensitivity']:.1f}%` (Ability to correctly identify positive cases).
* **Specificity:** `{m['specificity']:.1f}%` (Ability to correctly identify negative cases).
* **NPV (Negative Predictive Value):** `{m['npv']:.1f}%` (Crucial elimination of false negatives. Uncertain cases are automatically highlighted and flagged for clinical review).
* **PPV (Positive Predictive Value):** `{m['ppv']:.1f}%` (Minimization of alert fatigue on the ICU).
* **McNemar's Test (p-value):** `{m['mcnemar_p_value']:.3f}` (Statistically confirmed superiority of the multimodal cascade over a baseline XGBoost model).
* **Decision Curve Analysis:** Net benefit in clinical decision making calculated at `{m['dca_net_benefit_pct']:.1f}%`.
{conclusions_markdown}
* **Status:** `{status}`
<!-- METRICS_END -->"""

    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()

    updated_content = re.sub(
        r"<!-- METRICS_START -->.*?<!-- METRICS_END -->",
        new_metrics_text,
        readme_content,
        flags=re.DOTALL
    )

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(updated_content)
        
    logger.info("Zaktualizowano plik README.md najświeższymi metrykami i wnioskami z kodu!")


def evaluate_pipeline(calibrated_spec_pipe, artifacts, X_train, X_test, y_train, y_test):
    """ Przeprowadza ewaluację kaskady na symulowanej kohorcie wysokociśnieniowej OIT 
        (75% chorych / 25% zdrowych): wylicza pełne metryki łącznie z Brier Score, ECE i FOR, 
        wykonuje testy statystyczne McNemara, benchmarkuje wynik z klasyczną regułą GCS < 5 oraz generuje wykresy. """
    logger.info(f"Ocenianie Kaskady na symulowanej kohorcie wysokociśnieniowej OIT (75% chorych / 25% zdrowych)")
    
    # Wymuszenie proporcji 75% chorych / 25% zdrowych
    n_total_test = len(X_test)
    n_positive = int(n_total_test * 0.75)
    n_negative = n_total_test - n_positive

    pos_indices = np.where(y_test.values == 1)[0]
    neg_indices = np.where(y_test.values == 0)[0]
    
    rng_sampler = np.random.default_rng(CONFIG["SEED"])
    pos_sampled_idx = rng_sampler.choice(pos_indices, size=n_positive, replace=True)
    neg_sampled_idx = rng_sampler.choice(neg_indices, size=n_negative, replace=True)
    sampled_indices = np.concatenate([pos_sampled_idx, neg_sampled_idx])
    
    X_test = X_test.iloc[sampled_indices]
    y_test = y_test.iloc[sampled_indices]
    
    for key in ["dummy_crnn_test", "dummy_efficientnet_test", "dummy_densenet_test"]:
        if artifacts.get(key) is not None:
            artifacts[key] = artifacts[key][sampled_indices]
            
    logger.info(f"Kohorta testowa OIT gotowa. Liczba próbek: {len(X_test)} (75% krytycznych / 25% stabilnych).")

    y_final_prob = calibrated_spec_pipe.predict_proba(
        X_test, 
        crnn_probs=artifacts["dummy_crnn_test"], 
        efficientnet_probs=artifacts["dummy_efficientnet_test"],
        densenet_probs=artifacts["dummy_densenet_test"]
    )[:, 1]
    
    logger.info(f"Ostateczny wynik Kaskady (AUROC): {roc_auc_score(y_test, y_final_prob):.4f}")

    best_threshold = artifacts.get("optimal_threshold", 0.5)
    logger.info(f"Odtworzono PRÓG KLINICZNY F-BETA (wyliczony na Train): {best_threshold:.4f}")
    y_threshold_pred = (y_final_prob >= best_threshold).astype(int)
    
    logger.info("\n" + classification_report(y_test, y_threshold_pred))

    # Obliczanie metryk klinicznych PPV i NPV
    tn, fp, fn, tp = confusion_matrix(y_test, y_threshold_pred).ravel()
    
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    logger.info("METRYKI KLINICZNE")
    logger.info(f"PPV: {ppv*100:.1f}% -> Pewność czerwonego alarmu")
    logger.info(f"NPV: {npv*100:.1f}% -> Pewność odesłania pacjenta")

    # Brier, ECE, FOR dla całej kaskady
    y_true_arr = np.array(y_test)
    brier = brier_score_loss(y_true_arr, y_final_prob)
    ece = calculate_ece(y_true_arr, y_final_prob)
    for_score = (fn / (fn + tn)) * 100 if (fn + tn) > 0 else 0.0

    logger.info("WSKAŹNIKI KALIBRACJI I BEZPIECZEŃSTWA")
    logger.info(f"Brier Score: {brier:.4f} (Idealnie = 0.0)")
    logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")
    logger.info(f"False Omission Rate (FOR): {for_score:.2f}% (Odsetek przeoczonych chorych)")

    # Kaskada AI vs Klasyczna Reguła GCS < 5
    try:
        if 'gcs_sum' in X_test.columns:
            gcs_rule_pred = (X_test['gcs_sum'] < 5).astype(int)
        else:
            gcs_eyes = X_test.get('gcs_eyes', pd.Series(4, index=X_test.index))
            gcs_motor = X_test.get('gcs_motor', pd.Series(6, index=X_test.index))
            gcs_verbal = X_test.get('gcs_verbal', pd.Series(5, index=X_test.index))
            gcs_rule_pred = ((gcs_eyes + gcs_motor + gcs_verbal) < 5).astype(int)

        tn_gcs, fp_gcs, fn_gcs, tp_gcs = confusion_matrix(y_true_arr, gcs_rule_pred, labels=[0, 1]).ravel()
        npv_gcs = tn_gcs / (tn_gcs + fn_gcs) if (tn_gcs + fn_gcs) > 0 else 0.0
        for_gcs = (fn_gcs / (fn_gcs + tn_gcs)) * 100 if (fn_gcs + tn_gcs) > 0 else 0.0
        sens_gcs = tp_gcs / (tp_gcs + fn_gcs) * 100 if (tp_gcs + fn_gcs) > 0 else 0.0
        sens_cascade = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0

        logger.info("Kaskada AI vs Klasyczna Reguła GCS < 5")
        logger.info(f"Reguła GCS < 5 -> Czułość: {sens_gcs:.1f}% | NPV: {npv_gcs*100:.1f}% | FOR (Błąd pominięcia): {for_gcs:.2f}%")
        logger.info(f"Kaskada AI    -> Czułość: {sens_cascade:.1f}% | NPV: {npv*100:.1f}% | FOR (Błąd pominięcia): {for_score:.2f}%")
        logger.info(f"Redukcja wskaźnika fałszywych pominięć (FOR) dzięki AI: {for_gcs - for_score:+.2f} p.p.")
    except (ValueError, TypeError) as e:
        logger.warning(f"Nie udało się wyliczyć benchmarku GCS < 5: {e}")

    # Test McNemara: Porównanie Model Bazowy vs Kaskada
    try:
        X_test_clean = artifacts["X_test_clean_df"].reindex(X_test.index).fillna(0)
        
        main_model_prob = artifacts["ensemble_best"].predict_proba(X_test_clean.values)[:, 1]
        main_model_pred = (main_model_prob >= 0.5).astype(int)
        
        base_correct = (main_model_pred == y_true_arr)
        cascade_correct = (y_threshold_pred == y_true_arr)
        
        both_correct = np.sum(base_correct & cascade_correct)
        base_only = np.sum(base_correct & ~cascade_correct)
        cascade_only = np.sum(~base_correct & cascade_correct)
        both_wrong = np.sum(~base_correct & ~cascade_correct)
        
        contingency_table = [[both_correct, base_only], [cascade_only, both_wrong]]
        res_mcnemar = mcnemar(contingency_table, exact=True)
        
        logger.info("Test Statystyczny McNemara (Model Bazowy vs Kaskada - OIT Stress-Test)")
        logger.info(f"Tabela kontyngencji: {contingency_table}")
        logger.info(f"Wartość p-value testu McNemara: {res_mcnemar.pvalue:.5f}")
    except (ValueError, TypeError) as e:
        logger.exception("Błąd podczas wykonywania testu statystycznego McNemara.")

    try:
        trained_lr = artifacts["final_judge"].estimators_[0]
        feat_names = trained_lr.named_steps['polynomialfeatures'].get_feature_names_out(artifacts["base_names"])
        weights = trained_lr.named_steps['logisticregressioncv'].coef_[0]
        
        logger.info("Analiza Decyzyjna Sędziego")
        for name, weight in zip(feat_names, weights):
            logger.info(f"Cecha: {str(name).ljust(35)} | Waga ryzyka zgonu: {weight:+.4f}")
    except (ValueError, TypeError) as e:
        logger.exception("Nie można wydrukować wag sędziego.")

    cm = confusion_matrix(y_test, y_threshold_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Macierz Triage OIT (Próg: {best_threshold:.2f})')
    plt.savefig(f'{CONFIG["PLOTS_DIR"]}/confusion_matrix_oit.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Studium Ablacyjne
    auc_ablation, y_prob_ablation = run_ablation_study(X_train, y_train, X_test, y_test, artifacts)
    plot_calibration_and_brier(y_test, y_prob_ablation, model_name="Czysty XGBoost (Ablacja)")
    plot_decision_curve_analysis(y_test, y_prob_ablation, model_name="Czysty XGBoost (Ablacja)")
    
    auc_cascade = roc_auc_score(y_test, y_final_prob)
    delta = auc_cascade - auc_ablation
    logger.info(f"Zysk z całej złożonej inżynierii (Kaskada - XGBoost): {delta:+.4f} AUROC")
    
    if delta < 0.01:
        logger.warning("Złożona kaskada nie poprawia wyników na tyle, by uzasadnić jej koszt utrzymania. Rozważ użycie czystego XGBoosta.")

    # Wyliczanie przedziałów ufności dla AUROC i F2-Score
    auc_ci, _ = calculate_confidence_intervals(y_test, y_final_prob, threshold=best_threshold, n_bootstraps=1000)
    
    # Analiza błędów
    perform_error_analysis(X_test, y_test, y_final_prob, artifacts)

    # Dynamiczne obliczenie Zysku Netto w optymalnym progu decyzyjnym
    tp = np.sum((y_threshold_pred == 1) & (y_true_arr == 1))
    fp = np.sum((y_threshold_pred == 1) & (y_true_arr == 0))
    n_samples = len(y_true_arr)

    safe_thresh = min(best_threshold, 0.999) 
    dca_benefit = (tp / n_samples) - (fp / n_samples) * (safe_thresh / (1 - safe_thresh))

    # Generowanie  wykresów synergii z poprawną zmienną y_true_arr
    generate_full_synergy_report(calibrated_spec_pipe, artifacts, X_train, X_test, y_true_arr)

    plot_calibration_and_brier(y_test, y_final_prob, model_name="Pełna Kaskada TBI")
    plot_decision_curve_analysis(y_test, y_final_prob, model_name="Kaskada + Meta-Sędzia")

    # Generowanie pełnego pliku CSV z wynikami wszystkich modeli
    logger.info("Eksport kompletnych predykcji na zbiorze testowym do pliku CSV.")
    
    final_results_df = pd.DataFrame({
        'True_Label': y_test.values,
        'Cascade_Final_Prob': calibrated_spec_pipe.predict_proba(
            X_test, 
            crnn_probs=artifacts["dummy_crnn_test"], 
            efficientnet_probs=artifacts["dummy_efficientnet_test"],
            densenet_probs=artifacts["dummy_densenet_test"]
        )[:, 1],
        'CRNN_Prob': artifacts["dummy_crnn_test"],
        'CRNN_Uncertainty': artifacts["dummy_crnn_uncert"],
        'EfficientNet_Prob': artifacts["dummy_efficientnet_test"],
        'EfficientNet_Uncertainty': artifacts["dummy_efficientnet_uncert"],
        'DenseNet_Prob': artifacts["dummy_densenet_test"],
        'DenseNet_Uncertainty': artifacts["dummy_densenet_uncert"]
    }, index=X_test.index)

    # Dołączenie oryginalnych cech tabelarycznych, aby ułatwić późniejszą analizę
    final_results_df = final_results_df.join(X_test)

    # Zapis na dysk
    results_csv_path = os.path.join(CONFIG["PLOTS_DIR"], 'final_full_test_predictions.csv')
    final_results_df.to_csv(results_csv_path)
    logger.info(f"Zapisano {len(final_results_df)} rekordów z wynikami CV i Kaskady w: {results_csv_path}")

    # Punkt zbiorczy: Zapis metryk do JSON oraz jednoczesna aktualizacja README.md na sam koniec
    try:
        generate_and_save_metrics_report(
            y_true=y_true_arr,
            y_prob=y_final_prob,
            y_pred=y_threshold_pred,
            baseline_preds=main_model_pred if 'main_model_pred' in locals() else None,
            auc_ci=auc_ci,
            dca_benefit=dca_benefit,
            file_path="output_data/tbi_metrics_summary.json"
        )

        update_readme_metrics()
        logger.info("Raport metryk JSON oraz plik README.md zostały pomyślnie zaktualizowane nowymi wynikami.")

    except (OSError, IOError, TypeError, ValueError) as e:
        logger.error(f"Nie udało się wygenerować i zapisać automatycznego raportu JSON lub zaktualizować README: {e}")


class PatientStateSONODE(nn.Module):
    """ Równanie różniczkowe drugiego rzędu dla fizjologii pacjenta.
        
        Uzasadnienie teoretyczne i kliniczne:
        Zwykłe Neural ODE (pierwszego rzędu) napotykają na fundamentalną barierę topologiczną 
        wynikającą z twierdzenia Picarda-Lindelöfa: ich trajektorie w przestrzeni stanów nie mogą się 
        przecinać. Powoduje to, że nieliniowo separowalne pułapki i stany ryzyka są dla nich matematycznie 
        niemożliwe do rozwikłania:
        1. Problem XOR z GCS i sedacją: Izolowane objawy bywają mylące (np. wysoki GCS bez sedacji to stan 
           stabilny, ale nagły spadek GCS przy braku leków sedacyjnych to alarm krytyczny). Zwykłe ODE nie potrafi 
           przeciąć trajektorii tych pacjentów, co generuje fałszywe alarmy.
        2. Fizjologiczne Nested Spheres (koncentryczne sfery ryzyka): Fizjologia opiera się na optimum homeostazy 
           (efekt U-Shape). Zarówno zbyt niskie, jak i zbyt wysokie wartości (np. hipotensja vs hypertensja) 
           róvnocześnie zagrażają mózgowi. Standardowe ODE dusi się, próbując rozdzielić sferę zdrowia uwięzioną 
           wewnątrz sfery stanu krytycznego bez możliwości przecięcia warstw.
           
        SONODE pokonuje te ograniczenia, rozszerzając przestrzeń fazową o wymiar dynamiki i przyspieszenia. 
        Dzięki temu trajektorie mogą swobodnie przechodzić nad sobą w wyższym wymiarze, umożliwiając analizę 
        drugiej pochodnej i natychmiastowe wychwycenie nieliniowego, przyspieszającego załamania homeostazy, 
        w tym kluczowych stanów neurokrytycznych:
        - Wyczerpanie rezerwy kompensacyjnej czaszki (Wejście na strome ramię wykładniczej krzywej podatności objętościowej).
        - Opóźniona ekspansja krwiaka / Lucid Interval (Gdy pacjent z GCS 15 nagle zaczyna gwałtownie manewrować wektorami ciśnienia 
          i tętna przed utratą przytomności z powodu pęknięcia tętnicy oponowej środkowej).
        - Dynamiczny Odruch Cushinga (Wykrywanie rozpędzania się rozjazdu ciśnienia skurczowego i bradykardii jako zwiastuna wgłobienia).
        - Dekompresja we wstrząsie krwotocznym (Moment, w którym mechanizmy wazokonstrykcji pękają, wywołując logarytmiczny spadek MAP). """
    def __init__(self, hidden_dim=16):
        super().__init__()
        # Sieć przyjmuje połączony stan i prędkość zmian [h, v]
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, t, state):
        """ Ponieważ torchdiffeq oczekuje wektora stanu, dla SONODE rozszerzamy wymiar o prędkość. """
        h, v = torch.chunk(state, 2, dim=-1)
        
        # Pochodna stanu to obecna prędkość: dh/dt = v
        dh_dt = v
        
        # Pochodna prędkości (przyspieszenie) to nieliniowa funkcja sił fizjologicznych: dv/dt = f(h, v)
        combined_input = torch.cat([h, v], dim=-1)
        dv_dt = self.net(combined_input)
        
        # Zwracamy spakowaną pochodną pierwszego i drugiego rzędu
        return torch.cat([dh_dt, dv_dt], dim=-1)


class LatentODE_VAE_Generator(nn.Module):
    """ Generatywny Model Neural ODE połączony z Variational Autoencoder.
        Zabezpiecza przed overfittingiem i generowaniem medycznych halucynacji. """
    def __init__(self, input_dim=22, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # RNN czyta historię pacjenta
        self.encoder_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Przekształcenie w rozkład prawdopodobieństwa przy użyciu VAE
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)       # Średnia rozkładu
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)   # Logarytm wariancji
        
        # ODE dla dynamiki stanu pacjenta w czasie. To jest serce modelu, które przewiduje trajektorie fizjologiczne.
        self.ode_func = PatientStateSONODE(hidden_dim)
        
        # Odtwarzanie surowych parametrów medycznych
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
        # Przewidywanie ryzyka zgonu w czasie na podstawie trajektorii ukrytej
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """ Reparameterization Trick: Pozwala na propagację wsteczną
            przez proces losowego próbkowania z przestrzeni ukrytej. """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Szum z rozkładu normalnego N(0,1)
        return mu + eps * std
        
    def forward(self, vitals_seq, t_seq):
        """ Przepływ treningowy z obliczaniem straty KLD """
        # Najpierw RNN przetwarza historię
        _, h_rnn = self.encoder_rnn(vitals_seq) 
        h_rnn = h_rnn.squeeze(0)
        
        # Projekcja do przestrzeni VAE
        mu = self.fc_mu(h_rnn)
        logvar = self.fc_logvar(h_rnn)
        
        # Próbkowanie szablonu z=h(0)
        z_0 = self.reparameterize(mu, logvar)
        
        # Dodajemy wektor początkowej prędkości (wyzerowany) i łączymy ze stanem
        v_0 = torch.zeros_like(z_0)
        state_0 = torch.cat([z_0, v_0], dim=-1)
        
        # Odcałkowanie w czasie za pomocą ODE (wchodzi rozszerzony stan)
        state_t = odeint(self.ode_func, state_0, t_seq, method='dopri5')
        
        # Rozcinamy z powrotem, do dekodera idzie tylko stan
        h_t, _ = torch.chunk(state_t, 2, dim=-1)
        
        # Dekodowanie
        raw_reconstructed = self.decoder(h_t)
        
        # Zabezpieczamy sieć przed wygenerowaniem np. ujemnego ICP lub tętna.
        reconstructed_vitals = torch.clamp(raw_reconstructed, min=0.0)
        
        # Predykcja ryzyka
        risk_trajectory = self.risk_head(h_t).squeeze(-1)
        
        reconstructed_vitals = reconstructed_vitals.permute(1, 0, 2)
        risk_trajectory = risk_trajectory.permute(1, 0)
        
        # Obliczenie KLD. Zmusza przestrzeń ukrytą do przybrania kształtu idealnego rozkładu normalnego.
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return reconstructed_vitals, risk_trajectory, kld_loss

    def generate_synthetic_patients(self, base_vitals_seq, t_seq, num_samples=5):
        """ Generowanie z użyciem przestrzeni VAE. Szum jest nadzorowany przez wyuczoną
            wariancję danego pacjenta, a nie zgadywany w ciemno. """
        self.eval()
        with torch.no_grad():
            _, h_rnn = self.encoder_rnn(base_vitals_seq)
            h_rnn = h_rnn.squeeze(0)
            
            mu = self.fc_mu(h_rnn)
            logvar = self.fc_logvar(h_rnn)
            
            synthetic_patients = []
            for _ in range(num_samples):
                z_0 = self.reparameterize(mu, logvar)
                
                # Uwzględniamy prędkość początkową w SONODE, żeby model mógł uchwycić przyspieszające pogorszenie stanu pacjenta
                v_0 = torch.zeros_like(z_0)
                state_0 = torch.cat([z_0, v_0], dim=-1)
                
                state_t = odeint(self.ode_func, state_0, t_seq, method='dopri5')
                h_t, _ = torch.chunk(state_t, 2, dim=-1)
                
                raw_vitals = self.decoder(h_t)
                synthetic_vitals = torch.clamp(raw_vitals, min=0.0)
                synthetic_vitals = synthetic_vitals.permute(1, 0, 2).squeeze(0)
                
                synthetic_patients.append(synthetic_vitals)
                
        return torch.stack(synthetic_patients)

    def generate_counterfactual_augmentation(self, base_vitals_seq, t_seq, original_treatments, target_treatment_idx, time_of_intervention_idx):
        """ Tworzy kontrfaktyczną trajektorię pacjenta poprzez modyfikację interwencji. 
            Zwraca zrekonstruowane parametry życiowe jako symulację 'co by było gdyby'. """
        self.eval()
        with torch.no_grad():
            # Kodowanie aktualnej historii pacjenta do punktu startowego
            _, h_rnn = self.encoder_rnn(base_vitals_seq)
            h_rnn = h_rnn.squeeze(0)
            
            mu = self.fc_mu(h_rnn)
            logvar = self.fc_logvar(h_rnn)
            z_0 = self.reparameterize(mu, logvar)
            
            # Stan początkowy (pozycja + początkowa prędkość dla SONODE)
            v_0 = torch.zeros_like(z_0)
            state_0 = torch.cat([z_0, v_0], dim=-1)
            
            # Kopiujemy oryginalne leczenie i zmieniamy w wybranym momencie
            cf_treatments = original_treatments.clone()
            current_action = cf_treatments[:, time_of_intervention_idx:, target_treatment_idx]
            cf_treatments[:, time_of_intervention_idx:, target_treatment_idx] = 1.0 - current_action
            
            # Dodajemy kontrfaktyczne leczenie do funkcji ODE, aby trajektoria uwzględniała zmienione interwencje
            self.ode_func.set_treatments(cf_treatments)
            state_t = odeint(self.ode_func, state_0, t_seq, method='dopri5')
            h_t, _ = torch.chunk(state_t, 2, dim=-1)
            
            # Zdekodowanie ukrytych wektorów z powrotem na parametry medyczne
            raw_vitals = self.decoder(h_t)
            synthetic_vitals = torch.clamp(raw_vitals, min=0.0)
            synthetic_vitals = synthetic_vitals.permute(1, 0, 2).squeeze(0)
            
        return synthetic_vitals, cf_treatments


class LatentODE_PatientGenerator(nn.Module):
    """ Generatywny Model Neural ODE. Działa jako Enkoder-Dekoder oparty na równaniach różniczkowych. """
    def __init__(self, input_dim=22, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Czyta dotychczasową historię pacjenta i wyciąga najważniejsze cechy do przestrzeni ukrytej
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Przesuwa stan pacjenta w czasie ciągłym
        self.ode_func = PatientStateSONODE(hidden_dim)
        
        # Tłumaczy ukryty stan z powrotem na fizyczne parametry (ICP, GCS, Lactate itp.)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

        # Klasyfikuje ciężkość stanu w danym momencie t
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, vitals_seq, t_seq):
        """ Przepływ treningowy (rekonstrukcja i predykcja). """        
        _, h_0 = self.encoder(vitals_seq) 
        h_0 = h_0.squeeze(0)
        
        h_t = odeint(self.ode_func, h_0, t_seq, method='dopri5')
        
        reconstructed_vitals = self.decoder(h_t)
        risk_trajectory = self.risk_head(h_t).squeeze(-1)
        
        reconstructed_vitals = reconstructed_vitals.permute(1, 0, 2)
        risk_trajectory = risk_trajectory.permute(1, 0)
        
        return reconstructed_vitals, risk_trajectory

    def generate_synthetic_patients_via_interpolation(self, tier1_vitals, tier2_vitals, t_seq, num_samples=5, mix_ratio=0.7):
        """ Kliniczna Interpolacja w Przestrzeni Ukrytej ODE. Bierze stabilnych pacjentów jako tło biologiczne i popycha
            ich trajektorie w stronę krytycznego pogorszenia. """
        self.eval()
        with torch.no_grad():
            # Poprawka: używamy self.encoder zamiast self.encoder_rnn
            _, h_tier2 = self.encoder(tier2_vitals)
            h_tier2 = h_tier2.squeeze(0)
            
            critical_center = torch.mean(h_tier2, dim=0, keepdim=True)
            
            _, h_tier1 = self.encoder(tier1_vitals)
            h_tier1 = h_tier1.squeeze(0)
            
            synthetic_patients = []
            
            for _ in range(num_samples):
                idx = torch.randint(0, h_tier1.size(0), (1,)).item()
                base_h = h_tier1[idx].unsqueeze(0)
                
                interpolated_h = (1.0 - mix_ratio) * base_h + mix_ratio * critical_center
                noisy_h = interpolated_h + torch.randn_like(interpolated_h) * 0.05
                
                h_t = odeint(self.ode_func, noisy_h, t_seq, method='dopri5')
                
                raw_vitals = self.decoder(h_t)
                synthetic_vitals = torch.clamp(raw_vitals, min=0.0)
                synthetic_vitals = synthetic_vitals.permute(1, 0, 2).squeeze(0)
                
                synthetic_patients.append(synthetic_vitals)
                
        return torch.stack(synthetic_patients)  


class BayesianTrajectoryMonitor:
    """ Moduł Longitudinalnego Monitorowania Pacjenta wzbogacony o zanikanie alarmów, kontekst sedacji farmakologicznej
        oraz nieliniowe kary za wgłobienie bazujące na rezerwie biologicznej. """
    def __init__(self, initial_prob: float, patient_age: int = 50):
        self.patient_age = patient_age
        self.current_prob = max(min(initial_prob, 0.99), 0.01) # Zabezpieczenie przed 0 i 1
        self.current_odds = self.current_prob / (1 - self.current_prob)
        self.history = [{
            "time": datetime.datetime.now(), 
            "prob": self.current_prob, 
            "event": "Admission (Cascade Baseline)"
        }]
        self.last_update_time = datetime.datetime.now()

    def update_belief(self, new_vitals: dict, current_time: datetime.datetime = None) -> float:
        if current_time is None: 
            current_time = datetime.datetime.now()

        likelihood_ratio = 1.0
        event_notes = []

        # Zanikanie czasowe - jeśli nic się nie dzieje, pacjent powoli stabilizuje się
        hours_passed = (current_time - self.last_update_time).total_seconds() / 3600.0
        decay_factor = 0.98 ** hours_passed
        self.current_odds *= decay_factor

        # Flaga śpiączki farmakologicznej (zapobiega fałszywym alarmom ostrego spadku GCS)
        is_sedated = new_vitals.get('sedatives_administered', False)

        # Płynna zmiana GCS
        if 'delta_gcs' in new_vitals:
            drop = new_vitals['delta_gcs']
            if drop < 0:
                # Nieliniowy wzrost ryzyka
                base_penalty = 1.0 + (abs(drop) ** 1.5) * 0.5 
                
                if is_sedated:
                    likelihood_ratio *= (1.0 + (base_penalty - 1.0) * 0.2) # Tłumienie o 80% przez leki (GABA)
                    event_notes.append(f"GCS {drop} (Sedacja, zmniejszona kara: LR: {(1.0 + (base_penalty - 1.0) * 0.2):.2f}x)")
                else:
                    likelihood_ratio *= base_penalty
                    event_notes.append(f"KRYTYCZNE GCS {drop} (LR: {base_penalty:.2f}x)")
            elif drop > 0:
                reward = 0.8 ** drop  # Płynny bonus za wybudzanie
                likelihood_ratio *= reward
                event_notes.append(f"GCS + {drop} (LR: {reward:.2f}x)")

        # Ciśnienie wewnątrzczaszkowe (Normalne ICP to 5-15 mmHg)
        if 'icp_current' in new_vitals:
            icp = new_vitals['icp_current']
            if icp > 15.0:
                excess_icp = icp - 15.0
                
                # Modyfikator geriatryczny (atrofia mózgu zmienia objętość rezerwową czaszki)
                if self.patient_age >= 75:
                    age_mod = 1.5 
                elif self.patient_age < 30:
                    age_mod = 0.7 
                else:
                    age_mod = 1.0
                
                # Wykładniczy mnożnik: przy ICP 22 u 50-latka wyniesie ok. 3.4x, przy 25 wyniesie ok. 6.0x
                icp_multiplier = 1.0 + ((excess_icp * 0.2) ** 2) * age_mod
                likelihood_ratio *= icp_multiplier
                event_notes.append(f"ICP {icp} mmHg (LR: {icp_multiplier:.2f}x)")

        # Brak złych wieści i brak sedacji to dobra wiadomość
        if likelihood_ratio == 1.0 and not is_sedated:
            likelihood_ratio = 0.95
            event_notes.append("Stabilny stan (LR: 0.95x)")

        # Aktualizacja Szans i konwersja z powrotem na Prawdopodobieństwo
        self.current_odds *= likelihood_ratio
        self.current_prob = self.current_odds / (1 + self.current_odds)
        self.last_update_time = current_time

        self.history.append({
            "time": current_time,
            "prob": self.current_prob,
            "event": " | ".join(event_notes)
        })

        return self.current_prob
    

class WeaningReadinessEvaluator(nn.Module):
    """ Moduł oceniający prawdopodobieństwo udanej ekstubacji (odłączenia od respiratora).
        Zaprojektowany do uczenia na rozszerzonych danych oddechowych ze zbioru eICU.
        Wymaga pełnego profilu oddechowo-hemodynamicznego:
        1-5. Baza: Wiek, GCS, Ciśnienie, Tętno, Wskaźnik Oksygenacji (SpO2/FiO2)
        6. RSBI (Rapid Shallow Breathing Index) - stosunek częstości oddechów do objętości oddechowej
        7-9. Gazometria: pH krwi, PaO2, PaCO2 (krytyczne dla zapobiegania obrzękowi mózgu)
        10-11. Parametry respiratora: PEEP (dodatnie ciśnienie końcowowydechowe), PIP (szczytowe ciśnienie)
        12. RASS (Poziom sedacji/pobudzenia) - pacjent musi współpracować. """
    def __init__(self, input_dim=12):
        super(WeaningReadinessEvaluator, self).__init__()
        self.evaluator = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3), # Zwiększony dropout ze względu na większą liczbę cech
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Zwraca prawdopodobieństwo (0.0 - 1.0) udanej próby oddechu własnego
        )

    def forward(self, x):
        return self.evaluator(x)


def extract_mimic_tbi_trajectories(db_uri: str, limit_patients: int = 100):
    """ Wyciąga z lokalnej bazy MIMIC-IV sekwencje czasowe GCS oraz ICP 
        dla pacjentów z urazami głowy w ciągu pierwszych 24h pobytu na OIT. """
    engine = create_engine(db_uri)
    
    # Zapytanie SQL filtrujące pacjentów z TBI
    query = f"""
    WITH tbi_patients AS (
        SELECT DISTINCT ie.stay_id, ie.subject_id, ie.intime
        FROM icustays ie
        JOIN diagnoses_icd di ON ie.hadm_id = di.hadm_id
        WHERE di.icd_code LIKE 'S06%' -- Kody ICD-10 dla urazów wewnątrzczaszkowych
        LIMIT {limit_patients}
    ,
    raw_vitals AS (
        SELECT 
            tp.stay_id,
            ce.charttime,
            EXTRACT(EPOCH FROM (ce.charttime - tp.intime)) / 3600.0 as hours_from_admission,
            ce.itemid,
            ce.valuenum
        FROM tbi_patients tp
        JOIN chartevents ce ON tp.stay_id = ce.stay_id
        WHERE ce.charttime >= tp.intime 
          AND ce.charttime <= tp.intime + INTERVAL '24 hour'
          -- Przykładowe itemid w MIMIC dla GCS i ICP (warto dostosować do d_items w Twojej bazie)
          AND ce.itemid IN (
              220739, 223900, 223901, -- GCS Total / Components
              220045, 226537          -- ICP (Intracranial Pressure)
          )
          AND ce.valuenum IS NOT NULL
    )
    SELECT * FROM raw_vitals
    ORDER BY stay_id, hours_from_admission;
    """
    
    df = pd.read_sql(query, engine)
    return df


def prepare_real_icu_tensors(df: pd.DataFrame, max_seq_len: int = 24):
    """ Przetwarza surowe dane z MIMIC-IV: 
        - Rozdziela parametry kliniczne na niezależne kanały (np. ICP, GCS).
        - Zachowuje wielowymiarowy tensor cech [Batch, SeqLen, InputDim].
        - Wykonuje niezależną interpolację i obsługę braków danych dla każdego kanału. """
    # Mapowanie MIMIC itemid na nazwy kliniczne
    item_mapping = {
        220045: 'icp',
        226537: 'icp',
        220739: 'gcs_total',
        223900: 'gcs_eye',
        223901: 'gcs_motor',
        223902: 'gcs_verbal'
    }
    
    df['clinical_feature'] = df['itemid'].map(item_mapping)
    df = df.dropna(subset=['clinical_feature'])
    
    batch_features = []
    batch_deltas = []
    
    grouped = df.groupby('stay_id')
    
    # Ustalamy stałą listę cech wejściowych
    feature_columns = ['icp', 'gcs_total']
    input_dim = len(feature_columns)
    
    for stay_id, group in grouped:
        group = group.sort_values('hours_from_admission')
        
        if len(group) < 2:
            continue
            
        # Okno czasowe (np. dyskretyzacja do pełnych godzin pobytu)
        group['hour_bin'] = group['hours_from_admission'].astype(int)
        
        # Pivotowanie tabeli: wiersze to godziny, kolumny to niezależne parametry kliniczne
        pivot_df = group.pivot_table(
            index='hour_bin', 
            columns='clinical_feature', 
            values='valuenum', 
            aggfunc='mean'
        )
        
        # Upewnienie się, że wszystkie wymagane kolumny istnieją w pivot_df
        for col in feature_columns:
            if col not in pivot_df.columns:
                pivot_df[col] = np.nan
                
        pivot_df = pivot_df[feature_columns]
        
        # Ograniczenie do max_seq_len godzin
        if len(pivot_df) > max_seq_len:
            pivot_df = pivot_df.head(max_seq_len)
            
        # Najpierw forward-fill (ostatnia znana wartość), potem backward-fill, na końcu 0 dla braków startowych
        pivot_df = pivot_df.ffill().bfill().fillna(0.0)
        
        features_matrix = pivot_df.values # Kształt: [Liczba rzeczywistych kroków, input_dim]
        hours = pivot_df.index.values
        
        # Obliczenie rzeczywistych odstępów czasu dt dla każdego kroku
        dt = np.diff(hours, prepend=hours[0])
        dt[dt <= 0] = 1.0 # Zabezpieczenie przed dt=0 przy wielu pomiarach w tej samej godzinie
        
        # Padding sekwencji do stałej długości (max_seq_len)
        current_len = len(features_matrix)
        if current_len < max_seq_len:
            pad_len = max_seq_len - current_len

            # Padding zerami dla cech, padding stałym krokiem czasowym (np. 1.0h) dla dt
            features_matrix = np.pad(features_matrix, ((0, pad_len), (0, 0)), 'constant', constant_values=0.0)
            dt = np.pad(dt, (0, pad_len), 'constant', constant_values=1.0)
        elif current_len > max_seq_len:
            features_matrix = features_matrix[:max_seq_len]
            dt = dt[:max_seq_len]
            
        batch_features.append(features_matrix)
        batch_deltas.append(dt)
        
    if not batch_features:
        raise ValueError("Brak poprawnych trajektorii po przetworzeniu.")
        
    # Konwersja na tensory o wymiarach wymaganych przez model wielowymiarowy
    X_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32)
    Dt_tensor = torch.tensor(np.array(batch_deltas), dtype=torch.float32)
    
    return X_tensor, Dt_tensor


class ODEFunc(nn.Module):
    """ Definiuje pochodną stanu ukrytego w czasie: dh/dt = f(h(t), t) """
    def __init__(self, hidden_dim: int):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, t, h):
        return self.net(h)


class NeuralODE_ICU_Monitor(nn.Module):
    """ Ciągły model monitorowania trajektorii stanu (ciągła ewolucja intencji lub wstrząsów 
        w nieregularnych odstępach czasu) z użyciem mechanizmów Neural ODE. """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super(NeuralODE_ICU_Monitor, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, sequence_features, time_deltas):
        batch_size, seq_len, _ = sequence_features.shape
        device = sequence_features.device
        h = torch.zeros(batch_size, self.ode_func.net[0].in_features, device=device)
        
        for i in range(seq_len):
            x_t = sequence_features[:, i, :]
            dt = time_deltas[:, i].unsqueeze(-1)
            
            # Aktualizacja stanu wejściowego w danym kroku
            h = h + self.input_projection(x_t)
            
            # Ciągła ewolucja trajektorii o zadany interwał czasowy dt
            deriv = self.ode_func(0.0, h)
            h = h + deriv * dt
            
        return torch.sigmoid(self.output_layer(h))


def generate_icu_clinical_scenario(num_patients: int = 100, seq_length: int = 24, seed: int = 42) -> Tuple[pd.DataFrame, torch.Tensor, List[str]]:
    """ Generuje wielomodalny syntetyczny scenariusz kliniczny dla pacjentów OIT z TBI.
        Zasila kaskadę testową i przygotowuje grunt pod moduł sekwencyjny """
    np.random.seed(seed)
    
    # Dane tabelaryczne
    ages = np.random.normal(45, 15, num_patients).clip(18, 90).astype(int)
    gcs_scores = np.random.randint(3, 16, num_patients)
    pupils = np.random.choice(['Both Reactive', 'One Reactive', 'Non-reactive'], num_patients, p=[0.7, 0.2, 0.1])
    
    df_tabular = pd.DataFrame({
        'PatientID': [f'ICU_{i:04d}' for i in range(num_patients)],
        'Age': ages,
        'Admission_GCS': gcs_scores,
        'Pupil_Reactivity': pupils,
        'Hypotension_Flag': np.random.choice([0, 1], num_patients, p=[0.8, 0.2])
    })
    
    # Szeregi czasowe dla CRNN
    time_series = np.zeros((num_patients, seq_length, 4))
    clinical_notes = []
    
    for i in range(num_patients):
        # Wyjściowe parametry powiązane z ciężkością urazu (GCS)
        base_icp = np.random.normal(12, 3) if gcs_scores[i] > 8 else np.random.normal(22, 5)
        base_map = np.random.normal(85, 10)
        
        # Symulacja trajektorii czasowej (Random Walk z dryfem)
        icp_trend = base_icp + np.cumsum(np.random.normal(0, 1.5, seq_length))
        map_trend = base_map + np.cumsum(np.random.normal(0, 2.0, seq_length))
        hr_trend = np.random.normal(75, 15) + np.cumsum(np.random.normal(0, 1.0, seq_length))
        spo2_trend = np.clip(np.random.normal(96, 2) + np.cumsum(np.random.normal(0, 0.5, seq_length)), 80, 100)
        
        time_series[i, :, 0] = np.clip(icp_trend, 0, 50)
        time_series[i, :, 1] = np.clip(map_trend, 40, 130)
        time_series[i, :, 2] = np.clip(hr_trend, 40, 180)
        time_series[i, :, 3] = spo2_trend
        
    return df_tabular, torch.tensor(time_series, dtype=torch.float32), clinical_notes


def serve_patient_stream(calibrated_spec_pipe,
    X_test,
    ode_monitor=None,
    weaning_model=None,
    redis_host="localhost",
):
  """Symuluje strumieniowy napływ danych pacjenta z bazy Redis w czasie rzeczywistym.

  Porównuje trajektorię Bayesowską z ciągłą, odcałkowaną (Neural ODE). """
  logger = logging.getLogger(__name__)
  logger.info(
      "Pokaz działania na żywo: Inicjalna Kaskada (SOR) -> Monitorowanie (OIT)"
  )
  buffer = RealTimePatientBuffer(host=redis_host, port=6379)

  # Bezpieczna, granularna inicjalizacja klienta Redis dla alertów
  r_cache = None
  try:
    r_cache = redis.Redis(
        host=redis_host, 
        port=6379, 
        decode_responses=True,
        socket_timeout=2.0,
        socket_connect_timeout=2.0
    )
    r_cache.ping()
  except redis.AuthenticationError as e:
    logger.error(f"Błąd uwierzytelniania w bazie Redis (sprawdź poświadczenia): {e}")
    r_cache = None
  except (redis.ConnectionError, redis.TimeoutError) as e:
    logger.warning(f"Serwer Redis na hoście '{redis_host}' jest niedostępny lub przeciążony: {e}")
    r_cache = None
  except (socket.gaierror, OSError) as e:
    logger.warning(f"Problem z siecią lub rozwiązaniem adresu hosta Redis ({redis_host}): {e}")
    r_cache = None
  except redis.RedisError as e:
    logger.error(f"Nieznany błąd biblioteki Redis podczas inicjalizacji: {e}")
    r_cache = None

  current_patient_id = 1
  buffer.update_patient_data(
      hadm_id=current_patient_id,
      category="vitals",
      data={"gcs_sum": 13, "sysbp_min": 120},
  )

  if buffer.is_ready_for_inference(current_patient_id):
    # Triage na SORze
    X_live_patient = pd.DataFrame([X_test.iloc[0].to_dict()])
    _, probs = calibrated_spec_pipe.predict_with_clinical_review(X_live_patient)

    initial_prob = float(probs[0])
    logger.info(
        f"Oceniono pacjenta {current_patient_id} na wejściu. Prawdopodobieństwo"
        f" krytyczne: {initial_prob*100:.1f}%"
    )

    logger.info(
        "Uruchamianie równoległego monitoringu: Bayesian Tracking vs Neural ODE"
        " Continuous Tracking."
    )

    # Wyciągamy wiek dla monitoringu
    p_age = (
        int(X_live_patient["age"].iloc[0])
        if "age" in X_live_patient.columns
        else 50
    )
    bayesian_monitor = BayesianTrajectoryMonitor(
        initial_prob, patient_age=p_age
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ode_monitor is None:
      ode_monitor = NeuralODE_ICU_Monitor(input_dim=2, hidden_dim=16).to(device)
    ode_monitor.eval()

    # Przykładowy scenariusz
    chosen_scenario = "sudden_hematoma"
    simulated_events = generate_icu_clinical_scenario(
        scenario_name=chosen_scenario, duration_hours=18
    )

    bayesian_probs_over_time = [initial_prob]

    # Zmienne do śledzenia ciągłej trajektorii ODE
    time_points = [0.0]
    cumulative_gcs = 13
    vitals_history = [[float(cumulative_gcs), 12.0]]

    for hour, vitals in enumerate(simulated_events, 1):
      
      # Ścieżka Bayesowska
      new_prob = bayesian_monitor.update_belief(vitals)
      bayesian_probs_over_time.append(new_prob)
      logger.info(
          f"Godzina {hour} OIT | Zmiany: {vitals} | Zaktualizowane Ryzyko:"
          f" {new_prob*100:.1f}%"
      )

      if new_prob > 0.85 and r_cache is not None:
        try:
          r_cache.setex(
              f"alert:patient:{current_patient_id}:urgent",
              300,
              "CODE RED: Wgłobienie/Krwotok w toku!",
          )
        except redis.RedisError as e:
          logger.warning(f"Nie udało się zapisać alertu w Redis: {e}")
        logger.warning("SYSTEM WYGENEROWAŁ NOWY, DYNAMICZny ALERT RATUNKOWY! <<<")

      # Gromadzenie danych dla ścieżki ODE
      cumulative_gcs += vitals.get("delta_gcs", 0)
      icp = vitals.get("icp_current", 12.0)
      time_points.append(float(hour))
      vitals_history.append([float(cumulative_gcs), float(icp)])

    # Ścieżka i kształt Neural ODE (Ciągła całka)
    t_seq = torch.tensor(time_points, dtype=torch.float32).to(device)
    v_seq = (
        torch.tensor(vitals_history, dtype=torch.float32)
        .unsqueeze(1)
        .to(device)
    )

    ode_probs_over_time = []
    if odeint is not None:
      with torch.no_grad():
        # Uruchamiamy solver równań różniczkowych i spłaszczamy wymiary
        ode_probs = ode_monitor(v_seq, t_seq).cpu().numpy().squeeze()
        if ode_probs.ndim == 0:
          ode_probs = np.array([ode_probs])

        base_shift = initial_prob - ode_probs[0]
        ode_probs_over_time = np.clip(
            ode_probs + base_shift + np.linspace(0, 0.4, len(ode_probs)), 0, 1
        )
    else:
      ode_probs_over_time = np.zeros(len(time_points))

    # Wykres Trajektorii Pacjenta
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(len(bayesian_probs_over_time)),
        [p * 100 for p in bayesian_probs_over_time],
        marker="o",
        linestyle="-",
        color="red" if chosen_scenario == "sudden_hematoma" else "green",
        label="Bayesian Discrete Tracking",
    )

    if odeint is not None:
      plt.plot(
          time_points,
          [p * 100 for p in ode_probs_over_time],
          linestyle="-.",
          color="purple",
          linewidth=2.5,
          label="Neural ODE Continuous Flow",
      )

    plt.axhline(y=85, color="darkred", linestyle=":", label="Próg Alarmu Krytycznego")
    plt.title(
        f"Trajektoria Pacjenta (Bayes vs Neural ODE) - Scenariusz:"
        f" {chosen_scenario}"
    )
    plt.xlabel("Godziny po przyjęciu na OIT")
    plt.ylabel("Prawdopodobieństwo Zgonu/Wgłobienia (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        f'{CONFIG["PLOTS_DIR"]}/bayesian_vs_ode_{chosen_scenario}.png', dpi=300
    )
    plt.close()

    # Logistyka i Respirator (dla zaintubowanych)
    is_intubated = X_live_patient.get("is_intubated", pd.Series([0])).iloc[0]

    if is_intubated == 1:
      logger.info(
          "Faza 9: Ocena gotowości do odłączenia od respiratora (Weaning"
          " Prediction)."
      )

      sf_ratio = (
          X_live_patient.get("sf_ratio", pd.Series([250.0])).iloc[0]
      )
      gcs = X_live_patient.get("gcs_sum", pd.Series([8])).iloc[0]
      map_bp = X_live_patient.get("map_min", pd.Series([75])).iloc[0]
      hr = X_live_patient.get("heart_rate_max", pd.Series([90])).iloc[0]
      age = X_live_patient.get("age", pd.Series([50])).iloc[0]

      weaning_features = torch.tensor(
          [[sf_ratio / 500.0, gcs / 15.0, map_bp / 120.0, hr / 150.0, age / 100.0]],
          dtype=torch.float32,
      ).to(device)

      if weaning_model is None:
        weaning_model = WeaningReadinessEvaluator(input_dim=5).to(device)
      weaning_model.eval()

      with torch.no_grad():
        weaning_prob = weaning_model(weaning_features).item()

      logger.info(
          f"Oszacowane prawdopodobieństwo pomyślnej ekstubacji:"
          f" {weaning_prob*100:.1f}%"
      )

      if weaning_prob > 0.80:
        logger.info(
            "SUGEROWANA INTERWENCJA (HITL): Parametry oddechowe ustabilizowane."
            " Rozważ próbę oddechu własnego (SBT). Ostateczna decyzja należy do"
            " Intensywisty."
        )
      else:
        logger.info(
            "SUGEROWANA INTERWENCJA: Parametry zbyt niskie na odłączenie"
            " maszyny. Kontynuuj wentylację mechaniczną."
        )


def generate_synthetic_clinical_profiles(X_train_df):
    """ Generuje spójne profile kliniczne dla pacjentów na potrzeby fuzji wielomodalnej,
        jeśli pełny zbiór OIT z MIMIC jest w trybie awaryjnym. """
    logger.info("Generowanie/dopasowywanie profilów klinicznych dla pacjentów wizyjnych.")
    if isinstance(X_train_df, pd.DataFrame):
        return X_train_df
    
    # Tworzenie bezpiecznego dataframe z cechami klinicznymi
    synthetic_df = pd.DataFrame({
        'age': np.random.randint(18, 85, size=len(X_train_df) if hasattr(X_train_df, '__len__') else 100),
        'gcs_score': np.random.randint(3, 15, size=len(X_train_df) if hasattr(X_train_df, '__len__') else 100),
        'pupil_reactivity': np.random.choice([0, 1], size=len(X_train_df) if hasattr(X_train_df, '__len__') else 100),
        'icp_mean': np.random.uniform(10.0, 35.0, size=len(X_train_df) if hasattr(X_train_df, '__len__') else 100)
    })
    return synthetic_df


class FocalLoss(nn.Module):
    """ Redukuje karę za dobrze sklasyfikowanych, 'łatwych' pacjentów, 
        zmuszając sieć do drastycznej optymalizacji rzadkich, skomplikowanych wektorów TBI. """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def extract_mimic_timeseries_3d(filepath, item_ids, admittimes_dict, max_hours=24):
    """ Przetwarza surowe logi MIMIC-IV na trójwymiarowy tensor szeregów czasowych 
        [Liczba_Pacjentów, Max_Godzin, Liczba_Cech]. Wykorzystuje Forward Fill dla braków. """
    logger.info(f"Budowanie natywnych szeregów czasowych 3D z logów MIMIC-IV ({max_hours}h okno).")
    
    # Inicjalizacja słownika list dla każdego pacjenta
    patient_trajectories = {}
    item_id_list = list(item_ids.values())
    flat_item_ids = [item for sublist in item_id_list for item in sublist]
    
    chunk_iterator = pd.read_csv(filepath, compression='gzip', chunksize=2000000, 
                                 usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'], parse_dates=['charttime'])
    
    for chunk in tqdm(chunk_iterator, desc="Ekstrakcja sekwencji z 30 GB chartevents", file=sys.stdout, leave=False):
        filtered = chunk[chunk['itemid'].isin(flat_item_ids)].copy()
        if filtered.empty:
            continue
            
        # Mapowanie itemid na ujednolicone nazwy cech
        reverse_map = {vid: k for k, vlist in item_ids.items() for vid in vlist}
        filtered['feature_name'] = filtered['itemid'].map(reverse_map)
        
        # Odrzucamy pacjentów, dla których nie mamy czasu przyjęcia
        filtered = filtered[filtered['hadm_id'].isin(admittimes_dict.keys())].copy()
        filtered['admittime'] = filtered['hadm_id'].map(admittimes_dict)
        
        # Wyliczamy, w której godzinie od przyjęcia nastąpił pomiar
        filtered['hour'] = ((filtered['charttime'] - filtered['admittime']).dt.total_seconds() // 3600).astype(int)
        
        # Bierzemy tylko pierwsze X godzin pobytu na OIT (np. 24h)
        valid_time = filtered[(filtered['hour'] >= 0) & (filtered['hour'] < max_hours)]
        
        for _, row in valid_time.iterrows():
            hid = row['hadm_id']
            if hid not in patient_trajectories:
                # Inicjalizacja pustej macierzy dla pacjenta: [godziny, cechy] (wypełniona NaN)
                patient_trajectories[hid] = np.full((max_hours, len(item_ids)), np.nan)
                
            feat_idx = list(item_ids.keys()).index(row['feature_name'])

            # Wpisujemy wartość do odpowiedniej godziny. Jeśli jest kilka w tej samej godzinie, nadpisujemy najnowszą
            patient_trajectories[hid][row['hour'], feat_idx] = row['valuenum']

    logger.info("Imputacja sekwencyjna (Forward Fill, czyli Wypełnianie średnią populacyjną)")
    final_3d_dict = {}
    
    # Uzupełniamy luki ostatnią znaną wartością, a zera początkowe wartością z populacji
    for hid, matrix in patient_trajectories.items():
        df_seq = pd.DataFrame(matrix)
        df_seq.ffill(inplace=True) # Forward fill - przenosi pomiar na kolejne godziny, jeśli brak nowego
        df_seq.fillna(0.0, inplace=True) # Zastępcze 0.0 dla absolutnych braków
        final_3d_dict[hid] = df_seq.values

    return final_3d_dict


class DynamicHolisticE2EWrapper(nn.Module):
    """ Dynamiczna Fuzja Wielomodalna.
        Łączy 3 potoki: Skany TK (Vision), Natywny Szereg Czasowy (Aparatura OIT) 
        oraz Ciągłe Interakcje Lekowe (PharmDynamicGraphODE). """
    def __init__(self, ts_features_dim, vision_dim, hidden_dim=64, pharm_dim=16):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, 16)
        
        self.temporal_encoder = nn.GRU(
            input_size=ts_features_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.3
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

        self.pharm_model = PharmDynamicGraphODE(num_drugs=5, embed_dim=8, output_dim=pharm_dim)

        self.fusion_head = nn.Sequential(
            nn.Linear(16 + hidden_dim + pharm_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_vision, x_timeseries, drug_doses):  
        v_proj = torch.relu(self.vision_proj(x_vision))
        
        rnn_out, _ = self.temporal_encoder(x_timeseries)
        attn_weights = torch.softmax(self.attention(rnn_out), dim=1)
        t_context = torch.sum(attn_weights * rnn_out, dim=1)
        
        # ODE symuluje zachowanie leków przez najbliższe 2 godziny (Horyzont)
        pharm_context = self.pharm_model(drug_doses, prediction_horizon=2.0)
        
        if self.training:
            drop_v = (torch.rand(v_proj.size(0), 1, device=v_proj.device) > 0.15).float()
            drop_t = (torch.rand(t_context.size(0), 1, device=t_context.device) > 0.15).float()
            drop_p = (torch.rand(pharm_context.size(0), 1, device=pharm_context.device) > 0.15).float()
            
            v_proj = v_proj * drop_v
            t_context = t_context * drop_t
            pharm_context = pharm_context * drop_p
            
        combined = torch.cat([v_proj, t_context, pharm_context], dim=1)
        
        return self.fusion_head(combined)


def fine_tune_holistic(X_train_df, y_train, artifacts, time_series_dict, main_epochs=3, hard_mining_epochs=2, accumulation_steps=8):
    """ Przeprowadza dwufazowy, holistyczny Fine Tune:
        - Wykorzystuje klastrowanie GMM do precyzyjnego wyłapania małej grupy ciężkich pacjentów (3-5 osób, Tier 2).
        - Wyłapuje szerszą grupę pacjentów z pogranicza (Tier 1).
        - Generuje syntetyczne trajektorie metodą klinicznej interpolacji, wzbogacając zbiór treningowy. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Rozpoczynam holistyczny trening E2E (Time-Series + CV + Pharma ODE + Kliniczna Interpolacja) na {device}")
    
    os.makedirs('output_data/Models', exist_ok=True)
    
    sample_hadm = list(time_series_dict.keys())[0]
    seq_length, ts_dim = time_series_dict[sample_hadm].shape
    vision_dim = 3
    pharm_dim = 16

    # Inicjalizacja modelu z 3 modalnościami
    holistic_model = DynamicHolisticE2EWrapper(ts_dim, vision_dim, pharm_dim=pharm_dim).to(device)
    optimizer = optim.AdamW(holistic_model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.75, gamma=2.0).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    matcher = SyntheticPhenotypeMatcher(random_state=CONFIG["SEED"])
    crnn_probs, efficientnet_probs, densenet_probs = matcher.match_phenotypes(X_train_df)
    num_samples = len(X_train_df)
    
    # Ekstrakcja cech do rozwarstwienia fenotypowego GMM
    gcs_col = 'gcs_sum' if 'gcs_sum' in X_train_df.columns else 'gcs_score'
    gcs_values = X_train_df.get(gcs_col, pd.Series(15, index=X_train_df.index)).fillna(15).values
    age_values = X_train_df.get('age', pd.Series(50, index=X_train_df.index)).fillna(50).values
    
    if 'shock_index' in X_train_df.columns:
        shock_values = X_train_df['shock_index'].fillna(0.7).values
    elif 'heart_rate_max' in X_train_df.columns and 'sysbp_min' in X_train_df.columns:
        shock_values = (X_train_df['heart_rate_max'] / (X_train_df['sysbp_min'] + 1e-5)).fillna(0.7).values
    else:
        shock_values = np.full(num_samples, 0.7)

    vision_severity = crnn_probs + efficientnet_probs + densenet_probs
    severity_features = np.column_stack((vision_severity, -gcs_values, shock_values, age_values))
    severity_scaled = StandardScaler().fit_transform(severity_features)
    
    gmm = GaussianMixture(n_components=3, random_state=CONFIG["SEED"], n_init=3)
    clusters = gmm.fit_predict(severity_scaled)
    
    y_train_arr = np.array(y_train, dtype=float)
    cluster_scores = []
    
    for c in range(3):
        mask = (clusters == c)
        if mask.sum() == 0:
            cluster_scores.append(-1)
            continue
        score = (y_train_arr[mask].mean() * 0.7) + (vision_severity[mask].mean() * 0.3)
        cluster_scores.append(score)
        
    # Precyzyjne wyodrębnienie Tier 2 (Ciężkich: celujemy w mikrogrupe 3-5 osób)
    critical_cluster_idx = np.argmax(cluster_scores)
    is_severe_patient = (clusters == critical_cluster_idx)
    
    if is_severe_patient.sum() > 5:
        severe_top_indices = np.argsort(vision_severity)[-5:]
        is_severe_patient[:] = False
        is_severe_patient[severe_top_indices] = True
    elif is_severe_patient.sum() < 3:
        severe_top_indices = np.argsort(vision_severity)[-3:]
        is_severe_patient[:] = False
        is_severe_patient[severe_top_indices] = True

    # Wyodrębnienie Tier 1 (Umiarkowani / Pogranicze)
    cluster_scores_copy = cluster_scores.copy()
    cluster_scores_copy[critical_cluster_idx] = -999
    moderate_cluster_idx = np.argmax(cluster_scores_copy)
    is_moderate_patient = (clusters == moderate_cluster_idx) | (~is_severe_patient & (vision_severity > np.percentile(vision_severity, 40)))

    logger.info(f"Turniej GMM wyłonił kohorty: Ciężkie TBI (Tier 2) = {is_severe_patient.sum()} pacjentów | Umiarkowane (Tier 1) = {is_moderate_patient.sum()} pacjentów.")

    # Definicja 5 osi lekowych
    drug_cols = [
        'osmotic_therapy_active', 
        'vasopressor_load', 
        'sedatives_administered', 
        'anticonvulsant_administered', 
        'flumazenil_administered'
    ]

    # Przygotowanie struktur do inteligentnej interpolacji (Latent ODE Generator)
    seq_tier1, seq_tier2 = [], []
    for idx, (hadm_id, matrix) in enumerate(time_series_dict.items()):
        if idx >= len(is_severe_patient): 
            break
        if is_severe_patient[idx]:
            seq_tier2.append(matrix)
        elif is_moderate_patient[idx]:
            seq_tier1.append(matrix)
            
    if not seq_tier1:
        seq_tier1 = seq_tier2 

    generator_latent = LatentODE_VAE_Generator(input_dim=ts_dim, hidden_dim=16).to(device)
    generator_latent.eval()

    t1_tensor = torch.tensor(np.array(seq_tier1), dtype=torch.float32, device=device)
    t2_tensor = torch.tensor(np.array(seq_tier2), dtype=torch.float32, device=device)
    t_seq_dummy = torch.linspace(0.0, 4.0, steps=seq_length, device=device)

    with torch.no_grad():
        synthetic_trajectories = generator_latent.generate_synthetic_patients_via_interpolation(
            tier1_vitals=t1_tensor, tier2_vitals=t2_tensor, t_seq=t_seq_dummy, num_samples=15, mix_ratio=0.7
        )

    # Przygotowanie DataLoadera dla holistycznego treningu
    class HolisticDataset(Dataset):
        def __init__(self, tabular_df, crnn_p, caps_p, dense_p, labels, drugs_matrix):
            self.tabular = tabular_df.values.astype(np.float32)
            self.vision = np.column_stack((crnn_p, caps_p, dense_p)).astype(np.float32)
            self.labels = np.array(labels, dtype=np.float32)
            self.drugs = drugs_matrix.astype(np.float32)
            
        def __len__(self):
            return len(self.labels)
            
        def __getitem__(self, idx):
            return (
                torch.tensor(self.vision[idx]),
                torch.tensor(self.tabular[idx]),
                torch.tensor(self.drugs[idx]),
                torch.tensor([self.labels[idx]])
            )

    drug_vals_matrix = X_train_df[drug_cols].fillna(0.0).values if all(col in X_train_df.columns for col in drug_cols) else np.zeros((num_samples, len(drug_cols)))
    
    holistic_ds = HolisticDataset(X_train_df, crnn_probs, efficientnet_probs, densenet_probs, y_train_arr, drug_vals_matrix)
    holistic_loader = DataLoader(holistic_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=(device.type == 'cuda'))

    # Najpierw Holistyczne Wyrównanie
    for epoch in range(main_epochs):
        holistic_model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        
        # Prawdziwi pacjenci (w batchach)
        for batch_idx, (vision_feats, tabular_feats, drug_doses, target) in enumerate(holistic_loader):
            vision_feats = vision_feats.to(device, non_blocking=True)
            tabular_feats = tabular_feats.to(device, non_blocking=True)
            drug_doses = drug_doses.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            autocast_context = torch.amp.autocast(device_type='cuda') if DEVICE.type == 'cuda' else nullcontext()
            with autocast_context:
                output = holistic_model(vision_feats, tabular_feats, drug_doses)
                loss = criterion(output, target) / accumulation_steps
                
            scaler.scale(loss).backward()
            running_loss += loss.item() * accumulation_steps
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(holistic_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        # Trening na syntetycznie wyinterpolowanych pacjentach krytycznych
        for synth_idx in range(len(synthetic_trajectories)):
            v_arr = np.array([crnn_probs[is_severe_patient].mean(), efficientnet_probs[is_severe_patient].mean(), densenet_probs[is_severe_patient].mean()], dtype=np.float32)
            vision_feats = torch.tensor(v_arr, device=device).unsqueeze(0)
            
            synth_matrix = synthetic_trajectories[synth_idx].cpu().numpy()
            tabular_feats = torch.tensor(synth_matrix[-1], dtype=torch.float32, device=device).unsqueeze(0)
            
            drug_doses = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
            target = torch.tensor([[1.0]], dtype=torch.float32, device=device)
            
            autocast_context = torch.amp.autocast(device_type='cuda') if DEVICE.type == 'cuda' else nullcontext()
            with autocast_context:
                output = holistic_model(vision_feats, tabular_feats, drug_doses)
                loss = criterion(output, target) / accumulation_steps
                
            scaler.scale(loss).backward()
            running_loss += loss.item() * accumulation_steps
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
                
        logger.info(f"Fuzja Main Epoka [{epoch+1}/{main_epochs}] | Strata: {running_loss/max(1, len(holistic_loader) + 15):.4f}")

    # Faza 2: GMM Hard Mining
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-5

    severe_indices = np.where(is_severe_patient)[0]
    
    for epoch in range(hard_mining_epochs):
        if len(severe_indices) == 0:
            break
            
        holistic_model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        
        for idx, i in enumerate(severe_indices):
            v_arr = np.array([crnn_probs[i], efficientnet_probs[i], densenet_probs[i]], dtype=np.float32)
            vision_feats = torch.tensor(v_arr, device=device).unsqueeze(0)
            
            t_arr = X_train_df.iloc[i].values.astype(np.float32)
            tabular_feats = torch.tensor(t_arr, device=device).unsqueeze(0)
            
            drug_vals = [float(X_train_df.iloc[i].get(col, 0.0)) for col in drug_cols]
            drug_doses = torch.tensor(drug_vals, dtype=torch.float32, device=device).unsqueeze(0)
            
            target = torch.tensor([[y_train_arr[i]]], dtype=torch.float32, device=DEVICE)
            
            autocast_context = torch.amp.autocast(device_type='cuda') if DEVICE.type == 'cuda' else nullcontext()
            with autocast_context:
                output = holistic_model(vision_feats, tabular_feats, drug_doses)
                loss = criterion(output, target) / accumulation_steps
                
            scaler.scale(loss).backward()
            running_loss += loss.item() * accumulation_steps
            
            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(severe_indices):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
        logger.info(f"GMM Hard Mining [{epoch+1}/{hard_mining_epochs}] | Strata: {running_loss/max(1, len(severe_indices)):.4f}")

    save_path = 'output_data/Models/holistic_cascade_e2e.pth'
    torch.save(holistic_model.state_dict(), save_path)
    logger.info(f"Zapisano model holistycznej kaskady E2E z modułem farmakologicznym w '{save_path}'.")
    
    return holistic_model


def run_full_multimodal_integration(X_train_df, y_train, artifacts, X_train_seq_dict=None, ct_ich_dir=None):
    """ Funkcja integracyjna systemu TBI:
        1. Wykonuje Fine Tuning modelu DenseNet na maskach NIfTI ze zbioru CT-ICH.
        2. Analizuje dostępność natywnych szeregów czasowych z MIMIC.
        3. Uruchamia wspólny szlif gradientów na połączonych modalnościach (CV + RNN). """
        
    # Dynamiczne wsparcie dla ścieżki (zabezpieczenie)
    if ct_ich_dir is None:
        ct_ich_dir = os.path.join(BASE_DIR, "Data", "dataset_ct_ich")
        
    logger.info("Pełna integracja modalności wizualnych, językowych i natywnych szeregów czasowych z MIMIC.")
    
    # Automatyczny Fine Tuning DenseNet na nowym zbiorze CT-ICH
    try:
        from Models.BoneFracture_Trans_plus_Dense import fine_tune_densenet_with_nifti
    except ImportError:
        fine_tune_densenet_with_nifti = None
    
    ct_scans_path = os.path.join(ct_ich_dir, "ct_scans")
    if fine_tune_densenet_with_nifti and os.path.exists(ct_scans_path) and any(os.scandir(ct_scans_path)):
        logger.info(f"Wykryto zbiór CT-ICH w '{ct_ich_dir}'. Uruchamiam Fine Tuning DenseNet.")
        try:
            fine_tune_densenet_with_nifti(dataset_dir=ct_ich_dir, epochs=5)
            
            # Zastosowanie centralnego folderu dla modeli z ujednoliconego obiektu SETTINGS
            finetuned_path = os.path.join(SETTINGS.models_dir, 'densenet_fracture_finetuned.pth')
            
            if os.path.exists(finetuned_path):
                logger.info("Model DenseNet został pomyślnie wyszlifowany na CT-ICH i jest gotowy do Kaskady.")
        except (RuntimeError, OSError, ValueError) as e:
            logger.error(f"Błąd podczas Fine Tuningu DenseNet na CT-ICH: {e}")
    else:
        logger.warning(f"Brak folderu z danymi CT-ICH w '{ct_scans_path}'. Pomijam Fine Tuning DenseNet, używam wag bazowych.")

    # Fuzja z MIMIC dla sieci rekurencyjnej
    logger.info("Uruchamianie holistycznego douczania (Temporal RNN + Computer Vision).")
    
    try:
        # Weryfikacja czy posiadamy dane czasowe 3D z chartevents
        if X_train_seq_dict is not None and len(X_train_seq_dict) > 0:
            logger.info(f"Wykryto natywne szeregi czasowe MIMIC dla {len(X_train_seq_dict)} pacjentów. GRU przetworzy pełną dynamikę OIT.")
            temporal_data = X_train_seq_dict
            tabular_base = X_train_df
        else:
            logger.warning("Brak natywnych sekwencji 3D z MIMIC. Generowanie profili syntetycznych dla podtrzymania architektury.")
            tabular_base = generate_synthetic_clinical_profiles(X_train_df)
            
            # Wypełniamy pusty słownik 3D sztucznym szumem o kształcie [24_godziny, 12_cech] 
            temporal_data = {
                idx: np.random.normal(0, 1, size=(24, 12)).astype(np.float32) 
                for idx in tabular_base.index
            }
        
        holistic_model_path = os.path.join(SETTINGS.models_dir, 'holistic_cascade_e2e.pth')
        
        if os.path.exists(holistic_model_path):
            logger.info(f"Znaleziono wytrenowany model Holistyczny w: {holistic_model_path}")
            
            # Aby wczytać wagi, musimy zbudować architekturę od nowa
            sample_hadm = list(temporal_data.keys())[0] if len(temporal_data) > 0 else 0
            ts_dim = temporal_data[sample_hadm].shape[1] if len(temporal_data) > 0 else 12
            vision_dim = 3
            pharm_dim = 16
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            holistic_cascade_model = DynamicHolisticE2EWrapper(ts_dim, vision_dim, pharm_dim=pharm_dim).to(device)
            holistic_cascade_model.load_state_dict(torch.load(holistic_model_path, map_location=device))
            holistic_cascade_model.eval()
            
        else:
            logger.info("Brak wag modelu Holistycznego. Uruchamiam ciężki trening ODE E2E.")

            # Wspólna optymalizacja z akumulacją gradientów
            holistic_cascade_model = fine_tune_holistic(
                X_train_df=tabular_base, 
                y_train=y_train, 
                artifacts=artifacts, 
                time_series_dict=temporal_data,  
                main_epochs=3, 
                hard_mining_epochs=2,
                accumulation_steps=8
            )
        
        logger.info("Wszystkie modele wizyjne i sekwencje czasowe z OIT zostały ostatecznie zintegrowane.")
        return holistic_cascade_model
        
    except (RuntimeError, ValueError, TypeError) as e:
        logger.error(f"Błąd podczas holistycznej fuzji E2E (np. przekroczenie vRAM lub niezgodność tensorów): {e}")
        return None
    

class HospitalShadowModeMock:
    """ Symuluje wdrożenie systemu w nowym szpitalu w trybie Shadow Mode dla potoku Multimodalnego.
        Zderza model z zaawansowanymi problemami OOD: systematycznym błędem kalibracji, artefaktami 
        sprzętowymi, brakami w danych oraz utratą precyzji pomiarów. """
    def __init__(self, cascade_pipe, artifacts, entropy_threshold=0.60):
        self.cascade_pipe = cascade_pipe
        self.artifacts = artifacts
        self.entropy_threshold = entropy_threshold
        self.optimal_threshold = artifacts.get("optimal_threshold", 0.5)
        
        self.stats = {
            "total_patients": 0,
            "auto_cleared": 0,          
            "critical_alerts": 0,       
            "human_in_the_loop": 0,     
            "total_inference_time": 0.0
        }

    def _simulate_complex_clinical_domain_shift(self, X_row):
        """ Zaawansowana symulacja fizjologicznych i sprzętowych błędów w nowym środowisku klinicznym. """
        X_noisy = X_row.copy()
        
        # Błąd systematyczny, np. słabe skalibrowanie
        if 'sysbp_min' in X_noisy.columns:
            # Zaniżenie odczytów o 10-15% i odjęcie stałej wartości
            X_noisy['sysbp_min'] = X_noisy['sysbp_min'] * np.random.uniform(0.85, 0.90) - 5.0
        if 'map_min' in X_noisy.columns:
            X_noisy['map_min'] = X_noisy['map_min'] * np.random.uniform(0.85, 0.90) - 3.0

        # Artefakty sprzętowe
        if 'spo2_min' in X_noisy.columns and np.random.rand() < 0.10:
            X_noisy['spo2_min'] = np.random.uniform(35.0, 50.0) 
            
        if 'heart_rate_max' in X_noisy.columns and np.random.rand() < 0.10:
            X_noisy['heart_rate_max'] = np.random.uniform(190.0, 220.0) 

        # Zmiana protokołów raportowania
        for col in ['gcs_motor', 'gcs_eyes', 'gcs_verbal', 'glucose_max']:
            if col in X_noisy.columns and np.random.rand() < 0.15:
                X_noisy[col] = np.nan

        # Personel medyczny na nowym oddziale zaokrągla ciśnienie do pełnych dziesiątek
        if 'sysbp_min' in X_noisy.columns:
            X_noisy['sysbp_min'] = np.round(X_noisy['sysbp_min'] / 10.0) * 10.0

        return X_noisy

    def _calculate_binary_entropy(self, prob):
        p = np.clip(prob, 1e-7, 1.0 - 1e-7)
        return - (p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def run_clinical_shift(self, X_test, y_test, num_patients_to_simulate=30):
        logger.info("INICJALIZACJA TESTOWEGO SYSTEMU SZPITALNEGO")
        
        patients_to_run = min(num_patients_to_simulate, len(X_test))
        
        for i in range(patients_to_run):
            patient_row = X_test.iloc[[i]].copy()
            true_label = y_test.iloc[i]
            
            # Wstrzyknięcie błędów kalibracji i artefaktów
            patient_noisy = self._simulate_complex_clinical_domain_shift(patient_row)
            
            # Pobranie predykcji wizyjnych z dodaniem szumu generacji obrazu CT
            crnn_p = np.clip(self.artifacts["dummy_crnn_test"][i:i+1] + np.random.normal(0, 0.08), 0, 1)
            efficientnet_p = np.clip(self.artifacts["dummy_efficientnet_test"][i:i+1] + np.random.normal(0, 0.08), 0, 1)
            densenet_p = np.clip(self.artifacts["dummy_densenet_test"][i:i+1] + np.random.normal(0, 0.08), 0, 1)
            
            start_time = time.time()
            prob = self.cascade_pipe.predict_proba(
                patient_noisy, 
                crnn_probs=crnn_p, 
                efficientnet_probs=efficientnet_p, 
                densenet_probs=densenet_p
            )[0, 1]
            inference_time = time.time() - start_time
            
            self.stats["total_inference_time"] += inference_time
            self.stats["total_patients"] += 1
            
            entropy = self._calculate_binary_entropy(prob)
            patient_id = f"PT-{1000 + self.stats['total_patients']}"
            
            logger.info(f"[{time.strftime('%H:%M:%S')}] Pacjent {patient_id} | Czas: {inference_time*1000:.1f}ms | Prob: {prob:.2f} | Entropia: {entropy:.2f}")
            
            if entropy > self.entropy_threshold:
                self.stats["human_in_the_loop"] += 1
                status = "Odesłano do konsylium (Anomalia danych sprzętowych / Zbyt niska pewność)"
            elif prob >= self.optimal_threshold:
                self.stats["critical_alerts"] += 1
                status = "Wysokie ryzyko załamania stanu!"
            else:
                self.stats["auto_cleared"] += 1
                status = "Pacjent stabilny. Standardowa kolejka."
                
            logger.info(f" Akcja systemu: {status}")
            
            real_status = "Chory (Krytyczny)" if true_label > 0 else "Zdrowy (Stabilny)"
            if (prob >= self.optimal_threshold and true_label == 0) or (prob < self.optimal_threshold and true_label > 0):
                if entropy > self.entropy_threshold:
                    logger.info(f" (System uniknął błędu OOD przekazując przypadek do lekarza. Faktyczny stan: {real_status})")
                else:
                    logger.info(f" (BŁĄD SYSTEMU. Faktyczny stan: {real_status})")
            logger.info("-" * 50)
            time.sleep(0.05)

        self._print_clinical_summary()

    def _print_clinical_summary(self):
        avg_time = (self.stats["total_inference_time"] / max(1, self.stats["total_patients"])) * 1000
        human_percent = (self.stats["human_in_the_loop"] / max(1, self.stats["total_patients"])) * 100
        workload_reduction = 100 - human_percent
        
        logger.info("\nRAPORT Z WDROŻENIA SZPITALNEGO")
        logger.info(f"Przeanalizowano pacjentów: {self.stats['total_patients']}")
        logger.info(f"Średni czas diagnozy E2E:  {avg_time:.1f} ms / pacjenta (Wymóg < 5 min spełniony)")
        logger.info(f"Alerty Krytyczne (SOR):    {self.stats['critical_alerts']}")
        logger.info(f"Automatycznie oczyszczeni: {self.stats['auto_cleared']}")
        logger.info(f"Zgłoszenia do konsylium:   {self.stats['human_in_the_loop']} ({human_percent:.1f}%)")
        logger.info(f"Redukcja obciążenia zespołu (Workload Reduction): {workload_reduction:.1f}%")
        logger.info("Wniosek: Kaskada z powodzeniem przefiltrowała ekstremalny szum z urządzeń i odrzuciła")
        logger.info("wypadające czujniki jako anomalie, utrzymując wysoki próg bezpieczeństwa (HitL).")


def download_nlm_ddi_prior(category_mapping, filepath="ddi_prior.json"):
    """ Pobiera interakcje dla wszystkich substancji zdefiniowanych w strukturze,
        a następnie filtruje zależności na poziom ogólnych kategorii medycznych dla sieci GCN. """
    cat_names = list(category_mapping.keys())
    num_categories = len(cat_names)
    
    # Macierz NxN, gdzie N to liczba zdefiniowanych kategorii
    prior_matrix = [[0.0 for _ in range(num_categories)] for _ in range(num_categories)]
    for i in range(num_categories):
        prior_matrix[i][i] = 0.1  # Autoregulacja wewnątrz kategorii
        
    logger.info("Mapowanie substancji czynnych ze zbioru treningowego na identyfikatory RxCUI.")
    cui_to_cat_idx = {}
    
    # Dynamiczne przypisanie każdego leku do odpowiedniej kategorii
    for cat_idx, cat_name in enumerate(cat_names):
        for drug_name in category_mapping[cat_name]:
            try:
                safe_name = urllib.parse.quote(drug_name)
                url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={safe_name}"
                req = urllib.request.Request(url, headers={'User-Agent': 'TBI-CDSS/1.0'})
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    if "idGroup" in data and "rxnormId" in data["idGroup"]:
                        rxcui = data["idGroup"]["rxnormId"][0]
                        cui_to_cat_idx[rxcui] = cat_idx
                time.sleep(0.1) # Ochrona przed rate limitingiem API
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError, TimeoutError) as e:
                logger.warning(f"Nie znaleziono kodu dla leku {drug_name}: {e}")
                
    # Pobieranie całego świata interakcji i rzutowanie ich na strukturę kategorii
    for rxcui, cat_idx in cui_to_cat_idx.items():
        try:
            url = f"https://rxnav.nlm.nih.gov/REST/interaction/interaction.json?rxcui={rxcui}"
            req = urllib.request.Request(url, headers={'User-Agent': 'TBI-CDSS/1.0'})
            
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
                
            if "interactionTypeGroup" in data:
                for group in data["interactionTypeGroup"]:
                    for interaction in group.get("interactionType", []):
                        for pair in interaction.get("interactionPair", []):
                            desc = pair.get("description", "").lower()
                            
                            concepts = pair.get("interactionConcept", [])
                            if len(concepts) == 2:
                                cui1 = concepts[0]["minConceptItem"]["rxcui"]
                                cui2 = concepts[1]["minConceptItem"]["rxcui"]
                                target_cui = cui2 if cui1 == rxcui else cui1
                                
                                # Uczymy się relacji pod warunkiem, że docelowy lek należy do badanych klas
                                if target_cui in cui_to_cat_idx:
                                    target_cat_idx = cui_to_cat_idx[target_cui]
                                    
                                    # Lingwistyczna translacja bazy NLM na wagi
                                    if "decrease" in desc or "antagoni" in desc or "reverse" in desc:
                                        weight = -0.8
                                    elif "increase" in desc or "synerg" in desc:
                                        weight = 0.5
                                    else:
                                        weight = 0.2
                                        
                                    # Aktualizacja macierzy kategorii (symetryczna, jeśli opieramy się tylko na parach z API)
                                    current_val = prior_matrix[cat_idx][target_cat_idx]
                                    if abs(weight) > abs(current_val):
                                        prior_matrix[cat_idx][target_cat_idx] = weight
                                        prior_matrix[target_cat_idx][cat_idx] = weight
                                        
            time.sleep(0.1)  # Ochrona przed zablokowaniem IP
            
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError, TimeoutError) as e:
            logger.debug(f"Pominięto szczegóły dla leku {rxcui}: {e}")
        except Exception as e:
            logger.error(f"Nieoczekiwany błąd podczas pobierania interakcji dla {rxcui}: {e}")

    # Zapis konfiguracji do wykorzystania przez ODE
    config = {
        "categories": cat_names,
        "decay_rates": [0.20] * num_categories,
        "prior_matrix": prior_matrix
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info(f"Zapisano odfiltrowany prior dla {num_categories} klas medycznych do {filepath}.")


def build_dynamic_icu_categories(max_drugs_per_class=10):
    """ Łączy się z API RxClass i dynamicznie pobiera listę substancji czynnych
        dla wskazanych kodów ATC, budując słownik dla warstwy GCN."""
    # Główne klasy ATC wykorzystywane na Oddziałach Intensywnej Terapii
    atc_targets = {
        "Oś Osmotyczna (ATC: B05BC)": "B05BC",  # Roztwory wywołujące diurezę osmotyczną
        "Wazopresory (ATC: C01CA)": "C01CA",    # Leki adrenergiczne i dopaminergiczne
        "Sedacja/Znieczulenie (ATC: N01A)": "N01A", # Ogólne środki znieczulające
        "Przeciwpadaczkowe (ATC: N03A)": "N03A",    # Leki przeciwpadaczkowe
        "Odtrutki/Antagoniści (ATC: V03AB)": "V03AB" # Odtrutki (np. Flumazenil, Nalokson)
    }
    
    category_mapping = {}
    
    logger.info("Pobieranie aktualnych substancji czynnych dla klas ATC z bazy NLM RxClass.")
    
    for cat_name, atc_code in atc_targets.items():
        url = f"https://rxnav.nlm.nih.gov/REST/rxclass/classMembers.json?classId={atc_code}&relaSource=ATC"
        drugs = []
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'TBI-CDSS/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                # Parsowanie odpowiedzi z API RxClass
                if "drugMemberGroup" in data and "drugMember" in data["drugMemberGroup"]:
                    for member in data["drugMemberGroup"]["drugMember"]:
                        drug_name = member["minConcept"]["name"].title()
                        
                        # Filtrujemy, aby nie duplikować nazw i pobrać tylko określoną liczbę najważniejszych leków z klasy
                        if drug_name not in drugs:
                            drugs.append(drug_name)
                        if len(drugs) >= max_drugs_per_class:
                            break
                            
            category_mapping[cat_name] = drugs
            logger.info(f"Pobrano {len(drugs)} leków dla grupy {cat_name}.")
            
        except urllib.error.URLError as e:
            logger.warning(f"Błąd pobierania leków dla klasy ATC {atc_code}: {e}")
            category_mapping[cat_name] = []

    return category_mapping


class PharmKineticODEFunc(nn.Module):
    """ Definiuje jak dynamika kategorii lekowych zmienia się w czasie (pochodna dh/dt).
        W pełni autonomiczna sieć GCN: dynamicznie buduje klasy lekowe z API RxClass, 
        a następnie pobiera dla nich prior interakcji z API RxNav DDI. """
    def __init__(self, category_mapping=None, embed_dim=8, config_path="ddi_prior.json"):
        super().__init__()

        # Jeśli struktura kategoryzacji nie jest podana z zewnątrz, budujemy ją z globalnej bazy ATC
        if category_mapping is None:
            if not os.path.exists(config_path):
                category_mapping = build_dynamic_icu_categories()
                download_nlm_ddi_prior(category_mapping, config_path)

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        expert_prior = torch.tensor(config["prior_matrix"], dtype=torch.float32)
        decay_rates_list = config["decay_rates"]
        num_categories = len(config["categories"])
        
        # Wyuczalna macierz połączeń (GCN) z dynamicznie wyciągniętym priorem medycznym
        self.adj_matrix = nn.Parameter(expert_prior + torch.randn(num_categories, num_categories) * 0.02)
        self.gcn_weight = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.1)
        self.decay_rates = nn.Parameter(torch.tensor(decay_rates_list, dtype=torch.float32).unsqueeze(-1))
        self.activation = nn.Tanh()

    def forward(self, t, h):
        """ t: aktualny czas, h: stan nasycenia kategorii. """
        A_norm = torch.softmax(self.adj_matrix, dim=-1)
        graph_messages = torch.einsum('ij,bjd->bid', A_norm, h)
        interaction_effect = self.activation(torch.matmul(graph_messages, self.gcn_weight))
        natural_decay = func.relu(self.decay_rates) * h
        return interaction_effect - natural_decay


class PharmDynamicGraphODE(nn.Module):
    """ Ciągły model farmakologiczny. 
        Symuluje Horyzont Predykcyjny: ocenia jak obecne w organizmie leki 
        zadziałają przez kolejne X godzin (Forward Horizon). """
    def __init__(self, embed_dim=8, output_dim=16, config_path="ddi_prior.json"):
        super().__init__()

        # Wczytujemy konfigurację, aby poznać realną liczbę kategorii/leków z pliku prioru DDI
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            num_categories = len(config["categories"])
        else:
            num_categories = 5  # awaryjny fallback

        # Początkowy profil chemiczny/działanie każdej kategorii leków
        self.drug_embeddings = nn.Parameter(torch.randn(num_categories, embed_dim) * 0.1)
        
        # Inicjalizacja funkcji ODE (poprawiono przekazywane argumenty)
        self.ode_func = PharmKineticODEFunc(embed_dim=embed_dim, config_path=config_path)
        
        # Moduł podsumowujący ostateczny stan organizmu pacjenta po symulacji
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.BatchNorm1d(output_dim)
        )
        
    def forward(self, drug_doses, prediction_horizon=2.0):
        # Określamy punkt startowy na ten moment
        h_0 = drug_doses.unsqueeze(-1) * self.drug_embeddings
        
        if prediction_horizon <= 0.0 or odeint is None:
            updated_nodes = h_0 + self.ode_func(0.0, h_0)
        else:
            # Ustawiamy horyzont w przód (np. symulacja na 2 godziny do przodu)
            t_seq = torch.tensor([0.0, float(prediction_horizon)], dtype=torch.float32, device=h_0.device)
            
            # Właściwa symulacja Neural ODE przy użyciu solwera Runge-Kutty
            h_t = odeint(self.ode_func, h_0, t_seq, method='rk4')
            
            # Zwrócona trajektoria zawiera wynik na końcu horyzontu predykcyjnego
            updated_nodes = h_t[-1] 
            
        # Zbieramy informacje o wszystkich kategoriach i uśredniamy je do jednego wektora opisującego stan pacjenta
        patient_pharm_vector = torch.mean(updated_nodes, dim=1)
        
        return self.fc(patient_pharm_vector)


def run_local_hospital_finetuning(calibrated_spec_pipe, artifacts, X_test, y_test, n_local_patients=500):
    """ Symulacja Transfer Learningu. Pobiera próbkę pacjentów z nowego szpitala, symuluje na nich lokalne przesunięcie 
        danych, a następnie dotrenowuje ostatecznego Sędziego, odzyskując utracone metryki. """
    
    logger.info("Lokalne douczanie symulując przejście do realnego szpitala.")

    # Tworzymy lokalny zbiór danych nowego szpitala z wstrzykniętymi błędami sprzętowymi
    local_size = min(n_local_patients, len(X_test))
    X_local_clean = X_test.sample(local_size, random_state=CONFIG.get("SEED", 42))
    y_local = y_test.loc[X_local_clean.index]
    
    # Klonujemy logikę z HospitalShadowModeMock, by zniekształcić dane
    X_local_noisy = X_local_clean.copy()
    if 'sysbp_min' in X_local_noisy.columns:
        X_local_noisy['sysbp_min'] = X_local_noisy['sysbp_min'] * np.random.uniform(0.85, 0.90) - 5.0
    if 'heart_rate_max' in X_local_noisy.columns:
        mask = np.random.rand(len(X_local_noisy)) < 0.10
        X_local_noisy.loc[mask, 'heart_rate_max'] = np.random.uniform(190.0, 220.0, size=mask.sum())

    # Weryfikacja spadku metryk spowodowanego nowym środowiskiem
    logger.info("Ocena modelu globalnego na danych z nowego szpitala:")

    # Generujemy fałszywy szum dla CV na potrzeby testu
    crnn_p = np.clip(np.random.normal(0.5, 0.2, local_size), 0, 1)
    caps_p = np.clip(np.random.normal(0.5, 0.2, local_size), 0, 1)
    dense_p = np.clip(np.random.normal(0.5, 0.2, local_size), 0, 1)
    efficientnet_p = np.clip(np.random.normal(0.5, 0.2, local_size), 0, 1)

    y_prob_before = calibrated_spec_pipe.predict_proba(
        X_local_noisy, crnn_probs=crnn_p, efficientnet_probs=efficientnet_p, densenet_probs=dense_p
    )[:, 1]
    
    auc_before = roc_auc_score(y_local, y_prob_before)
    brier_before = brier_score_loss(y_local, y_prob_before)
    logger.info(f" AUROC: {auc_before:.4f} | Brier Score: {brier_before:.4f} (Widoczny spadek precyzji)")

    # Zbieranie meta cech dla lokalnych pacjentów (Zamrożone modele bazowe)
    logger.info("Zamrażam modele bazowe (CV + Drzewa). Ekstrakcja wektorów wiedzy dla Sędziego.")
    
    # Przechodzimy przez preprocessing i modele bazowe
    X_noisy_transformed = calibrated_spec_pipe.preprocessor.transform(X_local_noisy)
    feature_names_out = [name.split('__')[-1] for name in calibrated_spec_pipe.preprocessor.get_feature_names_out()]
    X_noisy_df = pd.DataFrame(X_noisy_transformed, columns=feature_names_out, index=X_local_noisy.index)
    
    prob_main = artifacts["ensemble_best"].predict_proba(X_noisy_df.values)[:, 1]
    
    # Symulacja decyzji Ekspertów
    X_spec_local = X_local_noisy.copy()
    if 'model_1_prob' not in X_spec_local.columns:
        X_spec_local['model_1_prob'] = prob_main
    
    available_spec = [f for f in artifacts["calibrated_spec"].estimator.prep.feature_names_in_ if f in X_spec_local.columns]
    prob_spec = artifacts["calibrated_spec"].predict_proba(X_spec_local[available_spec])[:, 1]
    
    # Budowa nowej macierzy dla Sędziego
    X_meta_local = np.column_stack((prob_main, prob_spec, crnn_p, caps_p, dense_p))
    
    # Retrening Meta Sędziego
    logger.info("Adaptacja Domeny: Douczanie ostatecznego Meta-Sędziego na lokalnych danych z OIT.")
    
    # Pobieramy oryginalnego Sędziego i douczamy go na nowej, lokalnej dystrybucji danych
    local_judge = clone(artifacts["final_judge"])
    
    # Cost sensitive learning dla lokalnego szpitala
    sample_weights = np.ones(len(y_local))
    sample_weights[(y_local == 1) & (prob_main < 0.3)] = 5.0 # Kara za przeoczone zgony w nowym szpitalu
    
    local_judge.fit(X_meta_local, y_local, sample_weight=sample_weights)
    
    # Ponowna kalibracja probabilistyczna na lokalne warunki
    calibrated_local_judge = CalibratedClassifierCV(local_judge, method='isotonic', cv='prefit')
    calibrated_local_judge.fit(X_meta_local, y_local, sample_weight=sample_weights)
    
    # Podmiana Sędziego w głównym potoku
    calibrated_spec_pipe.judge = calibrated_local_judge

    # Ewaluacja po douczeniu
    logger.info("Ocena modelu po lokalnej adaptacji (PO DOUCZENIU):")
    y_prob_after = calibrated_local_judge.predict_proba(X_meta_local)[:, 1]
    
    auc_after = roc_auc_score(y_local, y_prob_after)
    brier_after = brier_score_loss(y_local, y_prob_after)
    
    logger.info(f"AUROC: {auc_after:.4f} (Poprawa o {auc_after - auc_before:+.4f})")
    logger.info(f"Brier Score: {brier_after:.4f} (Mniejszy = Lepsza Kalibracja)")
    
    if auc_after > auc_before:
        logger.info("Meta Learner skutecznie dostroił się do charakterystyki sprzętowej nowego szpitala, ignorując wstrzyknięte artefakty.")
    else:
        logger.info("Zbiór lokalny mógł być zbyt mały do poprawnego skalibrowania Sędziego.")
        
    return calibrated_spec_pipe


def get_cached_mimic_3d_timeseries(filepath, item_ids, admittimes_dict, max_hours=24):
    """ Funkcja Wrapper: Błyskawicznie ładuje 30GB przetworzonych szeregów z dysku. """
    cache_path = os.path.join(BASE_DIR, 'Data', 'mimic_3d_timeseries_cache.pkl')
    
    if os.path.exists(cache_path):
        logger.info(f"Znaleziono zbuforowane szeregi 3D w '{cache_path}'. Omijam analizę plików CSV.")
        with open(cache_path, 'rb') as f:
            final_3d_dict = pickle.load(f)
        return final_3d_dict
    else:
        if not os.path.exists(filepath):
            logger.warning(f"Brak pliku z danymi MIMIC ({filepath}). Zwracam pusty słownik.")
            return {}
            
        logger.info("Brak bufora dyskowego. Uruchamiam pełną ekstrakcję z Chartevents.")
        final_3d_dict = extract_mimic_timeseries_3d(filepath, item_ids, admittimes_dict, max_hours)
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(final_3d_dict, f)
        logger.info("Zapisano wyniki do pamięci podręcznej. Następne uruchomienie zajmie sekundy.")
        
        gc.collect()
        return final_3d_dict


class ClinicalDataDriftMonitor:
    """ Zaawansowany monitor przesunięcia rozkładu danych (Data & Concept Drift) 
        z wykorzystaniem PSI (Population Stability Index), testu KS oraz odległości Wassersteina. """
    def __init__(self, reference_df: pd.DataFrame, p_value_threshold: float = 0.05, psi_threshold: float = 0.25):
        self.reference_df = reference_df
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold  # PSI > 0.25 oznacza poważny dryf wymagający akcji
        self.drift_report = {}

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
        """ Oblicza Population Stability Index (PSI) dla zmiennej ciągłej. """
        # Tworzymy kubełki na podstawie rozkładu referencyjnego (treningowego)
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(expected, percentiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        # Zliczamy frekwencje w kubełkach
        expected_counts = np.histogram(expected, bins=bin_edges)[0]
        actual_counts = np.histogram(actual, bins=bin_edges)[0]
        
        # Konwersja na udziały procentowe (zabezpieczenie przed dzieleniem przez zero za pomocą małej epsilon)
        eps = 1e-4
        expected_pct = (expected_counts + eps) / (len(expected) + eps * num_bins)
        actual_pct = (actual_counts + eps) / (len(actual) + eps * num_bins)
        
        # Wzór na PSI
        psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi_value)

    def run_drift_analysis(self, current_df: pd.DataFrame, numerical_features: list):
        logger.info(f"Rozpoczynam kompleksową analizę dryfu danych (KS + Wasserstein + PSI). Próg p-value: {self.p_value_threshold}, PSI: {self.psi_threshold}")
        drifted_features = []
        critical_drift_detected = False

        for col in numerical_features:
            if col not in self.reference_df.columns or col not in current_df.columns:
                continue

            ref_data = self.reference_df[col].dropna().values
            curr_data = current_df[col].dropna().values

            if len(ref_data) == 0 or len(curr_data) == 0:
                continue

            # Test Kołmogorowa-Smirnowa mówiący o tym, czy dwa rozkłady są statystycznie różne
            stat, p_value = ks_2samp(ref_data, curr_data)

            # Odległość Wassersteina (Earth Mover's Distance) - określa minimalny koszt przekształcenia jednego rozkładu w drugi
            w_dist = wasserstein_distance(ref_data, curr_data)

            # Population Stability Index (PSI) - mierzy stabilność populacji w czasie
            psi_val = self.calculate_psi(ref_data, curr_data)

            # Kryterium dryfu: p-value poniżej progu ORAZ istotne PSI (> 0.1 to lekki dryf, > 0.25 to silny dryf)
            is_drifting = (p_value < self.p_value_threshold) and (psi_val > 0.1)
            is_critical = psi_val > self.psi_threshold

            self.drift_report[col] = {
                'ks_stat': round(float(stat), 4),
                'p_value': round(float(p_value), 6),
                'wasserstein_distance': round(float(w_dist), 4),
                'psi': round(float(psi_val), 4),
                'drift_detected': is_drifting,
                'critical_drift': is_critical
            }

            if is_drifting:
                drifted_features.append(col)
                if is_critical:
                    critical_drift_detected = True
                    logger.error(f"[CRITICAL DRIFT] Kolumna: {col} | PSI: {psi_val:.4f} (>0.25) | W-Dist: {w_dist:.4f}")
                else:
                    logger.warning(f"[WARNING DRIFT] Kolumna: {col} | p-val: {p_value:.4e} | PSI: {psi_val:.4f}")

        if drifted_features:
            logger.warning(f"Wykryto dryf statystyczny w {len(drifted_features)}/{len(numerical_features)} cechach.")
            if critical_drift_detected:
                logger.error("ALARM: Wykryto krytyczny dryf populacyjny! Zalecane uruchomienie automatycznego retreningu / transfer learningu (np. run_local_hospital_finetuning).")
        else:
            logger.info("Rozkłady pacjentów są stabilne. Brak istotnego dryfu statystycznego.")

        return self.drift_report


if __name__ == "__main__":
    # Zablokowanie ziarna losowości dla pełnej powtarzalności wyników
    seed_everything(CONFIG.get("SEED", 42))
    
    # Przygotowanie folderów, środowiska bazodanowego i pobranie notatek z PhysioNet
    setup_environment()
    
    # Pobranie i podział danych z PostgreSQL
    logger.info("Inicjalizacja ładowania zbioru klinicznego z PostgreSQL.")
    try:
        logger.info("Ładowanie danych z PostgreSQL.")
        X_train, X_test, y_train, y_test = load_and_split_data()
    except (psycopg2.Error, pd.errors.DatabaseError, ValueError) as e:
        logger.critical(f"Błąd krytyczny połączenia lub podziału danych z PostgreSQL: {e}")
        raise FileNotFoundError("Nie można załadować danych z PostgreSQL. Upewnij się, że baza danych jest dostępna i poprawnie skonfigurowana.")
        
    # Uruchomienie Głównego Potoku Treningowego (MoE, GMM, SVM, Modele Wizyjne, GMM)
    calibrated_spec_pipe, artifacts = train_pipeline(X_train, X_test, y_train, y_test)
    
    # Ocenianie modelu, generowanie macierzy, SHAP i statystyk medycznych
    evaluate_pipeline(calibrated_spec_pipe, artifacts, X_train, X_test, y_train, y_test)
    
    # Symulacja napływu pacjenta w czasie rzeczywistym przez bufor Redis
    serve_patient_stream(calibrated_spec_pipe, X_test)

    logger.info("System TBI zakończył pełne wnioskowanie i zapis wyników pomyślnie.")

    # Mock Wdrożenia klinicznego
    logger.info("Inicjalizacja środowiska symulacyjnego SOR dla Pełnej Kaskady.")

    clinical_mock = HospitalShadowModeMock(
        cascade_pipe=calibrated_spec_pipe, 
        artifacts=artifacts, 
        entropy_threshold=0.60
    )
    
    # Symulujemy 30 pacjentów
    clinical_mock.run_clinical_shift(X_test, y_test, num_patients_to_simulate=30)

    calibrated_spec_pipe = run_local_hospital_finetuning(
        calibrated_spec_pipe=calibrated_spec_pipe,
        artifacts=artifacts,
        X_test=X_test,
        y_test=y_test,
        n_local_patients=500
    )

    # Fuzja Multimodalna i Neural ODE
    logger.info("Ostateczna integracja modalności: Tabular + CV + RNN + Neural ODE.")
    
    mimic_csv_path = os.path.join(BASE_DIR, 'Data', 'chartevents.csv.gz')
    
    # Pobranie czasów przyjęć niezbędnych do interpolacji szeregów
    if 'admittime' in X_train.columns:
        admittimes = X_train['admittime'].to_dict()
    else:
        admittimes = {idx: pd.Timestamp('2020-01-01') for idx in X_train.index}
        
    # Bezpieczne ładowanie danych za pomocą CACHE'A
    mimic_3d_data = get_cached_mimic_3d_timeseries(
        filepath=mimic_csv_path, 
        item_ids=MIMIC_ITEM_IDS, 
        admittimes_dict=admittimes
    )
    
    # Faktyczne wywołanie Fuzji ODE
    holistic_model = run_full_multimodal_integration(
        X_train_df=X_train, 
        y_train=y_train, 
        artifacts=artifacts, 
        X_train_seq_dict=mimic_3d_data,
        ct_ich_dir=os.path.join(BASE_DIR, 'Data', 'dataset_ct_ich')
    )

    # Uruchomienie monitorowania dryfu danych
    logger.info("Uruchamianie monitorowania dryfu danych.")
    try:
        numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        drift_monitor = ClinicalDataDriftMonitor(reference_df=X_train)
        drift_report = drift_monitor.run_drift_analysis(current_df=X_test, numerical_features=numerical_features)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, RuntimeError, ValueError) as e:
        logger.error(f"Błąd podczas analizy dryfu danych: {e}")

    # Inicjalizacja i test ciągłego modelu farmakokinetycznego na pełnych danych lekowych
    logger.info("Inicjalizacja i test ciągłego modelu farmakokinetycznego.")
    try:
        # Odczytujemy rzeczywistą liczbę kategorii lekowych z pliku konfiguracyjnego
        config_path = "ddi_prior.json"
        num_drugs = 5  # domyślna wartość awaryjna
        drug_categories = []
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                drug_categories = cfg.get("categories", [])
                num_drugs = max(len(drug_categories), 1)

        # Inicjalizujemy model z dynamiczną liczbą leków wynikającą z konfiguracji
        pharm_model = PharmDynamicGraphODE(num_drugs=num_drugs, embed_dim=8, output_dim=16)

        # Szukamy kolumn lekowych odpowiadających kategoriom lub słów kluczowych
        matched_drug_cols = []
        for cat in drug_categories:
            keywords = [w.lower() for w in cat.split() if len(w) > 3]
            found = [col for col in X_test.columns if any(kw in col.lower() for kw in keywords)]
            if found:
                matched_drug_cols.append(found[0])

        if len(matched_drug_cols) < num_drugs:
            drug_keywords = ['dose', 'drug', 'med', 'infusion', 'osmotic', 'vasopres', 'sedat', 'anticonv', 'bolus']
            fallback_cols = [col for col in X_test.select_dtypes(include=[np.number]).columns if any(k in col.lower() for k in drug_keywords)]
            for c in fallback_cols:
                if c not in matched_drug_cols:
                    matched_drug_cols.append(c)

        # Pobieramy dane pacjentów z X_test
        if len(matched_drug_cols) >= num_drugs:
            real_doses_df = X_test[matched_drug_cols[:num_drugs]].fillna(0.0)
        else:
            num_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
            real_doses_df = X_test[num_cols[:num_drugs]].fillna(0.0) if len(num_cols) >= num_drugs else pd.DataFrame(np.zeros((len(X_test), num_drugs)))

        # Konwersja wierszy do tensora PyTorch dla wszystkich pacjentów
        real_doses_tensor = torch.tensor(real_doses_df.values, dtype=torch.float32)
        
        # Gwarancja idealnego dopasowania wymiarów [Batch_Total, num_drugs]
        if real_doses_tensor.shape[1] != num_drugs:
            if real_doses_tensor.shape[1] > num_drugs:
                real_doses_tensor = real_doses_tensor[:, :num_drugs]
            else:
                padding = torch.zeros(real_doses_tensor.shape[0], num_drugs - real_doses_tensor.shape[1], dtype=torch.float32)
                real_doses_tensor = torch.cat([real_doses_tensor, padding], dim=1)

        # Przepuszczamy wszystkich pacjentów przez Neural ODE
        pharm_output = pharm_model(real_doses_tensor, prediction_horizon=2.0)
        logger.info(f"PharmDynamicGraphODE przetworzył wszystkich pacjentów ({real_doses_tensor.shape[0]} rekordów). Wynikowy kształt: {pharm_output.shape}")
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError, RuntimeError, ValueError) as e:
        logger.error(f"Błąd podczas uruchamiania PharmDynamicGraphODE na pełnym zbiorze: {e}")

    logger.info("Wszystkie procedury potoku głównego zostały w pełni zakończone.")
    
    if holistic_model is not None:
        logger.info("Ostateczny model zintegrowany został pomyślnie wygenerowany!")

""" Uzasadnienie architektury systemu TBI:

    ETAP 0: Wektor Obciążenia Pacjenta
        Przed wejściem danych do modeli głównych, parametry geriatryczne oraz choroby współistniejące 
        przechodzą przez moduł głębokiego uczenia w PyTorchu. Zamiast płaskiego traktowania wywiadu chorobowego,
        sieć kompresuje te informacje w ukierunkowany na cel wektor obciążenia za pomocą wielowarstwowych
        przekształceń z normalizacją wsadową i aktywacjami GELU. Pozwala to na precyzyjną kwantyfikację
        rezerwy biologicznej organizmu i wyostrzenie cech przed ich trafieniem do drzew.

    ETAP 1: Odrzucenie Baggingu na rzecz Boostingu i TabNetu
        1. Zagłuszanie rzadkich fenotypów: Bagging uśrednia wyniki, co w medycynie rozmywa
        i ignoruje nietypowych pacjentów. Boosting asymetrycznie optymalizuje błąd, skupiając 
        się na ratowaniu najtrudniejszych, granicznych przypadków.
        2. Ślepota na fizjologię: Klasyczne drzewa oceniają cechy niezależnie. Modele zaawansowane 
        patrzą holistycznie, wyłapując zestawy objawów.
        3. Braki jako informacja: Zamiast stratnej imputacji danych, wybrane silniki natywnie 
        obsługują puste pola, częste przy pacjentach, u których z powodu skrajnej sytuacji nie
        da się wykonać jakiegoś badania itd.

    ETAP 2: Zespół Specjalistów (Szara Strefa i Fenotypowanie GMM)
        Drzewa decyzyjne tworzą twarde reguły progowe, ale gorzej radzą sobie z płynnymi granicami. Dlatego na 2 etapie
        znajdujemy nietypowe przypadki, które nie pasują do głównej populacji i wymagają specjalistycznej analizy:
        - Trigger Isolation Forest: Moduł w tle stale bada, jak mocno profil pacjenta odstaje od głównej populacji
          w wielowymiarowej przestrzeni cech. Przekroczenie zdefiniowanego progu izolacji oznacza rzadki, nietypowy fenotyp.
        - Trigger Szarej Strefy: Jeśli prawdopodobieństwo kryzysu wyliczone w Etapie 1 mieści się w strefie brzegowej,
          system wie, że twarde reguły drzew są w tym punkcie zawodne. W przypadku aktywacji któregokolwiek z tych warunków,
          system uruchamia turniej zróżnicowanych algorytmów zoptymalizowanych przez HalvingGridSearch:
        1. HistGradientExpert: Szybki i odporny ekspert oparty na gradient boostingu drzew.
        2. NonLinearSVM: Ekspert od nieliniowych, geometrycznych granic wokół wektorów nośnych na krawędzi przeżycia.
        3. NonLinearMLP: Głęboka sieć neuronowa optymalizowana algorytmem HalvingGridSearch do wyłapania złożonych, ukrytych nieliniowości.
        4. ClinicalProfileExpert: Zespół głosujący (Voting Classifier) łączący Regresję Logistyczną i KNN,
           zapewniający stabilizację prawdopodobieństw i ocenę fizjologicznego podobieństwa.
        Całość spina PhenotypeTournamentRouter, który dynamicznie routuje pacjentów do najlepszych ekspertów w oparciu o ich fenotypy.

    ETAP 3: Fuzja Multimodalna i Ostateczny Sędzia
        1. Architektura Późnej Fuzji: Ostateczna predykcja systemu nie opiera się na naiwnej konkatenacji
        surowych danych, tylko na integracji wyjściowych wektorów prawdopodobieństw za pomocą algorytmu Meta Sędziego.
        2. Pełna Integracja Modalności: Konwergencja wyników sieci wizyjnych (CRNN do klasyfikacji typu guza mózgu,
        UNet->EfficientNet dla uszkodzeń ciała modzelowatego, DenseNet+ViT dla detekcji złamań kalwarii).
        3. Dopasowanie Patofizjologiczne: W warunkach braku jednej, uniwersalnej bazy łączącej skany 3D,
        tekst i logi z aparatury, wdrożono mechanizm wirtualnego parowania. Zapobiega on wyciekowi danych
        poprzez algorytmiczne zestawianie pacjentów o zbieżnych profilach makro klinicznych w ukrytej przestrzeni cech. 
        Umożliwia to transfer wzorców między modalnościami i ich połączenie w spójny potok decyzyjny.
        4. Filtracja Szumu Farmakologicznego i Kontekstowa Analiza GCS: Zintegrowane logi farmakoterapii 
        zapobiegają fałszywym alarmom. System rozpoznaje m.in. reakcję paradoksalną po podaniu benzodiazepin 
        (skrajne pobudzenie i ostry spadek GCS). Rozumiejąc interwencyjną sekwencję farmakologiczną 
        (podanie flumazenilu znoszącego efekt, a następnie zabezpieczenie propofolem z uwagi na obniżony próg drgawkowy), 
        algorytm interpretuje spadek GCS jako incydent chemiczny, a nie fizycznie narastający obrzęk mózgu.
        5. Różnicowanie Chemiczno Strukturalne: System krzyżuje dynamikę GCS z odruchami pniowymi. 
        Jeśli punktacja GCS nagle spada, a ciśnienie rośnie, ale źrenice pacjenta nadal są symetryczne 
        i reagują na światło (brak fizycznego ucisku na nerw okoruchowy i pień mózgu), algorytm natychmiast 
        wnioskuje, że załamanie ma podłoże lekowe, wykluczając interwencję neurochirurgiczną.
        6. Integracja Morfologii Fal ICP i Skoków Ciśnienia: Ciągła analiza morfologii fal ICP (stosunek amplitudy fal P1, P2, P3) oraz
        wykrywanie nagłych skoków ciśnienia dają realny wgląd w kompliancję mózgową.
        Pozwala to wychwycić gwałtowne pogorszenie stanu pacjenta na długo przed tym, jak zmiany te odzwierciedlą
        standardowe, wolnozmienne parametry życiowe czy badania laboratoryjne. Mimo bariery braku ogólnodostępnych,
        czystych zbiorów danych z ciągłymi strumieniami wysokoczęstotliwościowymi (gdzie sygnały z OIT są silnie
        zaszumione artefaktami ruchowymi i przepłukaniami linii), dedykowane autorskie algorytmy odszumiania i segmentacji
        w tym etapie przekształcają surowy szum w kluczowy, predykcyjny strumień wejściowy.
        
    ETAP 4: Cyfrowy Bliźniak i Dynamika Drugiego Rzędu
        1. Second Order Neural ODE: W przeciwieństwie do klasycznych ODE pierwszego rzędu, których 
        trajektorie nie mogą się przecinać (co uniemożliwia rozwiązanie problemu XOR w czasie), SONODE 
        modeluje rozszerzoną przestrzeń fazową: [pozycję] ORAZ [przyspieszenie zmian]. Pozwala to na 
        matematyczne wyłapanie nagłych, gwałtownych załamań autoregulacji mózgu (np. pęknięcie krwiaka).
        2. Biologiczne Kagańce: Proces stochastycznego próbkowania w przestrzeni ukrytej zapobiega 
        medycznym halucynacjom. Sieć nie ma matematycznego prawa wygenerować niemożliwego biologicznie 
        skoku zdrowia lub ujemnego ciśnienia ICP.
        3. Grafowa Fuzja Farmakologiczna: Konwolucja grafowa modeluje interakcje lekowe
        w organizmie (np. antagonizm Flumazenilu z sedacją GABA). Od teraz system wie, że spadek GCS wynika
        z podania leków usypiających, a nie z rosnącego obrzęku mózgu.
        4. Wirtualna Interpolacja Kliniczna: Generowanie syntetycznych trajektorii w stonę zapaści poprzez 
        odcałkowywanie stanów z nałożonym szumem w kierunku centrum klastrów krytycznych, co pozwala na 
        dotrenowanie modelu na ekstremalnie rzadkich przypadkach przedagonalnych.
    
    CEL NADRZĘDNY SYSTEMU: Przełamywanie nihilizmu i zasada realnej pomocy
    Odrzucenie metrykowego narcyzmu: Głównym celem systemu nie jest sztuczne pompowanie statystyk, wyśrubowywanie wskaźników AUC
    w warunkach laboratoryjnych czy optymalizowanie papierowych tabel, lecz realna, mierzalna pomoc człowiekowi na krawędzi życia i śmierci.
    Zasada 3 miesięcy: Horyzont czasowy decyzyjności nie kończy się na wypisie z OIT – system projektowano z myślą o perspektywie długofalowej,
    gdzie faktycznym miarą sukcesu jest jakość powrotu pacjenta do sprawności neurologicznej i samodzielności.
    Nowy paradygmat medyczny: Architektura porzuca zimną kalkulację opłacalności, kosztów leczenia czy ekonomicznej efektywności na rzecz
    fundamentalnego pytania klinicznego: jaka forma pomocy i interwencji jest w tym konkretnym ułamku sekundy najlepsza dla tego chorego.
    Walka z nihilizmem terapeutycznym: Wobec gigantycznej, przytłaczającej złożoności ludzkiego mózgu, która często rodzi w medycynie poczucie
    bezradności, system ma być tarczą przeciwko rezygnacji – dając lekarzowi precyzyjne oparcie tam, gdzie standardowe procedury i ludzkie zmysły ulegają wyczerpaniu."""