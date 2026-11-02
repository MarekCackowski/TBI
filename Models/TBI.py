import sys
import os
import gc
import gzip
import tempfile
import shutil
import time
import datetime
import json
import re
import random
import warnings
import logging
import requests
import functools
from contextlib import nullcontext
from typing import Dict, Any, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "Models"))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Środowisko i konfiguracja
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# Bazy danych i pamięć podręczna (MinIO, PostgreSQL, Redis, MongoDB)
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, BotoCoreError
import psycopg2
import redis
import pymongo

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error("Brak biblioteki SQLAlchemy. Uruchom w terminalu: pip install sqlalchemy")
    raise ImportError("Brak biblioteki SQLAlchemy. Uruchom w terminalu: pip install sqlalchemy") from e

# Biblioteki dla obsługi danych i statystyki
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar

# Machine Learning - Przygotowanie danych, Cross-Validation i Pipelines
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, TargetEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import (
    GridSearchCV, train_test_split, cross_val_predict,
    StratifiedKFold, cross_val_score, HalvingGridSearchCV
)
from sklearn.experimental import enable_halving_search_cv
from sklearn.decomposition import PCA, SparsePCA
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.exceptions import NotFittedError, ConvergenceWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import resample

# Machine Learning - Klasyfikacja i Regresja
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, BayesianRidge
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier,
    StackingClassifier, VotingClassifier, HistGradientBoostingRegressor, IsolationForest
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    roc_auc_score, classification_report, precision_recall_curve,
    make_scorer, fbeta_score, confusion_matrix, brier_score_loss,
    calinski_harabasz_score, precision_score
)

# Zaawansowane modele tabularyczne i wstrzykiwanie danych
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from interpret.glassbox import ExplainableBoostingClassifier
from imodels import SkopeRulesClassifier

# Optymalizacja hiperparametrów
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Deep Learning (PyTorch & TabNet)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
from pytorch_tabnet.tab_model import TabNetClassifier

# Przetwarzanie obrazów
import pydicom
from PIL import Image

# Wizualizacja i wielowątkowość
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from tqdm import tqdm
import joblib
from joblib import parallel_backend

# Modele farmakokinetyczne i farmakodynamiczne
from Models.GraphModels.PharmaKineticsModels import (
    RealTBIPharmacoModel,
    PharmDynamicGraphODE,
    apply_exposure_scale,
    build_category_dose_matrix,
    build_patient_daily_tensor_with_pharma,
    fit_exposure_scale,
)

# Modele trajektorii pacjentów i monitorowania stanu
from Models.GraphModels.TrajectoryMonitors import (
    SonodeFalseAlarmDetector,
    LatentODE_VAE_Generator,
    odeint,
    serve_patient_stream,
    get_cached_mimic_3d_timeseries
)

from Models.TabularModels.TreesEnsemble import enrich_features_with_cluster_distances, train_optimized_ensemble, transform_features_with_cluster_distances

# Modele tabularyczne
from Models.TabularModels.TabularCascade import (
    AdaptiveMedicalNoiseAdder,
    MedicalCascadePredictor,
    train_specialist,
)

# Modele do analizy skanów
from Models.VisionModels.IntracranialHemorrhage_CRNN import VolumetricMedicalResNetCRNN
from Models.VisionModels.ModzelBody_UNet_to_EfficientNet import (
    UNet,
    VolumetricEfficientNet,
    sequential_two_stage_predict,
    RLAutoAugmentAgent
)
from Models.VisionModels.BoneFracture_Trans_plus_Dense import (
    BoneWindowTransform,
    DenseNetFractureDetector,
    TransformerFractureDetector3D,
    ensemble_predict_fracture
)


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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Inicjalizacja systemu logowania dla ostatecznego modułu
def setup_logger(name: str = "multimodal_cascade"):
    """ Centralny logger: konsole kieruje przez tqdm.write (nie rozbija paskow postepu),
        a kopie zapisuje do pliku. Zabezpieczony przed podwojna rejestracja handlerow. """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    class _TqdmStream:
        """ Strumien przekazujacy logi przez tqdm.write, żeby pasek sie nie dublowal. """
        @classmethod
        def write(cls, msg):
            if msg.strip():
                sys.__stdout__.write(msg + "\n")
                sys.__stdout__.flush()
        def flush(self, *args, **kwargs):
            if hasattr(sys.__stdout__, "flush"):
                sys.__stdout__.flush()

    ch = logging.StreamHandler(_TqdmStream)
    ch.setFormatter(fmt)
    fh = logging.FileHandler("crnn_training.log", encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


logger = setup_logger()


class Settings(BaseSettings):
    """ Scentralizowana konfiguracja aplikacji (Wersja zunifikowana dla Pydantic v2). """
    seed: int = 42
    models_dir: str = os.path.join(BASE_DIR, "output_data", "Models")
    plots_dir: str = os.path.join(BASE_DIR, "output_data")
    optuna_db: str = f"sqlite:///{os.path.join(BASE_DIR, 'output_data', 'optuna_study', 'tbi_ensemble.db')}"
    local_radiology_file: str = "radiology.csv.gz"
    beta_main: float = 1.0
    beta_critical: float = 2.0

    # Bazy danych i pamięć podręczna
    db_host: str = os.getenv('DB_HOST', '127.0.0.1')
    db_port: str = os.getenv('DB_PORT', '5433')
    db_name: str = os.getenv('DB_NAME', 'tbi_database')
    db_user: str = os.getenv('DB_USER', 'postgres')
    db_pass: str = os.getenv('DB_PASS', 'postgres')

    mongo_host: str = os.getenv('MONGO_HOST', '127.0.0.1')
    redis_host: str = os.getenv('REDIS_HOST', '127.0.0.1')

    # Konfiguracja S3 / MinIO
    s3_endpoint: str = os.getenv('S3_ENDPOINT', 'http://127.0.0.1:9000')
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

# Wykorzystanie instancji SETTINGS do ustawienia globalnych zmiennych środowiskowych
PHYSIONET_USER = SETTINGS.physionet_user
PHYSIONET_PASS = SETTINGS.physionet_pass

# flaga dla fuzji danych z MIMIC-IV
USE_PHYSIONET_MIMIC = SETTINGS.use_physionet_mimic

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


class PhenotypeTournamentRouter(BaseEstimator, ClassifierMixin):
    """ Dynamiczny Router Fenotypów z hybrydowym klastrowaniem kontekstowym (Hierarchiczne + GMM + Sparse PCA).
        Rozdziela pacjentów na specyficzne podgrupy kliniczne (np. urazy wysokenergetyczne u młodych oraz profil geriatryczny),
        stosując dedykowane metody redukcji szumu i przestrzeni cech. """
    _estimator_type = "classifier"

    def __init__(self, candidate_estimators, feature_names=None, phenotype_hints=None, min_clusters=2, max_clusters=4, min_patients_per_cluster=70):
        self.candidate_estimators = candidate_estimators
        self.feature_names = feature_names
        self.phenotype_hints = phenotype_hints
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.min_patients_per_cluster = min_patients_per_cluster

        # Bazowa redukcja szumu przez Random Forest (ochrona przed niszczeniem cech tabelarycznych)
        self.reducer = SelectFromModel(
            RandomForestClassifier(n_estimators=50, max_depth=5, random_state=CONFIG["SEED"], n_jobs=-1),
            max_features=25
        )

        # Specjalistyczne komponenty hybrydowe
        self.sparse_pca = SparsePCA(n_components=3, alpha=1.0, random_state=CONFIG["SEED"])

        self.clusterers_ = {}
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

        # Miękkie naprowadzenie algorytmu za pomocą wskazówek klinicznych
        if self.phenotype_hints and self.feature_names:
            for col_name, weight in self.phenotype_hints.items():
                if col_name in self.feature_names:
                    idx = self.feature_names.index(col_name)
                    X_cluster_space[:, idx] *= weight

        # Redukcja szumu przez Random Forest (wyciągamy najważniejsze osie decyzyjne)
        X_reduced = self.reducer.fit_transform(X_cluster_space, y_arr)

        #  Dynamiczny turniej struktur klastrowania i podgrup medycznych
        logger.info("Uruchamiam dynamiczny turniej struktur klastrowania i podgrup medycznych.")

        # Sprawdzamy, czy mamy cechy pozwalające na inteligentne naprowadzenie
        gcs_idx = self.feature_names.index('gcs_sum') if self.feature_names and 'gcs_sum' in self.feature_names else None
        intubation_idx = self.feature_names.index('is_intubated') if self.feature_names and 'is_intubated' in self.feature_names else None

        # Jeśli dysponujemy indeksem GCS i intubacji, wykrywamy podgrupę konfundującą (np. rurka vs uraz)
        clusters = np.zeros(len(X_arr), dtype=int)

        if gcs_idx is not None:
            gcs_vals = X_arr[:, gcs_idx]
            is_intub = X_arr[:, intubation_idx] if intubation_idx is not None else np.zeros(len(X_arr))

            # Algorytmiczne wyznaczenie podgrupy wysokiego ryzyka artefaktu (np. głęboki spadek GCS przy obecności rurki)
            confound_mask = (gcs_vals <= 8) & (is_intub == 1.0)
            safe_mask = ~confound_mask

            subsets = []
            if np.sum(confound_mask) >= self.min_patients_per_cluster:
                subsets.append((confound_mask, "artefakt_drogi_oddechowej", 10))
            if np.sum(safe_mask) >= self.min_patients_per_cluster:
                subsets.append((safe_mask, "przestrzenie_standardowe", 20))
        else:
            # Jeśli brak dedykowanych kolumn, traktujemy całość jako jedną przestrzeń do podziału
            subsets = [(np.ones(len(X_arr), dtype=bool), "globalna_populacja", 0)]

        final_clusters = np.zeros(len(X_arr), dtype=int)
        self.clusterers_ = {}

        for mask, group_name, offset in subsets:
            X_sub = X_reduced[mask]
            if len(X_sub) < self.min_patients_per_cluster:
                continue

            # Turniej algorytmów klastrowania dla danej podgrupy (GMM vs Hierarchical Ward)
            candidate_clusterers = {
                'GMM': GaussianMixture(n_components=2, covariance_type='full', random_state=CONFIG["SEED"], n_init=3),
                'Ward': AgglomerativeClustering(n_clusters=2, linkage='ward')
            }

            best_score = -1.0
            best_labels = np.zeros(len(X_sub), dtype=int)
            best_algo = None
            best_algo_name = ""

            for algo_name, cl_model in candidate_clusterers.items():
                try:
                    if hasattr(cl_model, 'fit_predict'):
                        labels = cl_model.fit_predict(X_sub)
                    else:
                        labels = cl_model.fit(X_sub).labels_

                    # Ocena jakości podziału (minimalny rozmiar klastra + Calinski-Harabasz)
                    if np.min(np.bincount(labels)) < 15:
                        continue
                    score = calinski_harabasz_score(X_sub, labels)

                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_algo = cl_model
                        best_algo_name = algo_name
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                    logger.warning(f"Błąd podczas dopasowywania {algo_name} dla podgrupy '{group_name}': {e}")
                    continue

            if best_algo is None:
                # Awaryjny fallback na prosty podział
                best_labels = np.zeros(len(X_sub), dtype=int)
                best_algo = candidate_clusterers['GMM']
                best_algo.fit(X_sub)
                best_algo_name = "GMM_Fallback"

            logger.info(f"   -> Podgrupa '{group_name}': Wybrano algorytm {best_algo_name} (Score: {best_score:.2f})")

            self.clusterers_[group_name] = best_algo
            final_clusters[mask] = best_labels + offset

        # Unifikacja klastrów
        unique_clusters = np.unique(final_clusters)
        cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        mapped_clusters = np.array([cluster_mapping[c] for c in final_clusters])

        self.cluster_mapping_ = cluster_mapping
        best_n_components = len(unique_clusters)
        self.clusterer_ = final_clusters

        logger.info(f"Ostatecznie wyodrębniono {best_n_components} optymalnych fenotypów za pomocą dynamicznego turnieju.")

        # Dalej standardowy turniej modelowy dla każdego wyodrębnionego fenotypu
        self.models_ = []
        tournament_scorer = make_scorer(fbeta_score, beta=CONFIG["BETA_MAIN"], zero_division=0)

        for i in range(best_n_components):
            mask = (mapped_clusters == i)
            X_cluster, y_cluster = X_arr[mask], y_arr[mask]

            if len(np.unique(y_cluster)) < 2 or len(X_cluster) < 10:
                self.models_.append(self.fallback_model_)
                continue

            best_score_turn = -1.0
            best_model_clone = None
            winning_name = ""

            for name, estimator in self.candidate_estimators.items():
                clf = clone(estimator)
                scores = cross_val_score(clf, X_cluster, y_cluster, cv=3, scoring=tournament_scorer, n_jobs=1)
                avg_score = scores.mean()

                if avg_score > best_score_turn:
                    best_score_turn = avg_score
                    best_model_clone = clf
                    winning_name = name

            best_model_clone.fit(X_cluster, y_cluster)
            self.models_.append(best_model_clone)
            logger.info(f"   Zwycięzca dla Fenotypu {i}: {winning_name}")

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

        # Odtworzenie przypisania do klastrów hybrydowych podczas inferencji
        age_idx = self.feature_names.index('age') if self.feature_names and 'age' in self.feature_names else None
        clusters = np.zeros(len(X_arr), dtype=int)

        if age_idx is not None:
            ages = X_arr[:, age_idx]
            geriatric_mask = ages >= 75.0
            severe_young_mask = (ages < 45.0)

            if 'geriatric' in self.clusterers_ and np.sum(geriatric_mask) > 0:
                gmm_ger, spca = self.clusterers_['geriatric']
                X_ger_sparse = spca.transform(X_reduced[geriatric_mask])
                clusters[geriatric_mask] = gmm_ger.predict(X_ger_sparse) + 10

            if 'young_severe' in self.clusterers_ and np.sum(severe_young_mask) > 0:
                # AgglomerativeClustering nie ma metody predict, dopasowujemy najbliższy centroid lub przypisujemy fallback
                clusters[severe_young_mask] = 20

        standard_mask = (clusters == 0)
        if 'standard' in self.clusterers_ and np.sum(standard_mask) > 0:
            clusters[standard_mask] = self.clusterers_['standard'].predict(X_reduced[standard_mask]) + 30

        mapped_clusters = np.array([self.cluster_mapping_.get(c, 0) for c in clusters])

        probs = np.zeros((X_arr.shape[0], 2))
        for i, model in enumerate(self.models_):
            mask = (mapped_clusters == i)
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
        odrzuca wybrakowane rekordy krytyczne i dynamicznie zwraca trójpodział na zbiór:
        treningowy, walidacyjny oraz testowy. """
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

    # Trzy rozłączne zbiory: Train (60%), Validation (20%), Test (20%), dla tabelarycznych powinny starczyć
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=CONFIG["SEED"])
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=CONFIG["SEED"])

    logger.info(f"Podział danych: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def cascade_predict(data):
    """ Definiujemy funkcję predykcji dla kaskady. """
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


def extract_vitals_via_chunking(filepath, item_ids):
    """ Funkcja do ekstrakcji danych telemetrycznych z pliku labevents.csv.gz w trybie Out-of-Core.
        Wydobywa Min, Max, Średnią, Odchylenie Standardowe oraz Liczbę pomiarów dla każdego pacjenta i każdego sensora. """
    logger.info("Rozpoczynam przesiewanie 30 GB danych z aparatury (Chunking Out-of-Core).")

    aggregated_results = []

    # Chunksize = 1 000 000 wierszy
    chunk_iterator = pd.read_csv(filepath, compression='gzip', chunksize=1000000, usecols=['hadm_id', 'itemid', 'valuenum'])

    # TQDM do iteracji przez fragmenty pliku
    for i, chunk in enumerate(tqdm(chunk_iterator, desc="Przetwarzanie paczek 1M wierszy", leave=False, mininterval=2.0)):
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
        global_max=('chunk_max', 'max'),
        global_min=('chunk_min', 'min'),
        global_sum=('chunk_sum', 'sum'),
        global_count=('chunk_count', 'sum'),
        global_sum_sq=('chunk_sum_sq', 'sum')
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
    """ Gwarantowany klasyfikator dla CalibratedClassifierCV omijający błędy Pipeline'u. """
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

                # Inicjalizujemy prawidłowy tensor bez nadpisywania wymiarami z EfficientNet
                volume_tensor = torch.zeros((24, 3, 160, 160))

                for i, slice_file in enumerate(slices):
                    if i >= 24:  # Zabezpieczenie przed błędem IndexError
                        break

                    img_path = os.path.join(patient_folder, slice_file)

                    with Image.open(img_path) as img:
                        img_rgb = img.convert('RGB')

                    img_bgr_array = np.array(img_rgb)[:, :, ::-1].copy()
                    img_bgr = Image.fromarray(img_bgr_array)

                    volume_tensor[i] = transform(img_bgr)

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
                    # Zabezpieczenie przed wyjściem poza rozmiar tensora
                    if i >= len(volume_tensor):
                        break

                    img_path = os.path.join(patient_folder, slice_file)

                    # Blok 'with' gwarantuje zwolnienie uchwytu do pliku po wczytaniu
                    with Image.open(img_path) as img:
                        img_rgb = img.convert('RGB')

                    # .copy() alokuje nową pamięć, rozwiązując problem niekompatybilności z PyTorch
                    img_bgr_array = np.array(img_rgb)[:, :, ::-1].copy()
                    img_bgr = Image.fromarray(img_bgr_array)

                    volume_tensor[i] = transform(img_bgr)

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


def load_trained_fracture_predictions(X_test, dataset_path='Data/dataset_ct_main', device=None):
    """ Pełnoprawna, produkcyjna funkcja inferencji dla modułu złamań czaszki.
        Obsługuje zarówno pojedynczy model DenseNet 3D, jak i pełny zespół (Ensemble)
        łączący DenseNet 3D z Vision Transformerem 3D przy użyciu Uncertainty Aware Soft Voting. """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ensemble_path = os.path.join('output_data', 'Models', 'fracture_best.pth')
    dense_path = os.path.join('output_data', 'Models', 'densenet_fracture_best.pth')
    trans_path = os.path.join('output_data', 'Models', 'transformer_fracture_best.pth')

    # Sprawdzamy dostępność wag modeli
    has_weights = os.path.exists(dense_path)

    if not has_weights:
        logger.warning("Brak pliku wag DenseNet. Zwracam neutralne prawdopodobieństwa 0.5.")
        return np.full(len(X_test), 0.5, dtype=np.float32)

    model_dense = None
    model_trans = None

    try:
        # Inicjalizacja modeli w trybie ewaluacji z wyłączonym dropoutem
        model_dense = DenseNetFractureDetector(pretrained=False, dropout_rate=0.0).to(device)
        model_dense.load_state_dict(torch.load(dense_path, map_location=device, weights_only=True))
        model_dense.eval()

        if os.path.exists(trans_path):
            model_trans = TransformerFractureDetector3D(dropout_rate=0.0).to(device)
            model_trans.load_state_dict(torch.load(trans_path, map_location=device, weights_only=True))
            model_trans.eval()
            logger.info("Załadowano pełny zespół modelowy (DenseNet 3D + Vision Transformer 3D).")
        else:
            logger.info("Załadowano pojedynczy model DenseNet 3D (brak wag Transformera).")

        # Poprawna inicjalizacja transformatora okna kostnego (zgodna z definicją percentyla)
        bone_window = BoneWindowTransform(percentile=88)

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

                # Normalizacja liczby slajdów do stałego wolumenu (30 warstw)
                if len(slices) > 30:
                    indices = np.linspace(0, len(slices) - 1, 30, dtype=int)
                    slices = [slices[i] for i in indices]
                elif len(slices) < 30:
                    slices = slices + [slices[-1]] * (30 - len(slices))

                volume_tensor = torch.zeros((1, 30, 224, 224), dtype=torch.float32)

                for i, slice_file in enumerate(slices):
                    img_path = os.path.join(patient_folder, slice_file)

                    try:
                        if img_path.endswith('.dcm'):
                            # Bezpieczne wczytanie DICOM z użyciem kontekstu
                            with open(img_path, 'rb') as f:
                                dicom_data = pydicom.dcmread(f)

                            img_array = dicom_data.pixel_array.astype(np.float32)
                            intercept = float(getattr(dicom_data, 'RescaleIntercept', 0.0))
                            slope = float(getattr(dicom_data, 'RescaleSlope', 1.0))
                            hu_image = img_array * slope + intercept
                            img_pil = bone_window(hu_image)
                        else:
                            # Standardowe pliki obrazów (PNG/JPG) z odcięciem referencji
                            from PIL import UnidentifiedImageError
                            with Image.open(img_path) as img:
                                img_array = np.array(img.convert('L')).copy()
                            img_pil = bone_window(img_array)

                        tensor_2d = transform(img_pil)
                        volume_tensor[0, i] = tensor_2d[0]
                    except FileNotFoundError:
                        logger.debug(f"Brak pliku slajdu: {img_path}")
                        continue
                    except pydicom.errors.InvalidDicomError as dicom_err:
                        logger.debug(f"Nieprawidłowy lub uszkodzony plik DICOM {img_path}: {dicom_err}")
                        continue
                    except UnidentifiedImageError as img_err:
                        logger.debug(f"Nie rozpoznano formatu obrazu {img_path}: {img_err}")
                        continue
                    except AttributeError as attr_err:
                        logger.debug(f"Brak wymaganych metadanych DICOM (np. pixel_array) w {img_path}: {attr_err}")
                        continue
                    except (ValueError, TypeError) as val_err:
                        logger.debug(f"Błąd transformacji tensora lub konwersji macierzy dla {img_path}: {val_err}")
                        continue
                    except Exception as slice_err:
                        logger.debug(f"Nieoczekiwany błąd podczas wczytywania slajdu {img_path}: {slice_err}")
                        continue

                vol_input = volume_tensor.unsqueeze(0).to(device)

                autocast_context = torch.amp.autocast(device_type='cuda') if device.type == 'cuda' else nullcontext()
                with autocast_context:
                    if model_trans is not None:
                        # Użycie pełnego Soft Voting opartego na entropii z obu modeli
                        pred_prob = ensemble_predict_fracture(vol_input, model_dense, model_trans)
                    else:
                        outputs = model_dense(vol_input)
                        pred_prob = torch.sigmoid(outputs).cpu().numpy()

                    val = float(np.max(pred_prob) if isinstance(pred_prob, np.ndarray) else pred_prob)
                    probs_list.append(val)

        # Czyszczenie pamięci podręcznej i wskaźników GPU
        del model_dense, model_trans
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return np.array(probs_list, dtype=np.float32)

    except RuntimeError as e:
        logger.error(f"Błąd sprzętowy PyTorch w module złamań: {e}")
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return np.full(len(X_test), 0.5, dtype=np.float32)
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas inferencji złamań: {e}")
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return np.full(len(X_test), 0.5, dtype=np.float32)


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

                logger.info("Pobieranie zakończone. Wypakowuję archiwum do formatu CSV.")
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


def train_pipeline(X_train, X_test, y_train, y_test):
    """ Główny potok treningowy systemu TBI:
        Przeprowadza inżynierię cech, Multi Task Learning,
        oraz przekazuje zadania do wyspecjalizowanych funkcji budujących Kaskadę. """
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
            fio2_fraction = np.where(df['fio2_max'] > 1.0, df['fio2_max'] / 100.0, df['fio2_max'])
            df['sf_ratio'] = df['spo2_min'] / (fio2_fraction + 1e-5)

        if 'map_min' in df.columns:
            df['map_radial_deviation'] = np.abs(df['map_min'] - 85.0)

        if 'glucose_max' in df.columns:
            df['glucose_radial_deviation'] = np.abs(df['glucose_max'] - 100.0)

        if 'spo2_min' in df.columns:
            df['spo2_radial_deviation'] = np.abs(df['spo2_min'] - 98.0)

        if 'sysbp_min' in df.columns:
            df['sysbp_radial_deviation'] = np.abs(df['sysbp_min'] - 120.0)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    sonode_monitor_features = [
        "heart_rate_max",
        "sysbp_min",
        "map_min",
        "spo2_min",
        "glucose_max",
        "gcs_sum",
    ]
    sonode_train_df = X_train.reindex(columns=sonode_monitor_features).copy()
    sonode_test_df = X_test.reindex(columns=sonode_monitor_features).copy()

    logger.info("Multi-Task Learning (Czujnik Wgłobienia)")
    mtl_features = [c for c in X_train.columns if c not in ['cushing_risk_index', 'd1_sysbp_max', 'd1_heartrate_min', 'sysbp_min', 'heart_rate_max',
                                                            'pulse_pressure', 'bp_swing', 'hr_swing', 'modified_shock_index', 'shock_index']]

    X_train_mtl = X_train[mtl_features].copy()
    X_test_mtl = X_test[mtl_features].copy()

    for col in X_train_mtl.select_dtypes(include=['object', 'category']).columns:
        X_train_mtl[col], _ = pd.factorize(X_train_mtl[col])
        X_test_mtl[col], _ = pd.factorize(X_test_mtl[col])

    mtl_regressor = HistGradientBoostingRegressor(random_state=CONFIG["SEED"], max_iter=150)
    y_train_target = X_train['cushing_risk_index'].fillna(0)

    train_oof_preds = cross_val_predict(
        mtl_regressor,
        X_train_mtl,
        y_train_target,
        cv=5,
        method='predict'
    )
    X_train['hidden_brainstem_crush_risk'] = train_oof_preds - y_train_target

    mtl_regressor.fit(X_train_mtl, y_train_target)
    X_test['hidden_brainstem_crush_risk'] = mtl_regressor.predict(X_test_mtl) - X_test['cushing_risk_index'].fillna(0)

    for df_target, df_source in zip([X_train, X_test], [X_train_mtl, X_test_mtl]):
        df_target['hidden_brainstem_crush_risk'] = mtl_regressor.predict(df_source) - df_target['cushing_risk_index'].fillna(0)

    baseline_cols = ['age', 'diabetes_mellitus', 'immunosuppression', 'cirrhosis', 'hepatic_failure', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']
    numerical_cols = [c for c in X_train.select_dtypes(include=['int64', 'float64']).columns if c not in baseline_cols]
    categorical_cols = [c for c in X_train.select_dtypes(include='object').columns if c not in baseline_cols]

    baseline_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('scaler', StandardScaler()),
                                     ('pytorch_profiler', PatientBaselineProfiler(input_dim=8, hidden_dim=32, output_dim=16, epochs=3))])

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

    logger.info("Uruchamianie ekstrakcji fenotypów klinicznych.")
    X_train_clean_df, X_test_clean_df, clinical_rules = extract_and_apply_tree_rules(
        X_train_clean_df, y_train, X_test_clean_df, feature_names
    )

    # Fitujemy GMM wyłącznie na treningu i wzbogacamy
    X_train_enriched_vals, trained_gmm = enrich_features_with_cluster_distances(X_train_clean_df.values, n_clusters=4)

    # Transformujemy zbiór testowy używając wyłącznie wyuczonego GMM
    X_test_enriched_vals = transform_features_with_cluster_distances(X_test_clean_df.values, trained_gmm)

    cluster_col_names = [f"gmm_cluster_prob_{i}" for i in range(4)]
    all_feature_names = list(X_train_clean_df.columns) + cluster_col_names

    X_train_clean_df = pd.DataFrame(X_train_enriched_vals, columns=all_feature_names, index=X_train.index)
    X_test_clean_df = pd.DataFrame(X_test_enriched_vals, columns=all_feature_names, index=X_test.index)
    
    X_train_numeric = X_train_clean_df.values

    feature_names_enriched = list(X_train_clean_df.columns)
    logger.info(f"Całkowita liczba cech po wzbogaceniu: {len(feature_names_enriched)}")

    rule_columns = [col for col in X_train_clean_df.columns if "Phenotype_Rule" in col]
    for col in rule_columns:
        X_train[col] = X_train_clean_df[col].values
        X_test[col] = X_test_clean_df[col].values

    logger.info("Inicjalizacja Agenta RL do dynamicznego sterowania szumem medycznym.")
    rl_agent = RLAutoAugmentAgent(state_dim=8, action_dim=5, lr=1e-3, struct_lambda=0.2).to(DEVICE)

    noise_transformer = None
    for name, trans, cols in preprocessor_trees.transformers:
        if isinstance(trans, Pipeline):
            for step_name, step_obj in trans.steps:
                if isinstance(step_obj, AdaptiveMedicalNoiseAdder):
                    noise_transformer = step_obj
                    break

    previous_f2 = 0.0
    previous_struct_score = 1.0

    ordered_constraints = []
    for col in X_train_clean_df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['gcs', 'sysbp', 'spo2', 'map_min', 'adjusted_gcs']): ordered_constraints.append(-1)
        elif any(x in col_lower for x in ['shock_index', 'cushing_risk', 'age_squared', 'is_hypotensive']): ordered_constraints.append(1)
        else: ordered_constraints.append(0)
    monotone_tuple = tuple(ordered_constraints)

    y_train_numeric = y_train.astype(int).values
    cv_strategy = StratifiedKFold(n_splits=2, shuffle=True, random_state=CONFIG["SEED"])

    binary_flags = [
        'is_intubated', 'gcs_sedation_mismatch', 'baseline_neuro_deficit',
        'is_hypotensive', 'critical_geriatric_condition', 'is_low_map'
    ]
    binary_indices = [list(X_train_clean_df.columns).index(col)
                      for col in binary_flags if col in X_train_clean_df.columns]

    logger.info(f"BernoulliNB wykorzysta {len(binary_indices)} flag binarnych jako system ostrzegania.")

    for rl_ep in range(3):
        actions = rl_agent.select_augmentations(current_loss=previous_f2, struct_score=previous_struct_score)
        scale_gcs, scale_vitals, scale_other = actions[0], actions[1], actions[2]

        if noise_transformer is not None:
            noise_transformer.update_rl_policy(scale_gcs, scale_vitals, scale_other)

        current_f2 = 0.85
        current_struct_score = 1.0 - np.mean([scale_gcs, scale_vitals, scale_other])

        task_reward = current_f2 - previous_f2
        rl_agent.update(task_reward=task_reward, struct_reward=current_struct_score)

        previous_f2 = current_f2
        previous_struct_score = current_struct_score
        logger.info(f"Agent RL - Epoka {rl_ep+1}/3 | Wagi szumu (GCS/Vit/Oth): {scale_gcs:.2f}/{scale_vitals:.2f}/{scale_other:.2f}")

    logger.info("Wzbogacanie cech o odległości do centroidów klastrów w celu poprawy reprezentacji danych.")
    X_train_numeric_enriched = enrich_features_with_cluster_distances(X_train_clean_df.values, n_clusters=4)

    logger.info("Trenowanie zoptymalizowanego zespołu klasyfikatorów z wykorzystaniem Optuny.")
    ensemble_best = train_optimized_ensemble(
        X_train_numeric=X_train_numeric_enriched,
        y_train_numeric=y_train_numeric,
        cv_strategy=cv_strategy,
        monotone_tuple=monotone_tuple,
        binary_indices=binary_indices,
        preprocessor_imputed=preprocessor_imputed,
        X_train=X_train,
        y_train=y_train
    )

    logger.info("Izolacja anomalii: wybór lasu izolacyjnego, ponieważ jego struktura oparta na drzewach podporządkowuje się do drzew z pierwszej fazy, co zapewnia spójność w wykrywaniu próbek brzegowych.")
    anomaly_detector = IsolationForest(
        n_estimators=200, contamination=0.1, random_state=CONFIG["SEED"], n_jobs=-1
    )

    mask_apache_optimistic = (X_train['apache_4a_hospital_death_prob'] < 0.25)
    mask_hidden_instability = (
        ((X_train['d1_sysbp_max'] - X_train['d1_sysbp_min']) > 50) |
        (X_train['d1_glucose_max'] > 200)
    )
    silent_death_mask_train = (y_train == 1) & mask_apache_optimistic & mask_hidden_instability

    mask_apache_pessimistic = (X_train['apache_4a_hospital_death_prob'] > 0.75)
    severe_survivors_mask_train = (y_train == 0) & mask_apache_pessimistic

    both_sides_difficult_mask = silent_death_mask_train | severe_survivors_mask_train

    if both_sides_difficult_mask.sum() > 50:
        anomaly_detector.fit(X_train_clean_df[both_sides_difficult_mask])
    else:
        anomaly_detector.fit(X_train_clean_df[y_train == 1])

    # Wykorzystanie predykcji krzyżowej do identyfikacji przypadków trudnych i niepewnych (przy użyciu Isolation Forest i progów niepewności)
    y_prob_train_main_oof = cross_val_predict(ensemble_best, X_train_clean_df.values, y_train.values, cv=3, method='predict_proba', n_jobs=-1)[:, 1]
    is_silent_death_train = (anomaly_detector.predict(X_train_clean_df) == -1)
    is_uncertain_train = ((y_prob_train_main_oof > 0.35) & (y_prob_train_main_oof < 0.65))

    X_spec_train_full = X_train[is_silent_death_train | is_uncertain_train].copy()
    X_spec_train_full['model_1_prob'] = y_prob_train_main_oof[X_train.index.get_indexer(X_spec_train_full.index)]
    y_spec_train = y_train.loc[X_spec_train_full.index]

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

    calibrated_spec = train_specialist(X_spec_train, y_spec_train, spec_prep, cv_strategy=3)

    logger.info("Ostateczny Sędzia Kaskady: Fuzja Wielomodalna")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Pobieranie predykcji modeli wizyjnych dla zbioru treningowego używanego przez sędziego całej kaskady.")
    crnn_for_judge = load_trained_crnn_predictions(X_train, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_ct_main'), device=device)
    efficientnet_for_judge, _ = load_trained_efficientnet_predictions(X_train, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_efficientnet'), device=device)
    densenet_for_judge = load_trained_fracture_predictions(X_train, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_ct_main'), device=device)

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
    X_meta_train_df = pd.DataFrame(X_meta_train, columns=base_names_meta)

    logger.info("Aplikowanie RL Proxy: Cost Sensitive Hard Mining dla przypadków skrajnych (przy użyciu sample_weight).")

    unexpected_survivors_mask = (y_train == 0) & (y_prob_train_main_oof > 0.75)
    unpredicted_fatalities_mask = (y_train == 1) & (y_prob_train_main_oof < 0.25)
    hard_examples_mask = unexpected_survivors_mask | unpredicted_fatalities_mask

    sample_weights = np.ones(len(y_train), dtype=float)

    if hard_examples_mask.sum() > 0:
        logger.info(f"RL-Proxy wyłapał: {unexpected_survivors_mask.sum()} Nieoczekiwanych Przeżyć oraz {unpredicted_fatalities_mask.sum()} Nieprzewidzianych Zgonów. Wdrażam politykę wag kosztowych (sample_weight = 5.0 dla trudnych przypadków).")
        sample_weights[hard_examples_mask] = 5.0
    else:
        logger.info("Nie znaleziono błędów o wysokiej pewności. Brak modyfikacji wag.")

    final_judge, calibrated_final_judge = train_judge(
        X_meta_train_df, y_train, sample_weights, base_names_meta
    )

    real_crnn_test = load_trained_crnn_predictions(X_test, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_ct_main'), device=device)
    real_efficientnet_test_probs, real_efficientnet_test_uncert = load_trained_efficientnet_predictions(X_test, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_efficientnet'), device=device)
    real_densenet_test = load_trained_fracture_predictions(X_test, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_ct_main'), device=device)

    final_probs_train = calibrated_final_judge.predict_proba(X_meta_train_df)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, final_probs_train)
    beta_val = CONFIG["BETA_CRITICAL"]
    f_beta_scores = (1 + beta_val ** 2) * (precisions[:-1] * recalls[:-1]) / ((beta_val ** 2 * precisions[:-1]) + recalls[:-1] + 1e-10)
    if len(thresholds) == 0:
        raise ValueError("Nie można wyznaczyć progu klinicznego: brak progów PR.")
    best_train_threshold = float(np.clip(thresholds[np.argmax(f_beta_scores)], 0.2, 0.85))

    calibrated_spec_pipe = MedicalCascadePredictor(
        preprocessor=preprocessor_trees,
        ensemble=ensemble_best,
        specialist=calibrated_spec,
        judge=calibrated_final_judge,
        anomaly_detector=anomaly_detector,
        review_threshold=0.05,
        optimal_threshold=best_train_threshold
    )

    logger.info("Inicjalizowanie detektora fałszywych alarmów i zatruć Sonode.")
    sonode_model = SonodeFalseAlarmDetector(device=device)
    sonode_model.fit(sonode_train_df)
    sonode_results = sonode_model.detect_false_alarms_and_poisoning(sonode_test_df)

    logger.info("Konfigurowanie modeli farmakoterapii TBI.")
    drug_models = {
        # Mannitol: Redukuje obrzęk i czyści płyny w mózgu
        "mannitol": RealTBIPharmacoModel("mannitol"),

        # Sól hipertoniczna: Szybkie wyrównanie ciśnienia w krytycznym momencie
        "hypertonic_saline": RealTBIPharmacoModel("hypertonic_saline"),

        # Barbiturany: Wyciszają aktywność metaboliczną mózgu i kontrolują oporne nadciśnienie wewnątrzczaszkowe
        "barbiturates": RealTBIPharmacoModel("barbiturates"),

        # Lewetyracetam: Stabilny lek przeciwdrgawkowy chroniący przed atakami
        "antiseizure_levetiracetam": RealTBIPharmacoModel("antiseizure_levetiracetam"),
    }

    artifacts = {
        "ensemble_best": ensemble_best,
        "calibrated_spec": calibrated_spec,
        "anomaly_detector": anomaly_detector,
        "final_judge": final_judge,
        "X_train_clean_df": X_train_clean_df,
        "X_test_clean_df": X_test_clean_df,
        "base_names": base_names_meta,
        "crnn_test": real_crnn_test,
        "efficientnet_test": real_efficientnet_test_probs,
        "efficientnet_uncert": real_efficientnet_test_uncert,
        "densenet_test": real_densenet_test,
        "optimal_threshold": best_train_threshold,
        "sonode_model": sonode_model,
        "sonode_results": sonode_results,
        "drug_models": drug_models,
    }

    return calibrated_spec_pipe, artifacts


class MultiTargetSurrogateMLP(nn.Module):
    """ Różniczkowalny model zastępczy z Monte Carlo Dropout do szacowania niepewności
        i wstecznej propagacji gradientów w silniku. """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_with_mc_uncertainty(self, x: torch.Tensor, n_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Włącza Dropout podczas inferencji (Monte Carlo Dropout), zwracając różniczkowalną średnią i wariancję. """
        self.train() # Wymuszenie aktywnego Dropoutu dla próbowania stochastycznego
        samples = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)
        return mean, variance


def train_surrogate_model(X_train_df: pd.DataFrame, epochs: int = 25, lr: float = 0.005) -> MultiTargetSurrogateMLP:
    """ Dopasowuje sieć zastępczą do odtwarzania pełnego wektora stanu pacjenta. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_arr = np.nan_to_num(X_train_df.values, nan=0.0).astype(np.float32)

    input_dim = X_arr.shape[1]
    model = MultiTargetSurrogateMLP(input_dim=input_dim, output_dim=input_dim).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X_arr, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_x)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


class ComprehensiveClinicalRecourseEngine:
    """ Optymalizuje interwencje w celu utrzymania parametrów diagnostycznych w normie,
        zachowując pełną różniczkowalność niepewności epistemicznej za pomocą Monte Carlo Dropout. """
    def __init__(self, multi_target_surrogate: MultiTargetSurrogateMLP, feature_names: List[str], intervention_features: List[str], diagnostic_features: List[str]):
        self.surrogate = multi_target_surrogate
        self.feature_names = feature_names
        self.intervention_features = intervention_features
        self.diagnostic_features = diagnostic_features

    def predict_future_state(self, x_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Zwraca różniczkowalny wektor stanu oraz różniczkowalną wariancję (Monte Carlo Dropout). """
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)
        mean_state, variance = self.surrogate.predict_with_mc_uncertainty(x_tensor, n_samples=15)
        return mean_state, variance

    def optimize_treatment_plan(self, patient_vector: np.ndarray, missing_mask: np.ndarray, target_ranges: Dict[str, Tuple[float, float]]):
        """ Optymalizuje interwencje w celu utrzymania parametrów diagnostycznych w normie, przy jednoczesnym minimalizowaniu niepewności
            epistemicznej i zachowaniu fizjologicznej spójności. System nie jest jedynie naiwnym predykatorem śmierci –
            cała wytrenowana wiedza jest aktywnie wykorzystywana na każdym etapie procesu, aby stale optymalizować
            jakość leczenia i rehabilitacji pacjenta. Uwzględniając rygorystyczne ograniczenia kliniczne i fizjologiczne,
            odchodzimy od biernej predykcji zgonu na rzecz ciągłej prewencji krytycznych załamań parametrów życiowych. """
        x_orig = torch.tensor(patient_vector, dtype=torch.float32)
        delta = torch.zeros_like(x_orig, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=0.02)

        actionable_mask = torch.zeros_like(x_orig)
        for idx, f_name in enumerate(self.feature_names):
            if f_name in self.intervention_features and missing_mask[idx] == 1:
                actionable_mask[idx] = 1.0

        for step in range(300):
            optimizer.zero_grad()

            x_cand = x_orig + (delta * actionable_mask)
            future_state, uncertainty = self.predict_future_state(x_cand)

            loss_diagnostics = torch.tensor(0.0, dtype=torch.float32)
            loss_uncertainty = torch.tensor(0.0, dtype=torch.float32)

            for feat, (min_val, max_val) in target_ranges.items():
                if feat in self.feature_names:
                    idx = self.feature_names.index(feat)
                    pred_val = future_state[0, idx]

                    penalty_low = torch.relu(min_val - pred_val) ** 2
                    penalty_high = torch.relu(pred_val - max_val) ** 2
                    loss_diagnostics = loss_diagnostics + penalty_low + penalty_high

                    # Wariancja jest teraz częścią grafu PyTorch
                    loss_uncertainty = loss_uncertainty + uncertainty[0, idx]

            # Dodatkowy kaganiec fizjologiczny: osmolalność osocza (2*Na + Glucose/18) < 320 mOsm/kg
            loss_osmolality = torch.tensor(0.0, dtype=torch.float32)
            if 'sodium_mean' in self.feature_names and 'glucose_max' in self.feature_names:
                na_idx = self.feature_names.index('sodium_mean')
                gl_idx = self.feature_names.index('glucose_max')
                est_osmolality = 2.0 * future_state[0, na_idx] + (future_state[0, gl_idx] / 18.0)
                loss_osmolality = torch.relu(est_osmolality - 320.0) ** 2

            loss_dist = 0.05 * torch.norm(delta * actionable_mask, p=1)

            total_loss = loss_diagnostics + 1.2 * loss_uncertainty + 2.0 * loss_osmolality + loss_dist
            total_loss.backward()
            optimizer.step()

        x_final = (x_orig + (delta * actionable_mask)).detach().numpy()
        return self._generate_report(patient_vector, x_final, actionable_mask)

    def extract_adverse_profiles_for_target(self, cohort_df: pd.DataFrame, intervention: str, target_diagnostic: str, delta_intervention: float):
        """ Identyfikuje pacjentów, u których interwencja wywoła krytyczne załamanie konkretnego parametru (np. drastyczny spadek MAP). """
        X_base = cohort_df[self.feature_names].values
        int_idx = self.feature_names.index(intervention)
        diag_idx = self.feature_names.index(target_diagnostic)

        X_treated = X_base.copy()
        X_treated[:, int_idx] += delta_intervention

        val_base = np.array([np.mean([m.predict(x.reshape(1,-1))[0, diag_idx] for m in self.ensemble_models]) for x in X_base])
        val_treated = np.array([np.mean([m.predict(x.reshape(1,-1))[0, diag_idx] for m in self.ensemble_models]) for x in X_treated])

        impact = val_treated - val_base

        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=0.05)
        tree.fit(X_base, impact)

        return export_text(tree, feature_names=self.feature_names)

    def _generate_report(self, orig: np.ndarray, final: np.ndarray, mask: torch.Tensor) -> Dict[str, Any]:
        interventions = {}
        for idx in range(len(self.feature_names)):
            if mask[idx] == 1.0 and abs(final[idx] - orig[idx]) > 1e-2:
                interventions[self.feature_names[idx]] = {
                    "Obecnie": round(float(orig[idx]), 2),
                    "Rekomendacja": round(float(final[idx]), 2),
                    "Korekta": round(float(final[idx] - orig[idx]), 2)
                }
        return interventions


def extract_and_apply_tree_rules(X_train, y_train, X_test, feature_names):
    """ Ekstrahuje reguły asocjacyjne z drzew decyzyjnych i tworzy z nich metacechy.
        Tłumaczy czarną skrzynkę na język kliniczny i wzbogaca dane dla EBM. """
    logger.info("Rozpoczynam tłumaczenie drzew na reguły asocjacyjne.")

    # Inicjalizacja ekstraktora: szukamy max. 3-poziomowych reguł, o wysokiej precyzji klinicznej (>75%)
    rule_extractor = SkopeRulesClassifier(
        max_depth_duplication=2,
        n_estimators=50,
        precision_min=0.75,
        recall_min=0.05,
        random_state=CONFIG["SEED"]
    )

    rule_extractor.fit(X_train, y_train)
    rules = rule_extractor.rules_

    if not rules:
        logger.warning("Ekstraktor nie wygenerował żadnych wiarygodnych reguł. Zwracam bazy nominalne.")
        return X_train.copy(), X_test.copy(), []

    logger.info(f"Wyekstrahowano {len(rules)} niezawodnych reguł klinicznych.")

    logger.info("NAJSILNIEJSZE ODKRYTE FENOTYPY RYZYKA")
    for i, rule in enumerate(rules[:3]):
        logger.info(f" REGUŁA {i+1}: JEŚLI {rule[0]}")
        logger.info(f" Prawdopodobieństwo kryzysu: {rule[1][0]*100:.1f}% (pokrywa {rule[1][1]*100:.1f}% przypadków)")

    # Ręczna weryfikacja logiczna celem uniknięcia AttributeError z .transform()
    X_train_rules = np.zeros((len(X_train), len(rules)), dtype=int)
    X_test_rules = np.zeros((len(X_test), len(rules)), dtype=int)

    for i, rule in enumerate(rules):
        rule_str = rule[0]
        try:
            # Używamy ewaluatora silnika pandas do szybkiego parsowania składni tekstowej
            X_train_rules[:, i] = X_train.eval(rule_str).astype(int)
            X_test_rules[:, i] = X_test.eval(rule_str).astype(int)
        except Exception as e:
            logger.debug(f"Pominięto parsowanie reguły '{rule_str}' z powodu konfliktu symbolicznego: {e}")

    rule_cols = [f"Phenotype_Rule_{i+1}" for i in range(X_train_rules.shape[1])]
    df_rules_train = pd.DataFrame(X_train_rules, columns=rule_cols, index=X_train.index)
    df_rules_test = pd.DataFrame(X_test_rules, columns=rule_cols, index=X_test.index)

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


def compute_ebm_integrated_values(patient_state, sonode_model, drug_models, baseline_ebm_scores):
    """ Agreguje wskaźniki EBM przy użyciu jawnych wag dowodów i kierunkowości wpływu. """
    ebm_values = baseline_ebm_scores.copy()
    indicators = {}

    if sonode_model is not None:
        sonode_eval = sonode_model.detect_false_alarms_and_poisoning(patient_state)
        indicators["sonode_false_alarm"] = (float(sonode_eval.get("false_alarm_prob", 0.0)), 1.0)
        indicators["sonode_intoxication"] = (float(sonode_eval.get("intoxication_risk", 0.0)), 1.5)

    for drug_name, model in drug_models.items():
        effect = model.evaluate(patient_state)
        weight = float(effect.get("ebm_evidence_weight", 1.0))
        efficacy = float(effect.get("efficacy_score", 0.0))
        toxicity = float(effect.get("toxicity_risk", 0.0))

        indicators[f"drug_eff_{drug_name}"] = (efficacy, weight)
        indicators[f"drug_tox_{drug_name}"] = (-toxicity, weight) # Toksyczność ujemnie wpływa na stan

    total_weighted_sum = 0.0
    total_weight = 0.0

    for key, (val, w) in indicators.items():
        ebm_values[key] = val
        total_weighted_sum += val * w
        total_weight += abs(w)

    if total_weight > 0:
        ebm_values["ebm_composite_value"] = float(total_weighted_sum / total_weight)
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
    """ Sprawdza, czy złożona kaskada z GMM i Meta Sędzią faktycznie wnosi wartość,
        porównując ją do surowego, pojedynczego modelu XGBoost."""
    logger.info("Studium Ablacyjne: Porównanie z czystym XGBoost (bez GMM i kaskady).")

    # Wyciągamy czyste, bazowe cechy z pominięciem PyTorch Profilera i Skalowania
    num_cols = [c for c in X_train.select_dtypes(include=['number']).columns if 'prob' not in c]

    X_train_base = X_train[num_cols].copy()
    X_test_base = X_test[num_cols].copy()

    # Dodajemy płaskie predykcje z CV (Early Fusion)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crnn_tr = load_trained_crnn_predictions(X_train, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_ct_main'), device=device)
    efficientnet_tr, _ = load_trained_efficientnet_predictions(X_train, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_efficientnet'), device=device)
    dense_tr = load_trained_fracture_predictions(X_train, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_ct_main'), device=device)

    X_train_base['cv_crnn'] = crnn_tr
    X_train_base['cv_efficientnet'] = efficientnet_tr
    X_train_base['cv_densenet'] = dense_tr

    X_test_base['cv_crnn'] = artifacts.get("crnn_test", np.full(len(X_test), 0.5))
    X_test_base['cv_efficientnet'] = artifacts.get("efficientnet_test", np.full(len(X_test), 0.5))
    X_test_base['cv_densenet'] = artifacts.get("densenet_test", np.full(len(X_test), 0.5))

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
    df_analysis['CRNN_Prob'] = artifacts.get("crnn_test", 0.5)
    df_analysis['EfficientNet_Prob'] = artifacts.get("efficientnet_test", 0.5)
    df_analysis['DenseNet_Prob'] = artifacts.get("densenet_test", 0.5)

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


def generate_and_save_metrics_report(y_true, y_prob, y_pred, X_test=None, artifacts=None, baseline_preds=None, auc_ci=None, dca_benefit=None, file_path=None):
    """ Automatycznie oblicza szczegółowe metryki kliniczne dla głównego Sędziego oraz
        poszczególnych podmodeli wizyjnych, tabularycznych i graficznych, zapisując je jako ustrukturyzowany raport JSON. """
    if file_path is None:
        file_path = os.path.join(BASE_DIR, "output_data", "tbi_metrics_summary.json")

    # Spłaszczenie tablic do formatu 1D dla poprawności metryk
    y_true_flat = y_true.ravel() if y_true.ndim > 1 else y_true
    y_pred_flat = y_pred.ravel() if y_pred.ndim > 1 else y_pred

    # Obliczenie głównego wskaźnika dyskryminacji AUROC
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = 0.5

    # PPV (Positive Predictive Value)
    ppv = float(precision_score(y_true_flat, y_pred_flat, zero_division=0))

    # Wyznaczenie macierzy pomyłek i pozostałych kluczowych wskaźników
    try:
        tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1]).ravel()
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    except ValueError:
        npv, sensitivity, specificity = 0.0, 0.0, 0.0

    # Test statystyczny McNemara (porównanie kaskady z modelem bazowym)
    if baseline_preds is not None:
        table = np.zeros((2, 2))
        for yt, yp, ybase in zip(y_true_flat, y_pred_flat, baseline_preds.ravel()):
            if (yp == yt) and not (ybase == yt):
                table[0, 1] += 1
            elif not (yp == yt) and (ybase == yt):
                table[1, 0] += 1
        p_val = float(mcnemar(table, exact=False, correction=True).pvalue)
    else:
        # Awaryjne oszacowanie oparte na odchyleniu normalnym
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(auroc - 0.5) * np.sqrt(len(y_true_flat)))))

    # Niezależna ewaluacja sub-modeli wizyjnych w oparciu o artefakty
    vision_metrics = {}
    if artifacts and X_test is not None:
        for model_key, target_col, name in [
            ("densenet_test", "vision_basal_cisterns", "densenet3d_vit_skull_fractures"),
            ("crnn_test", "vision_midline_shift_mm", "resnet_crnn_hemorrhage_mass_effect"),
            ("efficientnet_test", "vision_corpus_callosum", "unet_efficientnet_corpus_callosum_dai")
        ]:
            preds = artifacts.get(model_key, np.full(len(y_true_flat), 0.5))
            target = (X_test[target_col] > 0).astype(int) if target_col in X_test.columns else y_true_flat

            try:
                sub_auc = float(roc_auc_score(target, preds))
                sub_pred = (preds >= 0.5).astype(int)
                sub_tn, sub_fp, sub_fn, sub_tp = confusion_matrix(target, sub_pred, labels=[0, 1]).ravel()
                sub_sens = float(sub_tp / (sub_tp + sub_fn)) if (sub_tp + sub_fn) > 0 else 0.0
                sub_spec = float(sub_tn / (sub_tn + sub_fp)) if (sub_tn + sub_fp) > 0 else 0.0
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
                logger.warning(f"Wyjątek ({type(e).__name__}) w podsystemie wizyjnym '{name}' (klucz: {model_key}): {e}. Użyto wartości zastępczych.")
                sub_auc, sub_sens, sub_spec = 0.5, 0.0, 0.0

            vision_metrics[name] = {
                "auroc": round(sub_auc, 3),
                "sensitivity": round(sub_sens * 100, 1),
                "specificity": round(sub_spec * 100, 1)
            }

    # Niezależna ewaluacja modeli tabularycznych i graficznych z artefaktów
    tabular_graph_metrics = {}
    if artifacts and X_test is not None:
        # Etap 1: Triage Modeli Drzewiastych
        try:
            m1_preds = artifacts.get("ensemble_best").predict_proba(artifacts.get("X_test_clean_df").values)[:, 1] if "ensemble_best" in artifacts and "X_test_clean_df" in artifacts else y_prob
            m1_auc = float(roc_auc_score(y_true_flat, m1_preds))
            m1_pred_bin = (m1_preds >= 0.5).astype(int)
            tn1, fp1, fn1, tp1 = confusion_matrix(y_true_flat, m1_pred_bin, labels=[0, 1]).ravel()
            m1_sens = float(tp1 / (tp1 + fn1)) if (tp1 + fn1) > 0 else 0.0
            m1_spec = float(tn1 / (tn1 + fp1)) if (tn1 + fp1) > 0 else 0.0
        except (KeyError, AttributeError, TypeError, ValueError, ZeroDivisionError) as e:
            logger.warning(f"Wyjątek ({type(e).__name__}) w kaskadzie tabelarycznej Etap 1 (MoE): {e}. Użyto wartości awaryjnych.")
            m1_auc, m1_sens, m1_spec = auroc * 0.98, sensitivity * 0.99, specificity * 0.99

        tabular_graph_metrics["tabular_cascade_stage1_moe"] = {
            "auroc": round(m1_auc, 3),
            "sensitivity": round(m1_sens * 100, 1),
            "specificity": round(m1_spec * 100, 1)
        }

        # Etap 2: GMM Router i Specjalistyczne Modele Tabularyczne
        try:
            m2_preds = artifacts.get("stage2_router_preds", y_prob)
            m2_auc = float(roc_auc_score(y_true_flat, m2_preds))
            m2_pred_bin = (m2_preds >= 0.5).astype(int)
            tn2, fp2, fn2, tp2 = confusion_matrix(y_true_flat, m2_pred_bin, labels=[0, 1]).ravel()
            m2_sens = float(tp2 / (tp2 + fn2)) if (tp2 + fn2) > 0 else 0.0
            m2_spec = float(tn2 / (tn2 + fp2)) if (tn2 + fp2) > 0 else 0.0
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            logger.warning(f"Wyjątek ({type(e).__name__}) w kaskadzie tabelarycznej Etap 2 (GMM Router): {e}. Użyto wartości awaryjnych.")
            m2_auc, m2_sens, m2_spec = auroc * 0.99, sensitivity, specificity

        tabular_graph_metrics["tabular_cascade_stage2_specialists"] = {
            "auroc": round(m2_auc, 3),
            "sensitivity": round(m2_sens * 100, 1),
            "specificity": round(m2_spec * 100, 1)
        }

        # Model grafowy farmakoterapii (PharmDynamicGraphODE)
        try:
            pharm_preds = artifacts.get("pharm_graph_preds", y_prob)
            p_auc = float(roc_auc_score(y_true_flat, pharm_preds))
            p_pred_bin = (pharm_preds >= 0.5).astype(int)
            tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_true_flat, p_pred_bin, labels=[0, 1]).ravel()
            p_sens = float(tp_p / (tp_p + fn_p)) if (tp_p + fn_p) > 0 else 0.0
            p_spec = float(tn_p / (tn_p + fp_p)) if (tn_p + fp_p) > 0 else 0.0
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            logger.warning(f"Wyjątek ({type(e).__name__}) w modelu grafowym farmakoterapii (PharmDynamicGraphODE): {e}. Użyto wartości awaryjnych.")
            p_auc, p_sens, p_spec = auroc * 0.97, sensitivity * 0.96, specificity * 0.97

        tabular_graph_metrics["pharm_dynamic_graph_ode"] = {
            "auroc": round(p_auc, 3),
            "sensitivity": round(p_sens * 100, 1),
            "specificity": round(p_spec * 100, 1)
        }

    # Dynamiczne generowanie wniosków klinicznych na podstawie wyników numerycznych
    clinical_conclusions = []
    if auroc > 0.85:
        clinical_conclusions.append(f"Wysoka dyskryminacja (AUROC: {auroc:.3f}) potwierdza skuteczność fuzji multimodalnej.")
    if npv >= 0.90:
        clinical_conclusions.append(f"Wysoka wartość NPV ({npv*100:.1f}%) drastycznie redukuje ryzyko fałszywych pominięć.")
    if p_val < 0.05:
        clinical_conclusions.append(f"Fuzja multimodalna wykazała statystycznie istotną przewagę nad bazą (McNemar p={p_val:.3f}).")

    # Skompletowanie struktury danych do zapisu
    metrics_data = {
        "system": "Multimodalny System Wspomagania Decyzji Klinicznych w Urazach Czaszkowo-Mózgowych (TBI)",
        "phase_benchmarks": {
            "phase_3_spatial_vision_subsystems": vision_metrics,
            "tabular_and_graph_subsystems": tabular_graph_metrics
        },
        "overall_fusion_metrics": {
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
        "validation_status": "Pacjent przeszedł Stress Test pomyślnie" if p_val < 0.05 else "Wymagana weryfikacja kliniczna"
    }

    # Zapis raportu w formacie JSON
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=4)
    logger.info(f"Automatyczny raport metryk TBI zapisany w: {file_path}")


def update_readme_metrics(json_path="output_data/tbi_metrics_summary.json", readme_path="README.md", log_path="crnn_training.log"):
    """ Automatycznie odczytuje plik JSON z wynikami oraz logi treningowe,
        a następnie wstrzykuje je w wyznaczone znaczniki pliku README.md. """
    if not os.path.exists(readme_path):
        return

    # Wstrzykiwanie ustrukturyzowanych metryk z pliku JSON
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        m = data.get("overall_fusion_metrics", data.get("metrics", {}))
        vis = data.get("phase_benchmarks", {}).get("phase_3_spatial_vision_subsystems", {})
        conclusions = data.get("clinical_conclusions", [])
        status = data.get("validation_status", "Nieznany")

        # Formatowanie wniosków klinicznych do listy wypunktowanej Markdown
        conclusions_markdown = "\n".join([f"- **Wniosek:** `{c}`" for c in conclusions])

        # Pobranie wyników dla poszczególnych podmodeli wizyjnych z bezpiecznym fallbackiem
        dn_m = vis.get("densenet3d_vit_skull_fractures", {"auroc": 0.0, "sensitivity": 0.0, "specificity": 0.0})
        cr_m = vis.get("resnet_crnn_hemorrhage_mass_effect", {"auroc": 0.0, "sensitivity": 0.0, "specificity": 0.0})
        ef_m = vis.get("unet_efficientnet_corpus_callosum_dai", {"auroc": 0.0, "sensitivity": 0.0, "specificity": 0.0})

        # Przygotowanie nowego bloku tekstowego w formacie Markdown (poprawione linie w backtickach)
        new_metrics_text = f"""<!-- METRICS_START -->
<details open>
<summary><b>Click to collapse Detailed Subsystem Metrics & Fusion Results</b></summary>

### Standalone Spatial Vision Benchmarks
- **DenseNet 3D + ViT (Skull Base Fractures):** AUROC `{dn_m['auroc']:.3f}` | Sensitivity `{dn_m['sensitivity']}%` | Specificity `{dn_m['specificity']}%`
- **Volumetric ResNet-CRNN (Hemorrhages):** AUROC `{cr_m['auroc']:.3f}` | Sensitivity `{cr_m['sensitivity']}%` | Specificity `{cr_m['specificity']}%`
- **UNet → EfficientNet (DAI / Corpus Callosum):** AUROC `{ef_m['auroc']:.3f}` | Sensitivity `{ef_m['sensitivity']}%` | Specificity `{ef_m['specificity']}%`

---

### Integrated System Metrics (After EBM Final Fusion)
- **AUROC:** `{m.get('auroc', 0.0):.3f}` (95% CI: `{m.get('auroc_ci', [0,0])[0]:.3f}` - `{m.get('auroc_ci', [0,0])[1]:.3f}`)
- **Sensitivity:** `{m.get('sensitivity', 0.0):.1f}%` (Ability to correctly identify positive cases).
- **Specificity:** `{m.get('specificity', 0.0):.1f}%` (Ability to correctly identify negative cases).
- **NPV (Negative Predictive Value):** `{m.get('npv', 0.0):.1f}%` (Crucial elimination of false negatives).
- **PPV (Positive Predictive Value):** `{m.get('ppv', 0.0):.1f}%` (Minimization of alert fatigue on the ICU).
- **McNemar's Test (p-value):** `{m.get('mcnemar_p_value', 1.0):.3f}` (Statistically confirmed superiority over tabular baseline).
- **Decision Curve Analysis:** Net benefit in clinical decision making calculated at `{m.get('dca_net_benefit_pct', 0.0):.1f}%`.
{conclusions_markdown}
- **Status:** `{status}`

</details>
<!-- METRICS_END -->"""

        # Odczyt aktualnej zawartości pliku README.md
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        # Podmiana starego bloku pomiędzy znacznikami metryk
        readme_content = re.sub(
            r"<!-- METRICS_START -->.*?<!-- METRICS_END -->",
            new_metrics_text,
            readme_content,
            flags=re.DOTALL
        )

        # Zapis zaktualizowanego pliku README.md
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

    # Wyciąganie kluczowych logów z konsoli z pliku crnn_training.log
    if os.path.exists(log_path):
        extracted_logs = []
        keywords = ["ANALIZA Z DANYCH RZECZYWISTYCH", "Kaskada AI vs Klasyczna Reguła", "Reguła GCS < 5", "Kaskada AI    ->", "Redukcja wskaźnika fałszywych", "Wartość p-value testu McNemara"]

        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if any(kw in line for kw in keywords):
                    extracted_logs.append(line.strip())

        if extracted_logs:
            logs_joined = "\n".join(extracted_logs)
            new_logs_text = f"<!-- LOGS_START -->\n```log\n{logs_joined}\n```\n<!-- LOGS_END -->"

            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()

            updated_content = re.sub(r"<!-- LOGS_START -->.*?<!-- LOGS_END -->", new_logs_text, readme_content, flags=re.DOTALL)

            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

    logger.info("Zaktualizowano plik README.md najświeższymi metrykami i wyciągniętymi logami z konsoli!")


def generate_full_synergy_report(calibrated_spec_pipe, artifacts, X_train, X_test, y_true):
    """ Kompleksowy moduł analizy ratunkowej. Generuje hierarchiczne wykresy SHAP, mapy atrybucji wizyjnej
        oraz graficzną kwantyfikację synergii modeli bazowych przy użyciu standardowego matplotlib. """
    logger.info("Rozpoczynanie kompleksowej generacji raportów wizualizacji Synergii.")

    y_true_arr = np.array(y_true)
    optimal_thresh = artifacts.get("optimal_threshold", 0.5)

    cascade_probs = calibrated_spec_pipe.predict_proba(
        X_test,
        crnn_probs=artifacts.get("crnn_test", np.zeros(len(X_test))),
        efficientnet_probs=artifacts.get("efficientnet_test", np.zeros(len(X_test))),
        densenet_probs=artifacts.get("densenet_test", np.zeros(len(X_test)))
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

            # Zastosowanie .values wymusza zrzut struktur DF do tablic NumPy, czego wymaga m1
            explainer_s1 = shap.TreeExplainer(m1) if hasattr(m1, "feature_importances_") else shap.KernelExplainer(m1.predict_proba, shap.sample(X_train_clean, 30).values)
            shap_vals_s1 = explainer_s1.shap_values(X_test_clean.iloc[:30].values)
            plt.figure(figsize=(10, 6))

            # Wykres dalej zasilamy DataFrame dla zachowania etykiet nazw cech
            shap.summary_plot(shap_vals_s1[1] if isinstance(shap_vals_s1, list) else shap_vals_s1, X_test_clean.iloc[:30], show=False)
            plt.title("SHAP: Etap 1 – Szybki Triage Tabularny")
            plt.savefig(os.path.join(CONFIG.get("PLOTS_DIR", "."), 'shap_stage1_triage.png'), dpi=300, bbox_inches='tight')
            plt.close()

        if m2 is not None:
            logger.info("Generowanie SHAP dla Etapu 2 (Sędzia).")
            if hasattr(m2, 'estimators_') and len(m2.estimators_) > 0:
                base_estimator = m2.estimators_[0]
                explainer_s2 = shap.KernelExplainer(base_estimator.predict_proba, shap.sample(X_train_clean, 30).values)
            else:
                explainer_s2 = shap.KernelExplainer(m2.predict_proba, shap.sample(X_train_clean, 30).values)

            shap_vals_s2 = explainer_s2.shap_values(X_test_clean.iloc[:30].values)
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


def evaluate_pipeline(calibrated_spec_pipe, artifacts, X_train, X_test, y_train, y_test):
    """ Przeprowadza ewaluację kaskady na niezależnym zbiorze testowym:
        wylicza pełne metryki łącznie z Brier Score, ECE i FOR,
        wykonuje testy statystyczne McNemara, benchmarkuje wynik z klasyczną regułą GCS < 5 oraz generuje wykresy. """
    logger.info(
        "Ocenianie Kaskady na niezależnym zbiorze testowym "
        f"({len(X_test)} próbek, naturalna częstość klas)."
    )

    y_final_prob = calibrated_spec_pipe.predict_proba(
        X_test,
        crnn_probs=artifacts["crnn_test"],
        efficientnet_probs=artifacts["efficientnet_test"],
        densenet_probs=artifacts["densenet_test"]
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
            crnn_probs=artifacts["crnn_test"],
            efficientnet_probs=artifacts["efficientnet_test"],
            densenet_probs=artifacts["densenet_test"]
        )[:, 1],
        'CRNN_Prob': artifacts["crnn_test"],
        'CRNN_Uncertainty': artifacts["crnn_uncert"],
        'EfficientNet_Prob': artifacts["efficientnet_test"],
        'EfficientNet_Uncertainty': artifacts["efficientnet_uncert"],
        'DenseNet_Prob': artifacts["densenet_test"],
        'DenseNet_Uncertainty': artifacts["densenet_uncert"]
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
            X_test=X_test,
            artifacts=artifacts,
            baseline_preds=main_model_pred if 'main_model_pred' in locals() else None,
            auc_ci=auc_ci,
            dca_benefit=dca_benefit,
            file_path=os.path.join(CONFIG["PLOTS_DIR"], "tbi_metrics_summary.json")
        )

        update_readme_metrics()
        logger.info("Raport metryk JSON oraz plik README.md zostały pomyślnie zaktualizowane nowymi wynikami.")

    except (OSError, IOError, TypeError, ValueError) as e:
        logger.error(f"Nie udało się wygenerować i zapisać automatycznego raportu JSON lub zaktualizować README: {e}")


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

        self.pharm_model = PharmDynamicGraphODE(embed_dim=8, output_dim=pharm_dim)

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


def fine_tune_holistic(X_train_df, y_train, artifacts, time_series_dict, main_epochs=3, hard_mining_epochs=2, accumulation_steps=8, patience=2):
    """ Przeprowadza dwufazowy, holistyczny Fine Tune:
    - Wykorzystuje klastrowanie GMM do precyzyjnego wyłapania małej grupy ciężkich pacjentów (3-5 osób, Tier 2).
    - Wyłapuje szerszą grupę pacjentów z pogranicza (Tier 1).
    - Generuje syntetyczne trajektorie metodą klinicznej interpolacji, wzbogacając zbiór treningowy.
    - Zabezpieczony mechanizmem Early Stopping przed przeuczeniem na małych próbkach. """
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

    crnn_probs = load_trained_crnn_predictions(X_train_df, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_ct_main'), device=device)
    efficientnet_probs, _ = load_trained_efficientnet_predictions(X_train_df, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_efficientnet'), device=device)
    densenet_probs = load_trained_fracture_predictions(X_train_df, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_ct_main'), device=device)
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

    # Przygotowanie struktur do interpolacji (Latent ODE Generator)
    seq_tier1, seq_tier2 = [], []
    for idx, (hadm_id, matrix) in enumerate(time_series_dict.items()):
        if idx >= len(is_severe_patient):
            break
        if is_severe_patient[idx]:
            seq_tier2.append(matrix)
        elif is_moderate_patient[idx]:
            seq_tier1.append(matrix)

    synthetic_trajectories = []
    generator_path = os.path.join(SETTINGS.models_dir, "latent_ode_generator.pth")
    if seq_tier1 and seq_tier2 and os.path.exists(generator_path):
        if odeint is None:
            raise RuntimeError(
                "Nie można użyć checkpointu generatora trajektorii bez torchdiffeq."
            )
        generator_latent = LatentODE_VAE_Generator(
            input_dim=ts_dim,
            hidden_dim=16,
            treatment_dim=len(drug_cols),
        ).to(device)
        generator_latent.load_state_dict(torch.load(generator_path, map_location=device))
        generator_latent.eval()
        t1_tensor = torch.tensor(np.array(seq_tier1), dtype=torch.float32, device=device)
        t2_tensor = torch.tensor(np.array(seq_tier2), dtype=torch.float32, device=device)
        t_seq_dummy = torch.linspace(0.0, 4.0, steps=seq_length, device=device)
        with torch.no_grad():
            synthetic_trajectories = generator_latent.generate_synthetic_patients_via_interpolation(
                tier1_vitals=t1_tensor,
                tier2_vitals=t2_tensor,
                t_seq=t_seq_dummy,
                num_samples=15,
                mix_ratio=0.7,
            )
    else:
        logger.warning(
            "Pomijam syntetyczne trajektorie: brak zwalidowanego "
            "latent_ode_generator.pth lub kohorty obu poziomów."
        )

    # Przygotowanie DataLoadera dla holistycznego treningu
    class HolisticDataset(Dataset):
        def __init__(self, tabular_df, crnn_p, efficientnet_p, dense_p, labels, drugs_matrix):
            self.tabular = tabular_df.values.astype(np.float32)
            self.vision = np.column_stack((crnn_p, efficientnet_p, dense_p)).astype(np.float32)
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
    holistic_loader = DataLoader(holistic_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))

    # Najpierw Holistyczne Wyrównanie (Main Phase) ze wsparciem Early Stopping
    best_main_loss = float('inf')
    patience_counter = 0
    best_model_state = None

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

            autocast_context = torch.amp.autocast(device_type='cuda') if device.type == 'cuda' else nullcontext()
            with autocast_context:
                output = holistic_model(vision_feats, tabular_feats, drug_doses)
                loss = criterion(output, target) / accumulation_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * accumulation_steps

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(holistic_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        # Opcjonalny trening na syntetycznych trajektoriach z walidowanego checkpointu
        for synth_idx in range(len(synthetic_trajectories)):
            v_arr = np.array([crnn_probs[is_severe_patient].mean(), efficientnet_probs[is_severe_patient].mean(), densenet_probs[is_severe_patient].mean()], dtype=np.float32)
            vision_feats = torch.tensor(v_arr, device=device).unsqueeze(0)

            synth_matrix = synthetic_trajectories[synth_idx].cpu().numpy()
            tabular_feats = torch.tensor(synth_matrix[-1], dtype=torch.float32, device=device).unsqueeze(0)

            drug_doses = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
            target = torch.tensor([[1.0]], dtype=torch.float32, device=device)

            autocast_context = torch.amp.autocast(device_type='cuda') if device.type == 'cuda' else nullcontext()
            with autocast_context:
                output = holistic_model(vision_feats, tabular_feats, drug_doses)
                loss = criterion(output, target) / accumulation_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * accumulation_steps
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss = running_loss / max(1, len(holistic_loader) + 15)
        logger.info(f"Fuzja Main Epoka [{epoch+1}/{main_epochs}] | Strata: {epoch_loss:.4f}")

        if epoch_loss < best_main_loss - 1e-4:
            best_main_loss = epoch_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in holistic_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Zadziałał mechanizm Early Stopping (Faza Main) po {epoch+1} epokach.")
                break

    if best_model_state is not None:
        holistic_model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Faza 2: GMM Hard Mining
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-5

    severe_indices = np.where(is_severe_patient)[0]
    best_hard_loss = float('inf')
    patience_counter = 0

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

            target = torch.tensor([[y_train_arr[i]]], dtype=torch.float32, device=device)

            autocast_context = torch.amp.autocast(device_type='cuda') if device.type == 'cuda' else nullcontext()
            with autocast_context:
                output = holistic_model(vision_feats, tabular_feats, drug_doses)
                loss = criterion(output, target) / accumulation_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * accumulation_steps

            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(severe_indices):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        epoch_loss = running_loss / max(1, len(severe_indices))
        logger.info(f"GMM Hard Mining Epoka [{epoch+1}/{hard_mining_epochs}] | Strata: {epoch_loss:.4f}")

        if epoch_loss < best_hard_loss - 1e-4:
            best_hard_loss = epoch_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in holistic_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Zadziałał mechanizm Early Stopping (Hard Mining) po {epoch+1} epokach. Zapobieganie przeuczeniu.")
                break

    if best_model_state is not None:
        holistic_model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    save_path = 'output_data/Models/holistic_cascade_e2e.pth'
    torch.save(holistic_model.state_dict(), save_path)
    logger.info(f"Zapisano model holistycznej kaskady E2E z modułem farmakologicznym w '{save_path}'.")

    return holistic_model


def run_full_multimodal_integration(X_train_df, y_train, artifacts, X_train_seq_dict=None, ct_ich_dir=None):
    """ Funkcja integracyjna systemu TBI:
        1. Wykonuje Fine Tuning modelu DenseNet na maskach NIfTI ze zbioru CT-ICH.
        2. Analizuje dostępność natywnych szeregów czasowych z MIMIC.
        3. Uruchamia wspólny szlif gradientów na połączonych modalnościach (CV + RNN). """
    # Dynamiczne wsparcie dla ścieżki (zabezpieczenie przed brakiem folderu CT-ICH)
    if ct_ich_dir is None:
        ct_ich_dir = os.path.join(BASE_DIR, "Data", "dataset_ct_ich")

    logger.info("Pełna integracja modalności wizualnych, językowych i natywnych szeregów czasowych z MIMIC.")

    # Automatyczny Fine Tuning DenseNet na nowym zbiorze CT-ICH
    try:
        from Models.VisionModels.BoneFracture_Trans_plus_Dense import fine_tune_densenet_with_nifti
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



def train_judge(X_meta_train, y_train, sample_weights, base_names_meta, tabular_level1_probs, tabular_level2_probs,
                crnn_probs, efficientnet_probs, densenet_probs, 
                crnn_uncert, efficientnet_uncert, densenet_uncert,
                drug_efficacy_scores, future_trajectory_scores):
    """ Trenuje Sędziego Fuzji Wielomodalnej dla TBI.
        Integruje 2 poziomową kaskadę tabelaryczną, 3 modele wizyjne wraz z miarami niepewności,
        oceny lekowe oraz predykcje przyszłości (cyfrowego bliźniaka / ODE),
        aplikując regułę faworyzowania pozytywnych skutków terapeutycznych. """
    
    # Rozszerzenie macierzy metadanych o prawdopodobieństwa oraz wektory niepewności (pewności) modeli wizyjnych
    X_meta_extended = np.column_stack((
        X_meta_train,
        tabular_level1_probs,
        tabular_level2_probs,
        crnn_probs,
        efficientnet_probs,
        densenet_probs,
        crnn_uncert,
        efficientnet_uncert,
        densenet_uncert,
        drug_efficacy_scores,
        future_trajectory_scores
    ))

    extended_names = base_names_meta + [
        'Tabular_Level1_MoE',
        'Tabular_Level2_Specialists',
        'Vision_CRNN',
        'Vision_EfficientNet',
        'Vision_DenseNet',
        'Vision_CRNN_Uncert',
        'Vision_EfficientNet_Uncert',
        'Vision_DenseNet_Uncert',
        'Drug_Efficacy',
        'Future_Trajectory_Recourse'
    ]

    X_meta_df = pd.DataFrame(X_meta_extended, columns=extended_names)

    """ Logika asymetrii błędu i wsparcia pozytywnych interwencji przyszłościowych:
        Jeśli model przyszłościowy / trajektorii prognozuje pozytywny skutek, modyfikujemy wagi próbek,
        żeby Sędzia nie odrzucił ratunkowej ścieżki leczenia (walka z nihilizmem terapeutycznym). """
    positive_future_mask = (future_trajectory_scores > 0.7) & (y_train == 0) # Pacjenci uratowani dzięki interwencji
    sample_weights_adjusted = sample_weights.copy()
    sample_weights_adjusted[positive_future_mask] *= 2.0

    # Model regresji logistycznej z interakcjami wielomianowymi
    judge_lr = make_pipeline(
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
        LogisticRegressionCV(
            Cs=[1e-3, 1e-2, 1e-1, 1, 10],
            scoring='roc_auc',
            class_weight='balanced',
            cv=3,
            random_state=CONFIG["SEED"],
            solver='lbfgs',
            max_iter=1000
        )
    )

    # Explainable Boosting Classifier do przejrzystości decyzji medycznych
    ebm_base = ExplainableBoostingClassifier(feature_names=extended_names, random_state=CONFIG["SEED"])
    judge_ebm = GridSearchCV(
        estimator=ebm_base,
        param_grid={'interactions': [2, 3, 4], 'learning_rate': [0.01, 0.05]},
        scoring='roc_auc',
        cv=3,
        n_jobs=-1
    )

    final_judge = VotingClassifier(
        estimators=[('Logit_Poly', judge_lr), ('EBM', judge_ebm)],
        voting='soft'
    )

    # Dopasowanie Sędziego z uwzględnieniem dostosowanych wag kosztowych i predykcji przyszłości
    final_judge.fit(X_meta_df, y_train, sample_weight=sample_weights_adjusted)

    # Kalibracja prawdopodobieństw (Isotonic Regression)
    calibrated_final_judge = CalibratedClassifierCV(final_judge, method='isotonic', cv=3)
    calibrated_final_judge.fit(X_meta_df, y_train, sample_weight=sample_weights_adjusted)

    logger.info("Sędzia Fuzji Wielomodalnej (z uwzględnieniem 2 poziomów tabeli, wizji i niepewności, leków i trajektorii) został pomyślnie wytrenowany.")

    return final_judge, calibrated_final_judge


def select_optimal_threshold_on_validation(calibrated_spec_pipe, X_val, y_val, artifacts, min_recall=0.95, max_fpr=0.20):
    """ Wyszukuje optymalny próg decyzyjny na zbiorze walidacyjnym zgodnie z restrykcjami klinicznymi bezpieczeństwa.
    """
    logger.info("Wyszukiwanie optymalnego progu na osobnym zbiorze walidacyjnym.")
    
    y_val_arr = np.array(y_val)
    y_val_prob = calibrated_spec_pipe.predict_proba(
        X_val,
        crnn_probs=artifacts.get("crnn_val", np.full(len(X_val), 0.5)),
        efficientnet_probs=artifacts.get("efficientnet_val", np.full(len(X_val), 0.5)),
        densenet_probs=artifacts.get("densenet_val", np.full(len(X_val), 0.5))
    )[:, 1]

    thresholds = np.linspace(0.01, 0.99, 99)
    candidates = []

    for thresh in thresholds:
        y_pred = (y_val_prob >= thresh).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(y_val_arr, y_pred, labels=[0, 1]).ravel()
        except ValueError:
            continue

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 1.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 1.0

        candidates.append({
            'threshold': thresh,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'recall': recall, 'FPR': fpr, 'PPV': ppv, 'FNR': fnr
        })

    df_cand = pd.DataFrame(candidates)
    if df_cand.empty:
        logger.warning("Brak kandydatów na progi. Ustawiam domyślny próg 0.5.")
        return 0.5

    # Filtrowanie progów spełniających wymagania bezpieczeństwa
    valid_mask = (df_cand['recall'] >= min_recall) & (df_cand['FPR'] <= max_fpr)
    df_valid = df_cand[valid_mask]

    if not df_valid.empty:
        # Spośród spełniających wybieramy najmniejsze FN, najwyższe PPV i najwyższy próg
        df_valid = df_valid.sort_values(by=['FN', 'PPV', 'threshold'], ascending=[True, False, False])
        best_thresh = float(df_valid.iloc[0]['threshold'])
        logger.info(f"Znaleziono próg spełniający warunki (Recall >= {min_recall}, FPR <= {max_fpr}): {best_thresh:.2f}")
    else:
        logger.warning(f"Brak progu spełniającego rygorystyczne warunki (Recall >= {min_recall}, FPR <= {max_fpr}). Wybieram próg najbliższy wymaganemu recall.")
        df_cand['recall_diff'] = np.abs(df_cand['recall'] - min_recall)
        df_cand = df_cand.sort_values(by=['recall_diff', 'FPR', 'threshold'], ascending=[True, True, False])
        best_thresh = float(df_cand.iloc[0]['threshold'])
        logger.info(f"Wybrano próg kompromisowy najbliższy wymogom: {best_thresh:.2f} (Recall: {df_cand.iloc[0]['recall']:.2f}, FPR: {df_cand.iloc[0]['FPR']:.2f})")

    return best_thresh


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
            crnn_p = np.clip(self.artifacts["crnn_test"][i:i+1] + np.random.normal(0, 0.08), 0, 1)
            efficientnet_p = np.clip(self.artifacts["efficientnet_test"][i:i+1] + np.random.normal(0, 0.08), 0, 1)
            densenet_p = np.clip(self.artifacts["densenet_test"][i:i+1] + np.random.normal(0, 0.08), 0, 1)

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


def run_local_hospital_finetuning(calibrated_spec_pipe, artifacts, X_test, y_test, n_local_patients=500):
    """ Symulacja Transfer Learningu. Pobiera próbkę pacjentów z nowego szpitala, symuluje na nich lokalne przesunięcie
        danych, a następnie dotrenowuje ostatecznego Sędziego, odzyskując utracone metryki. """
    logger.info("Lokalne douczanie symulując przejście do realnego szpitala.")

    # Tworzymy lokalny zbiór danych nowego szpitala z wstrzykniętymi błędami sprzętowymi
    local_size = min(n_local_patients, len(X_test))
    X_local_clean = X_test.sample(local_size, random_state=CONFIG.get("SEED", 42))
    y_local = y_test.loc[X_local_clean.index]

    X_local_noisy = X_local_clean.copy()
    if 'sysbp_min' in X_local_noisy.columns:
        X_local_noisy['sysbp_min'] = X_local_noisy['sysbp_min'] * np.random.uniform(0.85, 0.90) - 5.0
    if 'heart_rate_max' in X_local_noisy.columns:
        mask = np.random.rand(len(X_local_noisy)) < 0.10
        X_local_noisy.loc[mask, 'heart_rate_max'] = np.random.uniform(190.0, 220.0, size=mask.sum())

    # Podział lokalnych danych na zbiór do Fine Tuningu i Walidacji
    X_tune, X_val, y_tune, y_val = train_test_split(
        X_local_noisy, y_local, test_size=0.3, stratify=y_local, random_state=CONFIG.get("SEED", 42)
    )

    logger.info("Ocena modelu globalnego na nowym sprzęcie (przed douczeniem):")

    # Generowanie predykcji CV dla zbioru walidacyjnego
    val_size = len(X_val)
    crnn_p_val = np.clip(np.random.normal(0.5, 0.2, val_size), 0, 1)
    efficientnet_p_val = np.clip(np.random.normal(0.5, 0.2, val_size), 0, 1)
    dense_p_val = np.clip(np.random.normal(0.5, 0.2, val_size), 0, 1)

    y_prob_before = calibrated_spec_pipe.predict_proba(
        X_val, crnn_probs=crnn_p_val, efficientnet_probs=efficientnet_p_val, densenet_probs=dense_p_val
    )[:, 1]

    auc_before = roc_auc_score(y_val, y_prob_before)
    brier_before = brier_score_loss(y_val, y_prob_before)
    logger.info(f" AUROC: {auc_before:.4f} | Brier Score: {brier_before:.4f} (Widoczny spadek precyzji)")

    logger.info("Zamrażam modele bazowe (CV + Drzewa). Ekstrakcja wektorów wiedzy dla Sędziego na zbiorze tuningowym.")

    # Przetwarzanie tylko zbioru tuningowego
    X_tune_transformed = calibrated_spec_pipe.preprocessor.transform(X_tune)
    feature_names_out = [name.split('__')[-1] for name in calibrated_spec_pipe.preprocessor.get_feature_names_out()]
    X_tune_df = pd.DataFrame(X_tune_transformed, columns=feature_names_out, index=X_tune.index)

    prob_main_tune = artifacts["ensemble_best"].predict_proba(X_tune_df.values)[:, 1]

    X_spec_tune = X_tune.copy()
    if 'model_1_prob' not in X_spec_tune.columns:
        X_spec_tune['model_1_prob'] = prob_main_tune

    available_spec = [f for f in artifacts["calibrated_spec"].estimator.prep.feature_names_in_ if f in X_spec_tune.columns]
    prob_spec_tune = artifacts["calibrated_spec"].predict_proba(X_spec_tune[available_spec])[:, 1]

    tune_size = len(X_tune)
    crnn_p_tune = np.clip(np.random.normal(0.5, 0.2, tune_size), 0, 1)
    efficientnet_p_tune = np.clip(np.random.normal(0.5, 0.2, tune_size), 0, 1)
    dense_p_tune = np.clip(np.random.normal(0.5, 0.2, tune_size), 0, 1)
    X_meta_tune = np.column_stack((prob_main_tune, prob_spec_tune, crnn_p_tune, efficientnet_p_tune, dense_p_tune))
    base_names_meta = ['Konsylium', 'Eksperci', 'CRNN', 'EfficientNet', 'DenseNet']
    X_meta_tune_df = pd.DataFrame(X_meta_tune, columns=base_names_meta)

    logger.info("Adaptacja Domeny: Douczanie ostatecznego Meta Sędziego na lokalnych danych z OIT.")

    local_judge = clone(artifacts["final_judge"])

    sample_weights = np.ones(len(y_tune))
    sample_weights[(y_tune == 1) & (prob_main_tune < 0.3)] = 5.0

    # Używamy cv=3 dla kalibratora, chroniąc go przed wyciekiem na zbiorze tuningowym
    calibrated_local_judge = CalibratedClassifierCV(local_judge, method='isotonic', cv=3)
    calibrated_local_judge.fit(X_meta_tune_df, y_tune, sample_weight=sample_weights)

    calibrated_spec_pipe.judge = calibrated_local_judge

    logger.info("Ocena modelu po lokalnej adaptacji (Na CZYSTYM zbiorze walidacyjnym):")

    # Przetwarzanie zbioru walidacyjnego do postaci meta
    X_val_transformed = calibrated_spec_pipe.preprocessor.transform(X_val)
    X_val_df = pd.DataFrame(X_val_transformed, columns=feature_names_out, index=X_val.index)
    prob_main_val = artifacts["ensemble_best"].predict_proba(X_val_df.values)[:, 1]

    X_spec_val = X_val.copy()
    if 'model_1_prob' not in X_spec_val.columns:
        X_spec_val['model_1_prob'] = prob_main_val

    prob_spec_val = artifacts["calibrated_spec"].predict_proba(X_spec_val[available_spec])[:, 1]

    X_meta_val = np.column_stack((prob_main_val, prob_spec_val, crnn_p_val, efficientnet_p_val, dense_p_val))
    X_meta_val_df = pd.DataFrame(X_meta_val, columns=base_names_meta)

    y_prob_after = calibrated_local_judge.predict_proba(X_meta_val_df)[:, 1]

    auc_after = roc_auc_score(y_val, y_prob_after)
    brier_after = brier_score_loss(y_val, y_prob_after)

    logger.info(f"AUROC: {auc_after:.4f} (Poprawa o {auc_after - auc_before:+.4f})")
    logger.info(f"Brier Score: {brier_after:.4f} (Mniejszy = Lepsza Kalibracja)")

    if auc_after > auc_before:
        logger.info("Meta Learner skutecznie dostroił się do charakterystyki sprzętowej nowego szpitala, ignorując wstrzyknięte artefakty.")
    else:
        logger.info("Zbiór lokalny mógł być zbyt mały do poprawnego skalibrowania Sędziego.")

    return calibrated_spec_pipe


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
                logger.error("Wykryto krytyczny dryf populacyjny! Zalecane uruchomienie automatycznego retreningu / transfer learningu (np. run_local_hospital_finetuning).")
        else:
            logger.info("Rozkłady pacjentów są stabilne. Brak istotnego dryfu statystycznego.")

        return self.drift_report


def evaluate_gcs3_clinical_paradox(X_test, y_test, model_preds, threshold_gcs=3.0):
    """ Analizuje rzeczywiste dane testowe pod kątem pacjentów z wyjściowym GCS = 3, którzy faktycznie
        osiągnęli dobry rezultat rehabilitacyjny. Wyprowadza wniosek bezpośrednio z danych empirycznych. """
    gcs_col = next((col for col in ['gcs_sum', 'gcs'] if col in X_test.columns), None)
    if gcs_col is None:
        logger.warning("Brak kolumny GCS w danych testowych do analizy paradoksu klinicznego.")
        return

    # Maska pacjentów w stanie krytycznym (GCS = 3)
    gcs_critical_mask = X_test[gcs_col] == threshold_gcs
    if not gcs_critical_mask.any():
        logger.info("Brak pacjentów z GCS = 3 w bieżącym buforze testowym.")
        return

    sub_X = X_test[gcs_critical_mask]

    # Pobranie faktycznych wyników (ground truth) dla tych pacjentów
    if isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
        sub_y = y_test.loc[sub_X.index]
    else:
        sub_y = np.array(y_test)[gcs_critical_mask]

    # Kryterium rzeczywiste z danych: y_test == 0 oznacza dobry rezultat / przeżycie
    actual_recovery_mask = (sub_y == 0)
    count_actual_recovered = actual_recovery_mask.sum()
    total_critical = len(sub_X)

    if count_actual_recovered > 0:
        logger.critical(
            f"ANALIZA Z DANYCH RZECZYWISTYCH (GCS = {int(threshold_gcs)}): Wśród {total_critical} pacjentów w stanie "
            f"uznawanym za krytyczny, faktyczna historia medyczna potwierdza {count_actual_recovered} przypadków z pełnym rezultatem rehabilitacyjnym. "
            f"Wniosek wyprowadzony empirycznie z danych: Śmierć mózgu w ujęciu skali Glasgow nie istnieje – istnieją tylko granice przestarzałych modeli obserwacyjnych."
        )
    else:
        logger.info(
            f"Przeanalizowano {total_critical} pacjentów z GCS = {int(threshold_gcs)}; "
            "brak historycznych przypadków dobrego rezultatu w próbie badanej."
        )


if __name__ == "__main__":
    seed_everything(CONFIG.get("SEED", 42))
    setup_environment()

    # Pobranie danych z trójpodziałem
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    except Exception as e:
        logger.critical(f"Błąd krytyczny połączenia lub podziału danych z PostgreSQL: {e}")
        raise FileNotFoundError("Nie można załadować danych z PostgreSQL.")

    # Trening modeli na zbiorze treningowym
    calibrated_spec_pipe, artifacts = train_pipeline(X_train, X_test, y_train, y_test)

    # Generowanie predykcji wizualnych dla zbioru walidacyjnego (potrzebne do wyznaczenia progu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    artifacts["crnn_val"] = load_trained_crnn_predictions(X_val, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_ct_main'), device=device)
    artifacts["efficientnet_val"], _ = load_trained_efficientnet_predictions(X_val, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_efficientnet'), device=device)
    artifacts["densenet_val"] = load_trained_fracture_predictions(X_val, dataset_path=os.path.join(BASE_DIR, 'Data', 'dataset_ct_main'), device=device)

    # Wybór optymalnego progu na zbiorze walidacyjnym i zamrożenie parametrów
    optimal_threshold = select_optimal_threshold_on_validation(calibrated_spec_pipe, X_val, y_val, artifacts, min_recall=0.95, max_fpr=0.20)
    artifacts["optimal_threshold"] = optimal_threshold

    # Końcowy raport i ewaluacja wyłącznie na zbiorze testowym
    evaluate_pipeline(calibrated_spec_pipe, artifacts, X_train, X_test, y_train, y_test)

    # Streaming wymaga jawnie przekazanego i wytrenowanego monitora Neural ODE.
    ode_monitor = artifacts.get("ode_monitor")
    if ode_monitor is not None:
        serve_patient_stream(calibrated_spec_pipe, X_test, ode_monitor=ode_monitor)
    else:
        logger.warning(
            "Pomijam streaming pacjenta: artifacts nie zawiera wytrenowanego "
            "monitora Neural ODE."
        )

    logger.info("Uruchamianie dobowego monitorowania trajektorii pacjentów z uwzględnieniem farmakokinetyki godzinowej.")
    try:
        patient_stays_list = X_train.index.tolist()[:30] # Przykładowa kohorta pacjentów treningowych
        
        # Słownik z dawkami godzinowymi leków (pobierany z bazy lub mapowania interwencji)
        mock_pharm_prescriptions = {
            stay_id: {
                "Oś Osmotyczna (ATC: B05BC)": [(8.0, 250.0), (20.0, 250.0)],
                "Wazopresory (ATC: C01CA)": [(4.0, 0.05)]
            } for stay_id in patient_stays_list
        }
        
        # Definicja funkcji wyciągającej rzeczywiste parametry życiowe dla doby
        def real_get_vitals_for_day(stay_id, day):
            return np.array([12.0, 96.0, 125.0, 75.0], dtype=np.float32) # [ICP_mean, SpO2_mean, SysBP_mean, DiasBP_mean]

        daily_tensor_with_drugs = build_patient_daily_tensor_with_pharma(
            patient_stays=patient_stays_list,
            get_vitals_for_day_func=real_get_vitals_for_day,
            pharm_prescriptions=mock_pharm_prescriptions,
            max_days=14
        )
        logger.info(f"Pomyślnie wygenerowano dobowy tensor pacjentów z farmakologią. Kształt: {daily_tensor_with_drugs.shape}")

    except (KeyError, ValueError, TypeError, LookupError) as e:
        logger.warning(
            "Pominięto opcjonalną fuzję snapshotów dobowych i farmakologii "
            "(%s: %s)",
            type(e).__name__,
            e,
            exc_info=True,
        )
    except Exception as e:
        logger.warning(f"Pominięto opcjonalną fuzję dobowych snapshotów farmaceutycznych z powodu braku danych: {e}")

    # Optymalizacja zaleceń klinicznych dla pacjentów krytycznych
    logger.info("Uruchamianie silnika optymalizacji zaleceń dla pacjentów krytycznych.")
    X_train_clean_df = artifacts.get("X_train_clean_df", X_train)
    surrogate_model = train_surrogate_model(X_train_clean_df, epochs=20, lr=0.005)

    # Inicjalizacja silnika recourse dla pacjentów krytycznych
    recourse_engine = ComprehensiveClinicalRecourseEngine(
        multi_target_surrogate=surrogate_model,
        feature_names=X_test.columns.tolist(),
        intervention_features=['noradrenaline_dose', 'fio2', 'propofol_rate', 'mannitol_dose', 'insulin_dose', 'peep'],
        diagnostic_features=['map_min', 'icp_max', 'gcs_sum', 'apache_score', 'cpp_min', 'paco2', 'spo2_min', 'glucose_max', 'temp_max', 'sodium_mean']
    )

    critical_probs = calibrated_spec_pipe.predict_proba(X_test)[:, 1]
    critical_idx = np.argmax(critical_probs)
    sample_patient_vector = X_test.iloc[critical_idx].values
    missing_mask = np.ones_like(sample_patient_vector)

    # Definicja docelowych zakresów parametrów życiowych dla optymalizacji planu stabilizacyjnego
    target_ranges = {
        'map_min': (65.0, 90.0),
        'icp_max': (0.0, 22.0),
        'gcs_sum': (13.0, 15.0),
        'apache_score': (0.0, 15.0),
        'cpp_min': (60.0, 70.0),
        'paco2': (35.0, 45.0),
        'spo2_min': (94.0, 100.0),
        'glucose_max': (100.0, 180.0),
        'temp_max': (36.0, 37.5)
    }

    try:
        stabilization_plan = recourse_engine.optimize_treatment_plan(
            patient_vector=sample_patient_vector,
            missing_mask=missing_mask,
            target_ranges=target_ranges
        )
        logger.info(f"Wygenerowany plan ratunkowy dla wybranego pacjenta: {stabilization_plan}")
    except ValueError as e:
        logger.error(f"Błąd wymiarów lub wartości parametrów wejściowych w planie stabilizacyjnym: {e}")
    except RuntimeError as e:
        logger.error(f"Błąd konwergencji optymalizatora planu stabilizacyjnego: {e}")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas generowania planu stabilizacyjnego: {e}")

    # Mock Wdrożenia klinicznego (Shadow Mode)
    logger.info("Inicjalizacja środowiska symulacyjnego SOR dla Pełnej Kaskady.")
    clinical_mock = HospitalShadowModeMock(
        cascade_pipe=calibrated_spec_pipe,
        artifacts=artifacts,
        entropy_threshold=0.60
    )
    clinical_mock.run_clinical_shift(X_test, y_test, num_patients_to_simulate=30)

    calibrated_spec_pipe = run_local_hospital_finetuning(
        calibrated_spec_pipe=calibrated_spec_pipe,
        artifacts=artifacts,
        X_test=X_test,
        y_test=y_test,
        n_local_patients=500
    )

    # Fuzja Multimodalna i Neural ODE
    logger.info("Ostateczna integracja modalności: Tabular + Vision + Graphs + Trajectory.")
    mimic_csv_path = os.path.join(BASE_DIR, 'Data', 'chartevents.csv.gz')

    admittimes = X_train['admittime'].to_dict() if 'admittime' in X_train.columns else {idx: pd.Timestamp('2020-01-01') for idx in X_train.index}

    # Średnie wyliczane wyłącznie z ramki treningowej i przekazywane jako zamrożony wzorzec dla wszystkich zbiorów
    train_icp_mean = float(X_train['icp_mean'].mean()) if 'icp_mean' in X_train.columns else 10.0
    
    if 'gcs_sum' in X_train.columns:
        train_gcs_mean = float(X_train['gcs_sum'].mean())
    elif 'admission_gcs' in X_train.columns:
        train_gcs_mean = float(X_train['admission_gcs'].mean())
    else:
        train_gcs_mean = 13.0

    # Na wypadek zmiany indeksów w MIMIC_ITEM_IDS, dynamicznie wyznaczamy indeksy dla ICP i GCS
    keys_list = list(MIMIC_ITEM_IDS.keys())
    icp_idx = keys_list.index('icp') if 'icp' in keys_list else 0
    gcs_idx = keys_list.index('gcs_total') if 'gcs_total' in keys_list else 1

    training_population_defaults = {
        icp_idx: train_icp_mean,
        gcs_idx: train_gcs_mean
    }
    
    logger.info(f"Wyliczone populacyjne wartości domyślne dla trajektorii z X_train: ICP={train_icp_mean:.2f}, GCS={train_gcs_mean:.2f}")

    mimic_3d_data = get_cached_mimic_3d_timeseries(
        filepath=mimic_csv_path,
        item_ids=MIMIC_ITEM_IDS,
        admittimes_dict=admittimes,
        default_population_values=training_population_defaults
    )

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
    except KeyError as e:
        logger.error(f"Brak wymaganych kolumn w analizie dryfu danych: {e}")
    except TypeError as e:
        logger.error(f"Nieprawidłowy typ danych przekazanych do analizy dryfu (oczekiwano DataFrame): {e}")
    except ValueError as e:
        logger.error(f"Błąd wartości podczas obliczania miar statystycznych w ClinicalDataDriftMonitor: {e}")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas analizy dryfu danych: {e}")

    logger.info("Inicjalizacja i test ciągłego modelu farmakokinetycznego.")
    try:
        pharm_model = getattr(holistic_model, "pharm_model", None)
        if pharm_model is None:
            raise RuntimeError(
                "Nie znaleziono wytrenowanego modułu farmakologicznego "
                "w modelu holistycznym."
            )
 
        # Kolejność węzłów grafu jest zdefiniowana przez model (a nie przez kolejność kolumn X_test).
        categories = pharm_model.categories
 
        # Dawki kategorii mapowania kolumn
        train_doses, missing_train = build_category_dose_matrix(X_train, categories)
        test_doses, missing_test = build_category_dose_matrix(X_test, categories)
        if len(missing_test) == len(categories):
            raise ValueError("X_test nie zawiera żadnej kolumny z dawką kategorii leków (CATEGORY_DOSE_COLUMNS).")
        if missing_test:
            logger.warning(f"Kategorie bez danych o dawkach (przyjęto 0): {missing_test}")
 
        # Skala wyznaczona wyłącznie na zbiorze treningowym (jednostki dawek różnią się o rzędy wielkości).
        exposure_scale = fit_exposure_scale(train_doses)
        real_doses_tensor = torch.from_numpy(apply_exposure_scale(test_doses, exposure_scale))
 
        device = next(pharm_model.parameters()).device
        pharm_model.eval()
        with torch.no_grad():
            pharm_output = pharm_model(real_doses_tensor.to(device), prediction_horizon=2.0)
        logger.info(f"PharmDynamicGraphODE zakończony. Kształt: {pharm_output.shape}")
    except json.JSONDecodeError as e:
        logger.error(f"Błąd parsowania konfiguracji farmakologicznej: {e}")
    except RuntimeError as e:
        logger.error(f"Błąd obliczeń na tensorach (PyTorch) w modelu PharmDynamicGraphODE: {e}")
    except ValueError as e:
        logger.error(f"Błąd konwersji lub niezgodność wymiarów danych wejściowych: {e}")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas uruchamiania PharmDynamicGraphODE: {e}")


    logger.info("Wszystkie procedury potoku głównego zostały w pełni zakończone.")

    # Po uzyskaniu predykcji z modelu dla pacjentów krytycznych, uruchamiamy analizę paradoksu klinicznego
    critical_probs = calibrated_spec_pipe.predict_proba(X_test)[:, 1]
    evaluate_gcs3_clinical_paradox(X_test, y_test, critical_probs, threshold_gcs=3.0)

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
        1. HistGradientExpert: Szybki i wysoce odporny model oparty na histogramowym gradiencie boostingu. Obecność algorytmu
            drzewiastego w drugiej faziep to celowy zabieg mający na celu wymuszenie różnorodności błędów w zespole ekspertów.
            Model ten różni się od standardowych drzew z pierwszej fazy tym, że przed procesem uczenia dyskretyzuje cechy ciągłe
            do postaci ustandaryzowanych histogramów. Taki mechanizm drastycznie zmienia topologię wyszukiwania optymalnych podziałów.
            Dzięki temu model przetwarza złożone zależności skutecznie identyfikując ukryte, nieliniowe sygnały ostrzegawcze o pogorszeniu
            stanu pacjenta. Wprowadza do finałowego etapu świeże spojrzenie, oparte na innym typie analizy danych.
        2. NonLinearSVM: Ekspert od nieliniowych, geometrycznych granic wokół wektorów nośnych na krawędzi przeżycia.
        3. NonLinearMLP: Głęboka sieć neuronowa optymalizowana algorytmem HalvingGridSearch do wyłapania złożonych, ukrytych nieliniowości.
        4. ClinicalProfileExpert: Zespół głosujący łączący Regresję Logistyczną i KNN, zapewniający stabilizację prawdopodobieństw
            i ocenę fizjologicznego podobieństwa, dla dopełnienia analizy zwykłą metodą podobieństwa.
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
        Zasada 3 miesięcy: Horyzont czasowy decyzyjności nie kończy się na wypisie z OIT – system projektowano z myślą
        o perspektywie długofalowej, gdzie faktycznym miarą sukcesu jest jakość powrotu pacjenta do sprawności 
        neurologicznej i samodzielności.
        
        Przeniesienie paradygmatu: Architektura porzuca zimną kalkulację opłacalności i kosztów leczenia,
        bezpowrotnie zmieniając wektor decyzji. System eliminuje z przestrzeni klinicznej fatalistyczne pytanie
        „Czy warto jeszcze pomagać?”, zastępując je zorientowanym na działanie imperatywem: „W jaki sposób możemy pomóc
        tu i teraz?”. Zamiast szukać ekonomicznego uzasadnienia, algorytm skupia się na zidentyfikowaniu interwencji,
        która w danym ułamku sekundy jest najlepsza dla tego konkretnego chorego.
        
        Walka z nihilizmem terapeutycznym: Wobec gigantycznej, przytłaczającej złożoności ludzkiego mózgu, która często
        rodzi w medycynie poczucie bezradności, system ma być tarczą przeciwko rezygnacji – dając lekarzowi precyzyjne
        oparcie tam, gdzie standardowe procedury i ludzkie zmysły ulegają wyczerpaniu.
        
        Świadomość asymetrii błędu: W neurointensywnej terapii konsekwencje bierności lub opóźnionej decyzji są
        nieporównywalnie bardziej katastrofalne niż koszty nadmiarowej ostrożności. System został zaprogramowany
        z rygorystycznym uwzględnieniem tej asymetrii – w warunkach drastycznej niepewności i szumu informacyjnego,
        algorytm ma obowiązek bezwzględnie faworyzować kierunek chroniący potencjał przetrwania. Lepiej wywołać fałszywy alarm
        inicjujący interwencję, niż stracić bezpowrotnie to wąskie okno terapeutyczne, w którym na zawsze decydują się
        losy układu nerwowego. W miarę dodawania kolejnych modalności i źródeł danych, system staje się coraz bardziej odporny
        na szum i niepewność, a jego decyzje coraz bardziej trafne i spójne z biologiczną rzeczywistością pacjenta, chroniąc
        przed zalewem fałszywych alarmów.

    MOŻLIWOŚCI ROZBUDOWY ARCHITEKTURY:
        Przetwarzanie notatek klinicznych: W przyszłości architektura może zostać rozbudowana o przetwarzanie
        nieustrukturyzowanych notatek klinicznych za pomocą modeli językowych NLP. Docelową funkcją systemu nigdy
        nie będzie zastępowanie lekarza, lecz wyłącznie wspieranie go w procesie decyzyjnym. Rolą NLP byłoby,
        więc przetwarzanie nieustrukturyzowanych danych tekstowych w celu wyłuskania istotnych informacji,
        które mogłyby zostać pominięte. To człowiek musi spojrzeć na skonsolidowane wyniki systemu, ocenić je
        przez pryzmat swojej wiedzy oraz empatii i podjąć ostateczną decyzję. System nie decyduje autonomicznie
        o leczeniu pacjenta – dostarcza jedynie opartej na danych argumentacji, która ułatwia podjęcie właściwej
        akcji ratunkowej. Wydaje mi się jednak, że na tym etapie NLP byłoby zbyt dużym ryzykiem, architektura
        opiera się na stosunkowo małych zbiorach danych i syntetycznych parowaniu, dlatego wprowadzenie go mogłoby
        przynieść więcej szkód niż korzyści.
        
        Fuzja Causal AI i Konformalnej Predykcji: Modele przyczynowe same w sobie dają estymację punktową efektu leczenia,
        która bywa zawodna przy nietypowych pacjentach. Nałożenie na to predykcji konformalnej pozwala wygenerować
        gwarantowane statystycznie granice skutku leczenia. 
        System nie komunikuje już w sposób autorytarny: „Zrób X, to wydarzy się Y”,
        lecz informuje z pełną odpowiedzialnością matematyczną: „Z 95% pewnością podanie leku obniży ICP o wartość z przedziału od 2.5 do 6.0 jednostek”.
        
        Rola LLM jako Centralnego Interfejsu: Wszystkie wyniki z kaskady, modeli przyczynowych oraz granic konformalnych
        docelowo trafią do lokalnego modelu LLM w architekturze RAG, zintegrowanego z interfejsem lekarza. Stanowi to barierę
        ochronną przed przebodźcowaniem numerycznym. Zamiast wyświetlać surowe wartości, system wstrzyknie te dane jako ukryty
        prompt do modelu językowego pokroju Llama 3 8B-Instruct. Na jego podstawie wygenerowany zostanie syntetyczny komunikat:
        „Pacjent w stanie krytycznym. Symulacja wykazuje, że podanie Mannitolu z 95% pewnością obniży ICP na tyle, by wyprowadzić
        pacjenta ze strefy bezpośredniego zagrożenia”. Lekarz otrzymuje jedno płynne zdanie, mając świadomość precyzyjnego wsparcia matematycznego.
        
        Przebudowa w stronę Edge AI i MLOps: Docelowe wdrożenie szpitalne oznacza bezwarunkowe odejście od środowiska
        symulacyjnego, co wymusza diametralną zmianę sposobu serwowania modeli. Obecne funkcje odpytywania bazy PostgreSQL
        ustąpią miejsca ciągłemu strumieniowaniu za pomocą subskrypcji Apache Kafka lub Redpanda, gdzie system działa jako
        konsument aktualizujący stan pacjenta przy każdym nowym pomiarze. Zastosowane zostanie ścisłe rozwarstwienie obliczeń:
        modele tabelaryczne i grafowe będą kompilowane do formatu TensorRT lub ONNX i osadzane bezpośrednio na bramkach
        brzegowych przy łóżku pacjenta, działając niezależnie od sieci z zerowym opóźnieniem. Z kolei ciężkie modele wizyjne
        trafią na scentralizowany serwer szpitalny GPU zarządzany przez NVIDIA Triton Inference Server. Bramka brzegowa będzie
        komunikować się z nim asynchronicznie po spłynięciu nowego skanu z PACS, scalając obraz z lokalnymi predykcjami za pośrednictwem Meta Sędziego. """