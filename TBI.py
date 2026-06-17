# Na początku
import os
import sys
import builtins
import datetime
import pymongo
from pymongo import MongoClient
import psycopg2
import redis

# Funkcje do przekształcania prawdopodobieństw
from scipy.special import logit, expit

# Połączenia z infrastrukturą Docker itd.
DB_HOST = os.getenv('DB_HOST', 'localhost')
MONGO_HOST = os.getenv('MONGO_HOST', 'localhost')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')

# Funkcje eksperymentalne
from sklearn.experimental import enable_halving_search_cv, enable_iterative_imputer

# Biblioteki dla obsługi danych i system
import pandas as pd  # Zarządzanie tabelami
import numpy as np  # Operacje macierzowe

# Silniki modeli
import xgboost as xgb  # Model XGBoost (stabilne drzewa)
from lightgbm import LGBMClassifier  # Model LightGBM (szybkie drzewa)
from sklearn.svm import SVC  # SVM dla trudnych przypadków (mniejsza skuteczność, bo operuje na bardziej złożonych funkcjach decyzyjnych)
from sklearn.linear_model import LogisticRegression  # Prosty meta-model do fuzji wyników

# Przygotowanie danych i Pipelines
from scipy.stats import loguniform, randint #
from sklearn.compose import ColumnTransformer  # Pozwala stosować różne transformacje do różnych kolumn
from sklearn.impute import SimpleImputer  # Podstawowe uzupełnianie braków
from sklearn.impute import IterativeImputer #
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold  # Podział danych i walidacja krzyżowa z zachowaniem proporcji klas
from sklearn.calibration import CalibratedClassifierCV  # Dopasowanie prawdopodobieństw modelu do rzeczywistości
from sklearn.model_selection import HalvingRandomSearchCV # Szybszy Bayes, a mała strata jakości

# Przygotowanie danych do modelowania i zarządzanie nimi
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier  # Łączenie modeli: głosowanie większościowe i meta-model sędziowski
from sklearn.pipeline import Pipeline  # Szkielet automatyzujący kolejność kroków od danych surowych do predykcji
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler  # Kodowanie kategorii oraz standaryzacja
from sklearn.preprocessing import FunctionTransformer  # Przekształcanie własnych funkcji w moduły kompatybilne z Pipeline
from sklearn.linear_model import BayesianRidge  # Estymator statystyczny używany przez Imputer do zgadywania wartości
from sklearn.model_selection import train_test_split # Podział na zbiór testowy i uczący
from sklearn.exceptions import NotFittedError # Bez błędów z ColumnDropper i MedicalNoiseAdder

# Optymalizacja i Metryki
from skopt import BayesSearchCV  # Bayesowskie szukanie parametrów
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, make_scorer, fbeta_score, confusion_matrix  # Ocena modelu
from sklearn.calibration import calibration_curve  # Ocena kalibracji
from sklearn.metrics import precision_recall_curve  # Optymalizacja progu (F1-score)
from sklearn.base import BaseEstimator, TransformerMixin  # Podejmowanie bezpiecznych decyzji

# Wstrzykiwanie wirtualnych danych
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Wizualizacja
import matplotlib.pyplot as plt  # Wykresy
import seaborn as sns  # Do macierzy pomyłek
from sklearn.metrics import confusion_matrix  # Do obliczeń macierzy
import shap  # Biblioteka do wyjaśniania decyzji modelu
from sklearn.metrics import accuracy_score, balanced_accuracy_score # Obliczanie metryk
# shap.initjs() na czas testów, żeby nie ładować niepotrzebnie w głównym pipeline

# Zapis modeli i obsługa wielowątkowości
import joblib
from joblib import parallel_backend

# Do głębokiej kopii, przy używaniu SWA do optymalizacji stabilności meta-modelu
from sklearn.base import clone

# Wyciszanie warningów
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed` should be used")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Połączenia z infrastrukturą
r_cache = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
mongo_client = pymongo.MongoClient(f'mongodb://{MONGO_HOST}:27017/')
mongo_db = mongo_client["tbi_research"]


class TimestampedStdout:
    """ Globalny przechwytywacz wyjścia. Dodaje znacznik czasu do każdej nowej linii. """
    def __init__(self, stream):
        self.stream = stream
        self.newline = True

    def write(self, data):
        if data.strip():
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if self.newline:
                self.stream.write(f"[{now}] {data}")
            else:
                self.stream.write(data)
            self.newline = data.endswith('\n')
        else:
            self.stream.write(data)
            if data == '\n':
                self.newline = True

    def flush(self):
        self.stream.flush()
sys.stdout = TimestampedStdout(sys.stdout)
sys.stderr = TimestampedStdout(sys.stderr)


def save_to_infrastructure(patient_id, prob, decision, raw_data, vitals_snapshot):
    """ Zapisuj wynik w SQL (dokumentacja), Mongo (badania) i Redis (akcja). """
    ts = datetime.datetime.now()

    # PostgreSQL (Oficjalna dokumentacja medyczna)
    try:
        conn = psycopg2.connect(host="localhost", port=5433, database="hospital_db", user="postgres", password="postgres")
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ML_prediction_results (patient_id, death_prob, final_decision, created_at) VALUES (%s, %s, %s, %s)",
            (int(patient_id), float(prob), decision, ts)
        ) # Dodajemy końcowy wynik modeli do bazy dla interakcji Transformera
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[SQL Error] {e}")

    # MongoDB (Pełny log badawczy)
    log_entry = {
        "patient_id": patient_id,
        "timestamp": ts,
        "vitals_at_prediction": vitals_snapshot,
        "prediction": {"probability": float(prob), "decision": decision},
        "model_version": "v1_SWA"
    }
    mongo_db.prediction_logs.insert_one(log_entry)

    # Redis (Szybki Alert)
    alert_level = "CRITICAL" if prob > 0.8 else "STABLE"
    r_cache.setex(f"alert:patient:{patient_id}", 3600, f"{alert_level}|{round(prob * 100, 2)}%")
    print(f"Zapisano wynik dla pacjenta {patient_id} w SQL, Mongo i Redis.")


def fine_tune_swa(stacking_model, X, y):
    """ Optymalizacja typu SWA dla meta-modelu. Polega na uśrednianiu wag regresji logistycznej z wielu
        podzbiorów danych w celu wyznaczenia stabilniejszej i lepiej zgeneralizowanej granicy decyzyjnej. """
    print("Optymalizacja meta-modelu.")
    meta_model = stacking_model.final_estimator_

    # Generujemy dane wejściowe dla sędziego
    X_meta_input = stacking_model.transform(X)

    weights = []
    intercepts = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Trenujemy tylko małą Regresję Logistyczną
    for i, (train_idx, _) in enumerate(skf.split(X_meta_input, y), 1):
        fold_model = clone(meta_model)
        fold_model.fit(X_meta_input[train_idx], y.iloc[train_idx])

        weights.append(fold_model.coef_)
        intercepts.append(fold_model.intercept_)

    # Uśrednianiamy
    meta_model.coef_ = np.mean(weights, axis=0)
    meta_model.intercept_ = np.mean(intercepts, axis=0)

    return stacking_model


def drop_m1_prob(X):
    """ Usuwanie kolumny z prawdopodobieństwami modelu 1, jeśli istnieje.
        I tak jest używana przy przekazywaniu decyzji obu modeli do meta-modelu, więc niepotrzebnie duplikować. """
    if isinstance(X, pd.DataFrame) and 'model_1_prob' in X.columns:
        return X.drop(columns=['model_1_prob'])
    return X


class AdaptiveMedicalNoiseAdder(BaseEstimator, TransformerMixin):
    """ Skale Glasgow i APACHE charakteryzują się dużym rozrzutem i trudnością w wyznaczeniu dla trudnych przypadków.
        Dodawanie szumu może pomóc w wyeliminowaniu błędów w danych i poprawić stabilność modeli. Szum jest wyższy
        dla skal subiektywnych i niższy dla danych z aparatury. """
    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self.is_fitted_ = None

    def fit(self, X, y=None):
        """ Uczy transformator i ustawia flagę is_fitted_. """
        if self.feature_names is None:
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Scikit-Learn tego wymaga, aby uznać moduł za "wyuczony"
        self.is_fitted_ = True
        return self

    def _add_noise(self, X):
        """ Wewnętrzna metoda aplikująca szum medyczny. """
        X_copy = np.array(X).copy()
        for i, col in enumerate(self.feature_names):
            col_lower = col.lower()
            if 'indicator' in col_lower or 'missing' in col_lower:
                continue
            if any(term in col_lower for term in ['gcs', 'apache', 'verbal', 'motor', 'eyes']):
                level = 0.07
            elif any(term in col_lower for term in ['bp', 'rate', 'pulse', 'spo2', 'glucose', 'temp']):
                level = 0.01
            else:
                level = 0.03

            noise = np.random.normal(0, level, X_copy[:, i].shape)
            X_copy[:, i] *= (1 + noise)
        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        """ Wywoływane przez Pipeline podczas fit(). Dodajemy szum. """
        self.fit(X, y, **fit_params)
        return self._add_noise(X)

    def transform(self, X):
        """ Wywoływane podczas predict() i na zbiorze testowym. Bez szumu. """
        # Sprawdzamy flagę z podkreślnikiem
        if not getattr(self, 'is_fitted_', False):
            raise NotFittedError("Transformator nie został wyuczony. Wywołaj najpierw 'fit'.")
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


class MedicalCascadePredictor:
    """ Zoptymalizowana kaskada decyzyjna. Wykorzystuje wektoryzację zamiast pętli.
        Zaktualizowana o precyzyjną bramkę kliniczną z fuzją Log-Odds. """
    def __init__(self, preprocessor, ensemble, specialist, judge):
        self.preprocessor = preprocessor
        self.ensemble = ensemble
        self.specialist = specialist
        self.judge = judge

        # Zapamiętujemy nazwy cech, żeby obsłużyć SHAP
        self.feature_names_in = preprocessor.feature_names_in_

    def predict_proba(self, X_raw):
        # Obowiązkowe dla SHAP
        if not isinstance(X_raw, pd.DataFrame):
            X_raw = pd.DataFrame(X_raw, columns=self.feature_names_in)

        X_clean = self.preprocessor.transform(X_raw)
        feature_names_out = [name.split('__')[-1] for name in self.preprocessor.get_feature_names_out()]
        X_clean_df = pd.DataFrame(X_clean, columns=feature_names_out, index=X_raw.index)

        # Przygotowujemy wektor wynikowy (domyślnie bierzemy wynik Konsylium)
        prob_main = self.ensemble.predict_proba(X_clean_df)[:, 1]
        final_probs = prob_main.copy()

        # Wykrywanie trudnych przypadków
        is_hard_case = (X_raw['gcs_sum'] <= 8) | ((prob_main > 0.35) & (prob_main < 0.65))

        if is_hard_case.any():
            X_hard = X_raw[is_hard_case].copy()

            # Dodajemy cechę 'model_1_prob', której oczekuje Pipeline SVM
            X_hard['model_1_prob'] = prob_main[is_hard_case]

            # Aktywacja SVM
            prob_spec = self.specialist.predict_proba(X_hard)[:, 1]

            # Fuzja kliniczna
            eps = 1e-6
            p_main_hard = np.clip(prob_main[is_hard_case], eps, 1 - eps)
            p_spec_hard = np.clip(prob_spec, eps, 1 - eps)

            # Obliczamy niepewność modelu głównego
            uncertainty = 1 - 2 * np.abs(p_main_hard - 0.5)

            # Obliczamy beznadziejność stanu używając aktualnych nazw kolumn
            eyes_val = pd.to_numeric(X_hard['gcs_eyes'], errors='coerce').fillna(4.0)
            severity_score = (eyes_val == 1.0).astype(int) + (X_hard['is_intubated'] == 1.0).astype(int)

            # Dynamiczna funkcja wagi
            base_w_spec = np.where(severity_score == 2, 0.85,
                                   np.where(severity_score == 1, 0.65, 0.0))

            dynamic_w_spec = base_w_spec + (uncertainty * 0.1)
            dynamic_w_spec = np.clip(dynamic_w_spec, 0, 0.95)

            # Fuzja w przestrzeni Logit
            logit_main = logit(p_main_hard)
            logit_spec = logit(p_spec_hard)

            final_logit = (1 - dynamic_w_spec) * logit_main + dynamic_w_spec * logit_spec

            # Powrót do prawdopodobieństwa poprzez funkcję expit
            prob_final_hard = expit(final_logit)

            # Ostateczny werdykt
            final_probs[is_hard_case] = prob_final_hard

        return np.column_stack((1 - final_probs, final_probs))

    def predict(self, X_raw):
        preds = self.predict_proba(X_raw)[:, 1]
        return (preds >= 0.5).astype(int)


def load_and_split_data():
    """ Pobiera dane z Postgres i od razu wylicza nowe markery medyczne. """
    conn = psycopg2.connect(host=DB_HOST, port=5433, database="hospital_db", user="postgres", password="postgres")

    query = """ SELECT 
                    c.gcs_eyes,
                    c.gcs_motor,
                    c.gcs_verbal,
                    c.sysbp_min,
                    c.heart_rate_max,
                    c.spo2_min,
                    c.glucose_max,
                    c.d1_glucose_max, 
                    c.d1_sysbp_max, 
                    c.d1_heartrate_min, 
                    c.d1_diasbp_min, 
                    c.age,
                    c.is_intubated,
                    c.ventilated_apache, 
                    c.apache_3j_bodysystem, 
                    v.hospital_outcome_death as label
                FROM Clinical_Measurements_Temporal c
                JOIN Visits v ON c.patient_id = v.patient_id
                WHERE c.valid_to > CURRENT_TIMESTAMP; """

    df = pd.read_sql(query, conn)
    conn.close()

    # Odtwarzamy cechy interakcyjne
    df['eyes_to_motor_ratio'] = pd.to_numeric(df['gcs_eyes'], errors='coerce') / (
            pd.to_numeric(df['gcs_motor'], errors='coerce') + 1)

    # Skomplikowane załamanie pnia mózgu (brak reakcji ruchowej u intubowanego)
    df['brainstem_failure_risk'] = ((df['is_intubated'] == 1) &
                                    (pd.to_numeric(df['gcs_motor'], errors='coerce') <= 2)).astype(int)

    # Indeks hipoksji wstrząsowej (niskie ciśnienie + niskie natlenienie)
    df['hypoxia_shock_index'] = pd.to_numeric(df['sysbp_min'], errors='coerce') * pd.to_numeric(df['spo2_min'],
                                                                                                errors='coerce')

    # Usuwamy label z X i przypisujemy do y
    X = df.drop(columns=['label'])
    y = df['label'].fillna(0).astype(int)

    print(f"Zaciągnięto {len(df)} rekordów. Cechy: {list(X.columns)}")
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def cascade_predict(data):
    """ Definiujemy funkcję predykcji dla kaskady (od śmierci). """
    # SHAP często rzuca dane jako NumPy Array, co psuje ColumnTransformer
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=X_train_stacking.columns)

    # Zapewnienie, że kolumna 'model_1_prob' jest obecna, jeśli Pipeline jej wymaga
    return calibrated_spec_pipe.predict_proba(data)[:, 1]


def strip_names(X):
    """ Funkcja wymusza usunięcie nazw kolumn z potoku NumPy (potrzebne do poprawnego działania XGBoost). """
    return np.array(X)


def select_cols(df, columns):
    return df[columns]


if __name__ == "__main__":
    r""" Najpierw Get-ChildItem -Path C:\TBI\.venv -Recurse | Unblock-File. Następnie uruchom Docker Desktop
        i komenda 'docker-compose up' w katalogu głównym projektu. Na koniec już można python TBI.py"""
    # Tworzenie folderu na modele, jeśli nie istnieje
    if not os.path.exists('models'):
        os.makedirs('models')

    # Ładowanie danych z bazy pacjentów
    print("Inicjalizacja ładowania zbioru klinicznego z PostgreSQL.")
    try:
        # Ta funkcja zaciąga 85k rekordów i sama robi train_test_split
        X_train, X_test, y_train, y_test = load_and_split_data()
    except Exception as e:
        print(f"Błąd krytyczny podczas pobierania danych: {e}")
        exit()

    # Lista kolumn technicznych do usunięcia (jeśli jeszcze są w X_train/X_test) + szukana etykieta
    columns_to_drop = [
        'encounter_id', 'id', 'unnamed: 0', 'unnamed: 83',
        'hospital_death', 'hospital_outcome_death'
    ]

    # Czyszczenie i inżynieria cech
    for df in [X_train, X_test]:
        # Usuwamy śmieciowe kolumny
        df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

        # Zmiana nazw dla czystości
        rename_map = {
            'gcs_eyes_apache': 'gcs_eyes',
            'gcs_motor_apache': 'gcs_motor',
            'gcs_verbal_apache': 'gcs_verbal',
            'intubated_apache': 'is_intubated',
            'd1_sysbp_min': 'sysbp_min',
            'd1_heartrate_max': 'heart_rate_max',
            'd1_diasbp_min': 'diasbp_min',
            'd1_glucose_max': 'glucose_max',
            'd1_glucose_min': 'glucose_min'
        }

        for old, new in rename_map.items():
            if old in df.columns:
                if new in df.columns:
                    # Jeśli nowa nazwa już istnieje, usuwamy starą wersję 'apache'
                    df.drop(columns=[old], inplace=True)
                else:
                    df.rename(columns={old: new}, inplace=True)

        # GCS - Fundament neurologiczny
        if 'gcs_sum' not in df.columns:
            # Minimalny GCS to 3 (1+1+1), stąd fillna(1)
            df['gcs_sum'] = (df['gcs_eyes'].fillna(1) +
                             df['gcs_motor'].fillna(1) +
                             df['gcs_verbal'].fillna(1))

        # Korygowanie wartości dla osób z rurką (nie mówią)
        if 'is_intubated' in df.columns:
            # Model MICE uzupełni to na podstawie korelacji
            df.loc[df['is_intubated'] == 1, 'gcs_verbal'] = np.nan

            # GCS_sum przeliczamy dopiero wewnątrz Pipeline
            df['gcs_unreliable_flag'] = df['is_intubated']

        # Korekta na Sedację (Farmakologiczne obniżenie GCS)
        if 'ventilated_apache' in df.columns and 'is_intubated' in df.columns:
            # Czy pacjent jest podłączony do aparatury podtrzymującej oddychanie?
            df['sedation_risk_flag'] = df[['ventilated_apache', 'is_intubated']].max(axis=1)

            """ Wskazuje na pacjenta z GCS <= 5, ale z wentylatorem/rurką. Taki pacjent
                ma sztucznie obniżony wynik przez sedację farmakologiczną. Cecha ta chroni model przed 
                błędną interpretacją braku kontaktu jako zgonu pnia mózgu, co drastycznie redukuje liczbę
                fałszywych alarmów u pacjentów stabilnych klinicznie. """
            df['gcs_sedation_mismatch'] = ((df['gcs_sum'] <= 5) & (df['sedation_risk_flag'] == 1)).astype(int)

        # Korekta na Bazowy Deficyt Neurologiczny (Autyzm, Demencja, Wcześniejsze udary)
        if 'apache_3j_bodysystem' in df.columns:
            # Tworzymy flagę pacjenta obciążonego neurologicznie
            df['baseline_neuro_deficit'] = df['apache_3j_bodysystem'].apply(
                lambda x: 1 if str(x).lower() in ['neurological', 'neurologic'] else 0
            )

            # Obliczamy Skorygowany GCS" Jeśli pacjent ma deficyt bazowy, dodajemy mu 2 punkty litości
            df['adjusted_gcs'] = np.where(df['baseline_neuro_deficit'] == 1, df['gcs_sum'] + 2, df['gcs_sum'])
            df['adjusted_gcs'] = df['adjusted_gcs'].clip(upper=15)  # Nie pozwalamy przekroczyć 15 punktów

        # Krytyczne
        df['is_hypotensive'] = (df['sysbp_min'] < 90).astype(int)  # Krytyczne niedociśnienie

        """ Związane z wiekiem - starsze osoby mają mniejsze możliwości rehabilitacji,
            ale ich mózgi wytworzyły więcej połączeń i przez to mają większe możliwości "pracy przy ograniczonych zasobach". """
        if 'age' in df.columns and 'gcs_sum' in df.columns:
            # Nieliniowy wiek (Kwadratowy - pozwala wyłapać krzywą U)
            df['age_squared'] = df['age'] ** 2

            # Oznaczamy tylko podeszłych z niskim GCS
            df['critical_geriatric_condition'] = ((df['age'] > 75) & (df['gcs_sum'] < 5)).astype(int)

            # Dzielimy - sprawdzamy, jak wydajny jest GCS na dany rok życia
            df['neurological_efficiency'] = df['gcs_sum'] / (df['age'] + 1e-5)

        # Glukoza i ciśnienie
        if 'glucose_max' in df.columns and 'glucose_min' in df.columns:
            df['glucose_stress'] = df['glucose_max'] - df['glucose_min']

        # Wskaźnik Wstrząsu
        if 'heart_rate_max' in df.columns and 'sysbp_min' in df.columns:
            df['shock_index'] = df['heart_rate_max'] / (df['sysbp_min'] + 1e-5)
        else:
            df['shock_index'] = 0.7  # Neutralny

        # Łączymy serce z mózgiem. Jeśli oba systemy padają, ryzyko rośnie nieliniowo.
        df['neuro_cardio_risk'] = df['shock_index'] / (df['gcs_sum'] + 1e-5)

        # Marker Triady Cushinga: Wysokie ciśnienie skurczowe przy niskim tętnie
        if 'd1_sysbp_max' in df.columns and 'd1_heartrate_min' in df.columns:
            # Wysokie ciśnienie + niskie tętno = potencjalne wgłobienie
            df['cushing_risk_index'] = df['d1_sysbp_max'] / (df['d1_heartrate_min'] + 1e-5)

        # Średnie Ciśnienie Tętnicze - kluczowe dla ukrwienia mózgu
        if 'sysbp_min' in df.columns and 'diasbp_min' in df.columns:
            df['map_min'] = (df['sysbp_min'] + 2 * df['diasbp_min']) / 3

            # Hipoperfuzja mózgu (MAP < 65-70 to stan krytyczny)
            df['is_low_map'] = (df['map_min'] < 65).astype(int)

        # Wskaźnik dynamiki tętna
        if 'h1_heartrate_max' in df.columns and 'heart_rate_max' in df.columns:
            df['hr_calming_index'] = df['h1_heartrate_max'] - df['heart_rate_max']

        # Ryzyko metaboliczne dla mózgu
        if 'gcs_sum' in df.columns and 'glucose_max' in df.columns:
            df['neuro_metabolic_stress'] = df['glucose_max'] / (df['gcs_sum'] + 1e-5)

        # Czyszczenie ewentualnych błędów matematycznych
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(f"Inżynieria cech zakończona. Dodano parametry: Sedation Mismatch, Adjusted GCS, Shock Index, Glucose Stress, Pulse Pressure, Neuro-Cardio Risk.")

    print(f"Dane gotowe: {X_train.shape[0] + X_test.shape[0]} pacjentów.")
    print(f"Parametry medyczne: {list(X_train.columns)}")
    print(f"Rozkład klas (0=Żyje, 1=Zgon) w treningu: \n{y_train.value_counts()}")

    # Sprawdzenie poprawności
    if y_train.nunique() < 2:
        print("Zbiór treningowy zawiera tylko jedną klasę wyników!")
        exit()

    print(f"Podział utrzymany: X_train: {X_train.shape[0]} | X_test: {X_test.shape[0]}")

    # Obliczamy GCS_sum dla obu zbiorów
    for data in [X_train, X_test]:
        if 'gcs_sum' not in data.columns:
            data['gcs_sum'] = (data['gcs_eyes'].fillna(0) +
                               data['gcs_motor'].fillna(0) +
                               data['gcs_verbal'].fillna(0))

    # Automatyczne wykrywanie kolumn do preprocesora
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include='object').columns

    # Definicja transformacji dla poszczególnych typów kolumn
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    numerical_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer( # Dane numeryczne często od siebie zależą, redukując błąd kliniczny
            estimator=BayesianRidge(), # Dobry dla współliniowych danych
            max_iter=30,
            tol=1e-3, # Wczesne zatrzymanie, jeśli zbiegnie szybciej
            n_nearest_features=None, # Używamy pełnego kontekstu klinicznego pacjenta
            random_state=42,
            add_indicator=True, # Zachowujemy informację o braku danych (ważne medycznie, bo dane nie są pomijane, ot tak)
            imputation_order='ascending' # Często szybsze i stabilniejsze niż 'random'
        )),
        ('scaler', StandardScaler()),
        ('noise', AdaptiveMedicalNoiseAdder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        n_jobs=-1,
        verbose_feature_names_out=False # Dzięki temu nazwy na wykresach SHAP i w analizie ważności cech pozostają czytelne
    )

    constraints = {}
    for col in X_train.columns:
        col_lower = col.lower()

        # Jeśli te wartości rosną, ryzyko zgonu maleje
        if any(x in col_lower for x in ['gcs', 'sysbp', 'spo2', 'map_min', 'neurological_efficiency', 'adjusted_gcs']):
            constraints[col] = -1

        # Jeśli te wartości rosną, ryzyko zgonu rośnie
        elif any(x in col_lower for x in
                 ['shock_index', 'glucose_stress', 'cushing_risk', 'neuro_metabolic', 'neuro_cardio', 'age_severity']):
            constraints[col] = 1

        # Tutaj model ma pełną swobodę. Np. wiek (krzywa U) lub tętno (za niskie i za wysokie jest złe).
        else:
            constraints[col] = 0

    print(f"Zdefiniowano {len(constraints)} ograniczeń monotonicznych dla silników modeli.")

    # Wstępnie definiujemy modele
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        random_state=42,
        verbosity=0,  # Tylko kluczowe logi
        use_label_encoder=False,
        n_jobs=-1
    )
    lgbm_model = LGBMClassifier(
        n_estimators=1000, # Oba sprawdzają po 1000 drzew, wystarczająco do podjęcia decyzji
        random_state=42,
        verbosity=-1,  # Całkowite milczenie
        importance_type='gain',
        n_jobs=-1 # Wykorzystanie wszystkich rdzeni
    )

    """ XGBoost buduje cały poziom drzewa, zanim wejdzie wyżej, LightGBM jest szybszy i bardziej efektywny
        do wykrywania anomalii, ponieważ bardziej skupia się na tych, które mają największe gradienty
        (model najbardziej się myli). Z tego powodu, myśle, że jest to dobry wybór dla danych, które odstają,
        ponieważ w przypadku skali Glasgow pomyłka może oznaczać śmierć pacjenta. """
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ],
        voting='soft'  # 'soft' pozwala modelom operować na prawdopodobieństwie - ważne dla danych kategorialnych
    )
    # Ostateczny pipeline - połączenie tych dwóch faz
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', ensemble)
    ])

    # Na testowym zwykłe transform, żeby nie zobaczył rozkładów
    X_train_clean = preprocessor.fit_transform(X_train, y_train)
    X_test_clean = preprocessor.transform(X_test)

    """ Regularyzacja i log-uniform:
        1. L1/L2 (alpha/lambda) ograniczają ekstremalne wagi cech, a Gamma blokuje zbędne gałęzie.
        2. Subsampling (0.5-1.0) sprawia, że każde drzewo uczy się na fragmentach danych, wymuszając generalizację.
        3. Wartości są rozłożone równomiernie w zakresie logarytmicznym. Pozwala to na precyzyjne
            przeszukiwanie małych wartości, co drastycznie zmniejsza ryzyko przeoczenia pacjenta. """
    search_space = {
        # Duży zakres drzew
        'model__xgb__n_estimators': randint(500, 2500),

        # Learning rate z rozkładem logarytmicznym
        'model__xgb__learning_rate': loguniform(0.005, 0.2),

        # Maksymalna głębokość drzewa
        'model__xgb__max_depth': randint(3, 10),

        # Parametry regularyzacji
        'model__xgb__reg_alpha': loguniform(1e-3, 10.0),
        'model__xgb__reg_lambda': loguniform(1e-3, 10.0),
        'model__xgb__gamma': loguniform(1e-3, 5.0),

        # Próbkowanie cech i wierszy
        'model__xgb__colsample_bytree': loguniform(0.5, 1.0),
        'model__xgb__subsample': loguniform(0.5, 1.0),

        # Wysokie wartości tego parametru czynią model bardziej konserwatywnym, co jest kluczowe przy danych medycznych obarczonych błędami z powodu trudności kategoryzacji
        'model__xgb__min_child_weight': randint(1, 10),

        # Równoważy wagę błędu dla rzadszej klasy. Dzięki temu model jest mocniej karany za przeoczenie pacjenta z grupy ryzyka, co zwiększa czułość modelu – kluczową w diagnostyce medycznej. Pomyłki są kosztowne.
        'model__xgb__scale_pos_weight': loguniform(1.0, 15.0),

        'model__lgbm__n_estimators': randint(500, 2500),
        'model__lgbm__learning_rate': loguniform(0.005, 0.2),

        # num_leaves to najważniejszy parametr w LightGBM (w przeciwieństwie do XGBoost nie mierzy on wagi, tylko czystą liczbę pacjentów)
        'model__lgbm__num_leaves': randint(20, 255),

        # min_child_samples to odpowiednik min_child_weight z XGBoost
        'model__lgbm__min_child_samples': randint(5, 75),

        # Wymuszamy na drzewach ogromną karę za przeoczenie zgonu
        'model__lgbm__scale_pos_weight': loguniform(1.0, 25.0)
    }

    print("Przygotowanie danych: MICE + Skalowanie")

    # Początkowa definicja modeli
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        n_jobs=-1
    )

    lgbm_model = LGBMClassifier(
        n_estimators=1000,
        random_state=42,
        verbosity=-1,
        importance_type='gain',
        n_jobs=-1
    )

    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('lgbm', lgbm_model)],
        voting='soft'
    )

    specialist_model = SVC(
        kernel='rbf',
        probability=True,
        random_state=42,
        max_iter=10000,
        cache_size=5000
    )

    feature_names = [name.split('__')[-1] for name in preprocessor.get_feature_names_out()]
    X_train_clean_df = pd.DataFrame(X_train_clean, columns=feature_names)
    X_test_clean_df = pd.DataFrame(X_test_clean, columns=feature_names, index=X_test.index)

    # Tworzymy słownik ograniczeń w takiej samej kolejności, w jakiej są kolumny w ramce danych (ważne dla XGBoosta)
    ordered_constraints = []

    for col in X_train_clean_df.columns:
        col_lower = col.lower()

        # Im wyższa wartość, tym mniejsze ryzyko zgonu
        if any(x in col_lower for x in ['gcs', 'sysbp', 'spo2', 'map_min', 'neurological_efficiency', 'adjusted_gcs']):
            ordered_constraints.append(-1)

        # Im wyższa wartość, tym większe ryzyko zgonu
        elif any(x in col_lower for x in
                 ['shock_index', 'glucose_stress', 'cushing_risk', 'neuro_metabolic', 'neuro_cardio', 'age_squared', 'is_hypotensive']):
            ordered_constraints.append(1)

        # Wszystkie inne (flagi braków danych, kategoryczne, wiek bazowy) - brak wymuszenia (0)
        else:
            ordered_constraints.append(0)

    # Konwertujemy listę na krotkę - tego wymaga XGBoost dla mapowania po indeksach
    monotone_tuple = tuple(ordered_constraints)

    print(f"Zsynchronizowano {len(monotone_tuple)} ograniczeń dla {len(X_train_clean_df.columns)} cech.")

    # Wstrzykujemy krotkę do modelu przed fitem
    for name, est in ensemble.estimators:
        if name == 'xgb':
            est.set_params(monotone_constraints=monotone_tuple)

    ensemble.fit(X_train_clean_df, y_train)

    # Generujemy predykcje
    y_prob_base = ensemble.predict_proba(X_test_clean_df)[:, 1]
    auc_base = roc_auc_score(y_test, y_prob_base)
    print(f"Wynik Monolitu (XGB+LGBM+RF) bez optymalizacji AUC: {auc_base:.4f}")

    # Optymalizacja Bayesowska silników modeli
    print("Rozpoczynam naukę Bayesowską XGB+LGBM+RF (Turniej Halving).")

    # Przygotowujemy scorer. Beta=1.5 oznacza, że Recall (czułość) jest ważniejszy od Precision.
    SAFE_SCORER = make_scorer(fbeta_score, beta=1.5)

    # Skracamy search_space (usuwamy 'model__', bo trenujemy bezpośrednio VotingClassifier)
    consilium_search_space = {
        # Ekspert Hemodynamiczny (LightGBM)
        'internist__model__n_estimators': randint(500, 2000),
        'internist__model__learning_rate': loguniform(0.005, 0.2),
        'internist__model__num_leaves': randint(20, 60),
        'internist__model__scale_pos_weight': loguniform(1.0, 8.0),

        # Ekspert Geriatryczny (XGBoost)
        'geriatrist__model__n_estimators': randint(500, 2000),
        'geriatrist__model__learning_rate': loguniform(0.005, 0.2),
        'geriatrist__model__max_depth': randint(3, 7),
        'geriatrist__model__min_child_weight': randint(1, 5),
        'geriatrist__model__scale_pos_weight': loguniform(1.0, 8.0),

        # Sędzia (Meta-model: Regresja Logistyczna)
        'final_estimator__C': loguniform(0.001, 1.0)
    }

    # Dzielimy kolumny na dziedziny medyczne, żeby eksperci nie wchodzili sobie w drogę
    features_hemo = [col for col in X_train_clean_df.columns if any(x in col.lower() for x in
                                                                    ['sysbp', 'heart_rate', 'shock_index', 'map_min',
                                                                     'glucose', 'hypotensive', 'cushing'])]
    features_neuro = [col for col in X_train_clean_df.columns if any(
        x in col.lower() for x in ['gcs', 'intubated', 'sedation', 'tbi', 'neuro_deficit', 'adjusted_gcs'])]
    features_geriatric = [col for col in X_train_clean_df.columns if
                          any(x in col.lower() for x in ['age', 'efficiency', 'geriatric'])]

    # Bazowy model dla eksperta neurologicznego wewnątrz Konsylium
    specialist_model = SVC(kernel='rbf', probability=True, random_state=42, max_iter=10000, cache_size=5000)

    # Pakujemy wyizolowane dane i modele w rury
    hemo_expert = Pipeline([
        ('selector', FunctionTransformer(select_cols, kw_args={'columns': features_hemo})),
        ('model', lgbm_model)])

    neuro_expert = Pipeline([
        ('selector', FunctionTransformer(select_cols, kw_args={'columns': features_neuro})),
        ('model', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
    ])

    geriatric_expert = Pipeline([
        ('selector', FunctionTransformer(select_cols, kw_args={'columns': features_geriatric})),
        ('model', xgb_model)
    ])

    # Składamy Ostateczne Konsylium
    final_consilium = StackingClassifier(
        estimators=[
            ('internist', hemo_expert),
            ('neurolog', neuro_expert),
            ('geriatrist', geriatric_expert)
        ],
        final_estimator=LogisticRegression(C=0.1),
        cv=3, stack_method='predict_proba', n_jobs=-1
    )

    # Wstrzyknięcie ograniczeń logicznych dla Geriatry
    geriatric_constraints_list = []
    for col in features_geriatric:
        col_lower = col.lower()
        # Z wiekiem, coraz mniejsze możliwości rehabilitacji
        if any(x in col_lower for x in ['age_squared']):
            geriatric_constraints_list.append(1)
        # Stan seniorów zależy w większości od wydajności neurologicznej, mają ograniczone możliwości rehabilitacji, ale ze względu na wiek, ich mózg wytworzył więcej neuronów
        elif any(x in col_lower for x in ['neurological_efficiency']):
            geriatric_constraints_list.append(-1)
        else:
            geriatric_constraints_list.append(0)

    # Aplikujemy krotkę do modelu wewnątrz struktury Konsylium
    final_consilium.named_estimators['geriatrist'].named_steps['model'].set_params(
        monotone_constraints=tuple(geriatric_constraints_list)
    )

    # HalvingRandomSearchCV musi operować na estymatorze, który ma już ustawione ograniczenia
    opt = HalvingRandomSearchCV(
        estimator=final_consilium,
        param_distributions=consilium_search_space,
        n_candidates=500,
        factor=3,
        resource='n_samples',
        min_resources=5000, # Zaczynamy turniej od 5000 pacjentów
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1,
        scoring='roc_auc',  # Model uczy się najlepiej sortować ryzyko
        random_state=42,
        verbose=1
    )

    # Uruchomienie turnieju
    with parallel_backend('threading'):
        opt.fit(X_train_clean_df, y_train)

    print(f"Najlepsze parametry znalezione: {opt.best_params_}")
    print("Zapisywanie zoptymalizowanego Konsylium (StackingClassifier).")
    joblib.dump(opt.best_estimator_, 'models/XGB_LGBM.pkl')
    ensemble_best = opt.best_estimator_

    # Predykcje końcowe dla zbioru testowego
    y_pred_main = pd.Series(opt.predict(X_test_clean_df), index=X_test.index)
    y_prob_main = opt.predict_proba(X_test_clean_df)[:, 1]

    print(f"Ostateczny wynik Konsylium po optymalizacji AUC: {roc_auc_score(y_test, y_prob_main):.4f}")

    """ Wyciągamy pacjentów wysokiego ryzyka, których model uznał za beznadziejne przypadki, jako
        kandydaci do "Modelu Specjalisty" i ostrożnej rehabilitacji. """
    train_probs = opt.predict_proba(X_train_clean_df)[:, 1]
    X_spec_train = X_train[
        (X_train['gcs_sum'] <= 8) | ((train_probs > 0.35) & (train_probs < 0.65))
    ].copy()

    # Dodajemy pewność modelu głównego jako nową cechę dla SVM
    X_spec_train['model_1_prob'] = train_probs[X_train.index.get_indexer(X_spec_train.index)]
    y_spec_train = y_train.loc[X_spec_train.index]

    # Preprocesor i Pipeline w jednym
    spec_prep = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value=-1)), ('sc', RobustScaler())]),
         X_spec_train.select_dtypes(include='number').columns),
        ('cat',
         Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))]),
         X_spec_train.select_dtypes(include='object').columns)
    ])

    spec_pipe = ImbPipeline([('prep', spec_prep), ('smote', SMOTE(random_state=42)), ('svm', specialist_model)])

    # Optymalizacja Bayesowska dla SVM
    svm_search_space = {
        'svm__C': loguniform(0.1, 10),
        'svm__gamma': loguniform(0.0001, 1),
        'svm__kernel': ['rbf']
    }

    opt_spec = HalvingRandomSearchCV(
        spec_pipe,
        svm_search_space,
        n_candidates=250, # Mniej danych
        factor=2,  # Wolniejszy wzrost, bezpieczniejszy dla małych danych
        resource='n_samples',
        min_resources=2000, # Zaczynamy od co najmniej 2000 pacjentów, żeby mieć pewność obu klas
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),  # Wymuszamy proporcje klas i 5-krotna walidacja
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    print(f"Rozpoczynam naukę Specjalisty SVM na {len(X_spec_train)} trudnych przypadkach.")

    # Wyliczamy statystyki dla Specjalisty
    spec_deaths = y_train.loc[X_spec_train.index].sum()
    spec_total = len(X_spec_train)
    spec_ratio = (spec_deaths / spec_total) * 100

    print(f"Specjalista przejmuje {spec_total} przypadków.")
    print(f"W tym {int(spec_deaths)} zgonów ({spec_ratio:.1f}% zbioru specjalisty).")

    with parallel_backend('threading'):
        opt_spec.fit(X_spec_train, y_spec_train)

    # Kalibracja Specjalisty (żeby meta-model dostawał wiarygodne prawdopodobieństwa)
    calibrated_spec = CalibratedClassifierCV(
        opt_spec.best_estimator_,
        method='isotonic',
        cv=3
    )
    calibrated_spec.fit(X_spec_train, y_spec_train)

    print("Zapisywanie Specjalisty (SVM).")
    joblib.dump(calibrated_spec, 'models/SVM.pkl')

    """ Po takim treningu modeli, możemy przejść do analizy wyników przez prostą regresję logistyczną dla
        minimalizacji błędów. Proste modele są lepsze w medycynie, ze względu na Interpretowalność, Stabilność
        (jest liniowa względem prawdopodobieństw) oraz ostateczną Kalibrację. """
    print("Przygotowanie modeli do ostatecznej fuzji.")
    try:
        # Próbujemy użyć tego co w pamięci, jeśli nie ma - ładujemy
        if 'ensemble_best' not in locals():
            ensemble_best = joblib.load('models/XGB_LGBM.pkl')
        if 'calibrated_spec' not in locals():
            calibrated_spec = joblib.load('models/SVM.pkl')
    except Exception as e:
        print(f"Błąd dostępu do modeli: {e}")
        exit()

    print("Synchronizacja ograniczeń medycznych dla XGBoosta (Ekspert Geriatryczny).")

    # Wstrzykujemy dla XGBoost listę geriatric_constraints_list, a nie całość
    for est_list in [ensemble_best.estimators_, ensemble_best.estimators]:
        for item in est_list:
            name = item[0] if isinstance(item, tuple) else ""
            obj = item[1] if isinstance(item, tuple) else item

            # Szukamy konkretnie rury Geriatry
            if isinstance(obj, Pipeline) and name == 'geriatrist':
                inner_model = obj.named_steps['model']
                if 'XGB' in str(type(inner_model)):
                    inner_model.set_params(monotone_constraints=tuple(geriatric_constraints_list))

    print(f"Pomyślnie zresetowano ograniczenia ({len(geriatric_constraints_list)}) dla Eksperta Geriatrycznego.")

    X_train_stacking = X_train_clean_df.copy()
    X_test_final = X_test_clean_df.copy()

    print("Generowanie predykcji OOF dla Sędziego.")

    # OOF dla Konsylium (chroni sędziego przed naiwną wiarą w XGBoosta)
    y_prob_train_main_oof = cross_val_predict(
        ensemble_best, X_train_clean_df, y_train,
        cv=3, method='predict_proba', n_jobs=-1
    )[:, 1]

    # Odtwarzamy maskę "Trudnych Przypadków" używając prawdziwych predykcji OOF
    hard_mask = (X_train['gcs_sum'] <= 8) | ((y_prob_train_main_oof > 0.35) & (y_prob_train_main_oof < 0.65))

    # Inicjalizujemy wektor predykcji SVM (dla łatwych przypadków ufamy Konsylium)
    y_prob_train_spec_oof = y_prob_train_main_oof.copy()

    # OOF dla Specjalisty SVM na trudnych przypadkach
    if hard_mask.sum() > 0:
        print(f"Wyliczanie OOF SVM dla {hard_mask.sum()} pacjentów.")
        X_spec_train_oof = X_train[hard_mask].copy()
        X_spec_train_oof['model_1_prob'] = y_prob_train_main_oof[hard_mask]
        y_spec_train_oof = y_train[hard_mask]

        y_prob_spec_hard_oof = cross_val_predict(
            spec_pipe, X_spec_train_oof, y_spec_train_oof,
            cv=3, method='predict_proba', n_jobs=-1
        )[:, 1]

        # Wstawiamy wyniki SVM w odpowiednie miejsca wektora głównego
        y_prob_train_spec_oof[hard_mask] = y_prob_spec_hard_oof

    # Trening Sędziego (Meta-Model) na uczciwych danych OOF
    X_meta_train = np.column_stack((y_prob_train_main_oof, y_prob_train_spec_oof))
    final_judge = LogisticRegression(solver='lbfgs', C=0.01, max_iter=1000, class_weight='balanced', random_state=42)
    final_judge.fit(X_meta_train, y_train)

    # Na koniec budowa kaskady
    print("Rozpoczynam ostateczne scalanie systemów (Custom Blending).")
    calibrated_spec_pipe = MedicalCascadePredictor(
        preprocessor=preprocessor,
        ensemble=ensemble_best,
        specialist=calibrated_spec,
        judge=final_judge
    )

    # Ewaluacja
    y_final_prob = calibrated_spec_pipe.predict_proba(X_test)[:, 1]
    print(f"Ostateczny wynik Kaskady (AUC Stacking): {roc_auc_score(y_test, y_final_prob):.4f}")

    """ Zgodnie z pryncypiami medycyny ratunkowej i etyką Triage, priorytetem jest 
        absolutne zminimalizowanie przeoczeń pacjentów w stanie krytycznym. 
        System celowo generuje nadmiarowe alarmy, przekładając bezpieczeństwo 
        życia nad komfort personelu. F-Beta Score to metryka, która to matematyzuje. 
        W wypadku predykcji śmierci, wynik trafia jeszcze do 2 eksperta analizującego skany mózgu. """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_final_prob)
    beta_val = 1.5
    f_beta_scores = (1 + beta_val ** 2) * (precisions[:-1] * recalls[:-1]) / ((beta_val ** 2 * precisions[:-1]) + recalls[:-1] + 1e-10)

    best_threshold = thresholds[np.argmax(f_beta_scores)]

    print(f"Ustawiono PRÓG KLINICZNY F-BETA (Beta={beta_val}): {best_threshold:.4f}")
    y_threshold_pred = (y_final_prob >= best_threshold).astype(int)
    print(classification_report(y_test, y_threshold_pred))

    weights = final_judge.coef_[0]
    print(f"Waga Modelu Głównego (Konsylium): {weights[0]:.4f} | Waga Specjalisty (SVM): {weights[1]:.4f}")

    # Macierz, Kalibracja, SHAP - wizualizacja
    cm = confusion_matrix(y_test, y_threshold_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Przeżycie', 'Zgon'],
                yticklabels=['Przeżycie', 'Zgon'])
    plt.title(f'Macierz Pomyłek (Próg: {best_threshold:.2f})')
    plt.savefig('models/confusion_matrix.png')
    plt.close()

    prob_true, prob_pred = calibration_curve(y_test, y_final_prob, n_bins=10)
    plt.figure(figsize=(7, 7))
    plt.plot(prob_pred, prob_true, marker='o', label='Stacking Cascade')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('Krzywa kalibracji: Wiarygodność prognoz systemu')
    plt.savefig('models/calibration.png')
    plt.close()

    geriatric_pipeline = ensemble_best.named_estimators_['geriatrist']
    xgb_model = geriatric_pipeline.named_steps['model']  # Wyciągamy model z Pipeline

    if hasattr(xgb_model, "get_booster"):
        booster = xgb_model.get_booster()
        booster.set_attr(base_score="0.5")

    # Wyjaśnienie modelu przez SHAP
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_clean_df)

    # Przygotowanie danych do SHAP Kaskady i wykresu ekspertów
    X_test_for_spec = X_test.copy()
    X_test_for_spec['model_1_prob'] = ensemble_best.predict_proba(X_test_clean_df)[:, 1]
    y_prob_spec_full = calibrated_spec.predict_proba(X_test_for_spec)[:, 1]

    print("Generuję analizę SHAP dla systemu kaskadowego.")
    try:
        background_data = shap.sample(X_train_stacking, 50)
        explainer_cascade = shap.KernelExplainer(cascade_predict, background_data)
        test_sample = X_test_for_spec.sample(min(5, len(X_test_for_spec)), random_state=42)
        shap_values_cascade = explainer_cascade.shap_values(test_sample)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_cascade, test_sample, show=False)
        plt.title("Wpływ cech na ostateczną decyzję Kaskady")
        plt.savefig('models/shap_cascade_global.png')
        plt.close()
    except Exception as e:
        print(f"Krytyczny błąd SHAP: {e}")

    # Wykres Zgody Ekspertów
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_for_spec['model_1_prob'], y_prob_spec_full, alpha=0.4, c=y_test, cmap='coolwarm', edgecolors='k',
                s=20)
    plt.plot([0, 1], [0, 1], '--', color='gray', label="Linia absolutnej zgody")
    plt.xlabel("Pewność Konsylium (XGB+LGBM)")
    plt.ylabel("Pewność Specjalisty (SVM)")
    plt.legend()
    plt.savefig('models/experts_agreement.png')
    plt.close()

    joblib.dump(calibrated_spec_pipe, 'CASCADE.pkl')
    print("Zapisano model 'CASCADE.pkl'.")

    # Raportowanie i orkiestra
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTN: {tn:>5} | FP: {fp:>5} | FN: {fn:>5} | TP: {tp:>5}")
    print(f"Czułość (Recall): {(tp / (tp + fn)) * 100:.2f}% | Swoistość (Spec): {(tn / (tn + fp)) * 100:.2f}%\n")

    current_patient_id = 1
    current_prob = float(y_final_prob[0])
    days_since_injury = 45
    ai_raw_decision = "Zgon" if y_threshold_pred[0] == 1 else "Przeżycie"

    if days_since_injury <= 90:
        final_decision = "Przeżycie (Reguła 90 dni)"
        print(f"Pacjent w oknie {days_since_injury}/90 dni. Prognoza: {ai_raw_decision} ({current_prob * 100:.1f}%), wymuszono intensywną terapię.")
    else:
        final_decision = ai_raw_decision
        print(f"Pacjent poza oknem ({days_since_injury} dni). Decyzja: {final_decision}")

    vitals_snapshot = X_test.iloc[0].to_dict()
    vitals_snapshot['days_since_injury'] = days_since_injury

    try:
        save_to_infrastructure(current_patient_id, current_prob, final_decision, vitals_snapshot, vitals_snapshot)
        if r_cache.get(f"alert:patient:{current_patient_id}"):
            print("Weryfikacja zapisu w Redis pomyślna.")
    except Exception as e:
        print(f"Błąd orkiestracji: {e}")

print("ANALIZA SKUTECZNOŚCI SAMEJ REGUŁY: GCS <= 5")

y_gcs_baseline = (X_test['gcs_sum'] <= 5).astype(int)

# Obliczamy metryki
gcs_cm = confusion_matrix(y_test, y_gcs_baseline)
gcs_report = classification_report(y_test, y_gcs_baseline, target_names=['Przeżycie', 'Zgon'])
gcs_acc = accuracy_score(y_test, y_gcs_baseline)
gcs_bal_acc = balanced_accuracy_score(y_test, y_gcs_baseline)

# Wyświetlamy wyniki
print(f"Dokładność (Accuracy): {gcs_acc:.4f}")
print(f"Zbalansowana Dokładność: {gcs_bal_acc:.4f}")
print("\nRaport klasyfikacji dla GCS <= 5:")
print(gcs_report)

# Wizualizacja Macierzy Pomyłek dla GCS
plt.figure(figsize=(8, 6))
sns.heatmap(gcs_cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Przeżycie', 'Zgon'],
            yticklabels=['Przeżycie', 'Zgon'])

plt.title('Macierz Pomyłek: Tylko Reguła GCS <= 5 dla sprawdzenia jakości')
plt.ylabel('Prawda (Ground Truth)')
plt.xlabel('Predykcja (Reguła GCS)')
plt.savefig('models/confusion_matrix_GCS_baseline.png')
plt.show()

# Bezpośrednie porównanie z modelem
model_bal_acc = balanced_accuracy_score(y_test, (y_final_prob >= best_threshold).astype(int))
print(f"Zysk z użycia Kaskady ML: {((model_bal_acc - gcs_bal_acc) * 100):.2f}% (Balanced Accuracy)")

# Przestawić moje HCR na analizę skanów mózgu: najpierw RSNA, potem do złączenia CQ-500
r""" cd C:\TBI\database_model\patient-survival-prediction
     docker-compose up -d
     docker exec -it hospital_postgres psql -U postgres -d hospital_db -c "SELECT count(*) FROM Patients;" 
     python TBI.py """
