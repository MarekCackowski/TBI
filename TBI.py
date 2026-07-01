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
from scipy.stats import loguniform, randint # Podstawowe funkcje do definiowania przestrzeni poszukiwań hiperparametrów
from sklearn.compose import ColumnTransformer  # Pozwala stosować różne transformacje do różnych kolumn
from sklearn.impute import SimpleImputer  # Podstawowe uzupełnianie braków
from sklearn.impute import IterativeImputer  # Uzupełnianie braków na podstawie innych cech
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
from sklearn.frozen import FrozenEstimator # Estymator zapobiegający ponownemu trenowaniu już wytrenowanego modelu

# Optymalizacja i Metryki
from skopt import BayesSearchCV  # Bayesowskie szukanie parametrów
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, make_scorer, fbeta_score, confusion_matrix  # Ocena modelu
from sklearn.calibration import calibration_curve  # Ocena kalibracji
from sklearn.metrics import precision_recall_curve  # Optymalizacja progu (F1-score)
from sklearn.base import BaseEstimator, TransformerMixin  # Podejmowanie bezpiecznych decyzji

# Wizualizacja
import matplotlib.pyplot as plt  # Wykresy
import seaborn as sns  # Do macierzy pomyłek
from sklearn.metrics import confusion_matrix  # Do obliczeń macierzy
import shap  # Biblioteka do wyjaśniania decyzji modelu
from sklearn.metrics import accuracy_score, balanced_accuracy_score # Obliczanie metryk
# shap.initjs() na czas testów, żeby nie ładować niepotrzebnie w głównym pipeline

# Wstrzykiwanie wirtualnych danych
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

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
DB_HOST = os.getenv('DB_HOST', 'localhost')
MONGO_HOST = os.getenv('MONGO_HOST', 'localhost')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
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


def train_specialist_with_weights(X_train, y_train, konsylium_oof, lower_bound=0.25):
    """ Kalkulacja wag dla Specjalisty i trening.
        Większa waga dla pacjentów podwyższonego ryzyka. """
    # Definicja maski dla trudnych/podejrzanych przypadków (odrzucamy tylko ewidentnie bezpiecznych)
    uncertain_mask = (konsylium_oof > lower_bound)

    # Inicjalizacja bazowych wag dla całego zbioru
    sample_weights = np.ones(len(y_train))

    # Łagodniejsze zwiększenie wagi dla pacjentów ze strefy ryzyka
    sample_weights[uncertain_mask] = 1.5

    print(f"Liczba pacjentów z priorytetem dla Specjalisty (waga 1.5): {np.sum(uncertain_mask)}")

    # Inicjalizacja i trening modelu SVM z uwzględnieniem wag oraz wbudowanego balansu
    svm_specialist = SVC(
        kernel='rbf',
        probability=True,
        random_state=42,
        max_iter=10000,
        cache_size=5000,
        class_weight='balanced'
    )
    svm_specialist.fit(X_train, y_train, sample_weight=sample_weights)

    return svm_specialist

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


class MedicalCascadePredictor(BaseEstimator, TransformerMixin):
    """ 4-stopniowa kaskada decyzyjna: Konsylium -> Specjalista SVM -> Ekspert Geriatra -> Sędzia
        Wzbogacona o kontekst pacjenta (Wiek, GCS, MAP) na etapie fuzji. """
    def __init__(self, preprocessor, ensemble, specialist, geriatrist, judge):
        self.preprocessor = preprocessor
        self.ensemble = ensemble
        self.specialist = specialist
        self.geriatrist = geriatrist  # Dodano Eksperta Geriatrycznego
        self.judge = judge
        self.feature_names_in_ = preprocessor.feature_names_in_

    def predict_proba(self, X_raw, img_prob=None):
        if not isinstance(X_raw, pd.DataFrame):
            X_raw = pd.DataFrame(X_raw, columns=self.feature_names_in_)

        # Oczyszczanie i predykcja (dane tabelaryczne)
        X_clean = self.preprocessor.transform(X_raw)
        feature_names_out = [name.split('__')[-1] for name in self.preprocessor.get_feature_names_out()]
        X_clean_df = pd.DataFrame(X_clean, columns=feature_names_out, index=X_raw.index)

        # Konsylium bazowe
        prob_main = self.ensemble.predict_proba(X_clean_df)[:, 1]
        prob_spec = prob_main.copy()
        prob_ger = prob_main.copy()

        # Specjalista SVM (Wykrywanie stanu krytycznego)
        is_hard_case = (X_raw['gcs_sum'] <= 8) | (prob_main >= 0.50)
        if is_hard_case.any():
            X_hard = X_raw[is_hard_case].copy()
            X_hard['model_1_prob'] = prob_main[is_hard_case]
            prob_spec[is_hard_case] = self.specialist.predict_proba(X_hard)[:, 1]

        # Wyodrębnienie kontekstu klinicznego dla Sędziego
        context_age = X_clean_df['age'].values if 'age' in X_clean_df.columns else np.zeros(len(X_raw))
        context_gcs = X_clean_df['gcs_sum'].values if 'gcs_sum' in X_clean_df.columns else np.zeros(len(X_raw))
        context_map = X_clean_df['map_min'].values if 'map_min' in X_clean_df.columns else np.zeros(len(X_raw))

        # Budowa macierzy meta-cech dla Sędziego z wymaganym kontekstem i miejscem na Obrazy
        if img_prob is not None:
            if len(img_prob) != len(prob_main):
                raise ValueError("Rozmiar img_prob musi być zgodny z liczbą pacjentów!")
            X_meta = np.column_stack((prob_main, prob_spec, prob_ger, img_prob, context_age, context_gcs, context_map))
        else:
            X_meta = np.column_stack((prob_main, prob_spec, prob_ger, np.full_like(prob_main, 0.5), context_age, context_gcs, context_map))

        final_probs = self.judge.predict_proba(X_meta)[:, 1]
        return np.column_stack((1 - final_probs, final_probs))

    def predict(self, X_raw, img_prob=None):
        preds = self.predict_proba(X_raw, img_prob=img_prob)[:, 1]
        return (preds >= 0.5).astype(int)


def cascade_predict(data):
    """ Przyjmuje surowe/przetworzone cechy i zwraca prawdopodobieństwo końcowe kaskady. """
    # SHAP rzuca dane jako NumPy array, przywracamy strukturę DataFrame z poprawnymi nazwami cech
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=X_train.columns)

    return calibrated_spec_pipe.predict_proba(data)[:, 1]


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
    df['hypoxia_shock_index'] = pd.to_numeric(df['sysbp_min'], errors='coerce') * pd.to_numeric(df['spo2_min'], errors='coerce')

    # Usuwamy label z X i przypisujemy do y
    X = df.drop(columns=['label'])
    y = df['label'].fillna(0).astype(int)

    print(f"Zaciągnięto {len(df)} rekordów. Cechy: {list(X.columns)}")
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


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

        # Izolacja komponentu ruchowego na bazie Majdan et al.
        if 'gcs_motor' in df.columns:
            # 1 - Brak reakcji, 2 - Wyprost, 3 - Zgięcie patologiczne
            df['critical_motor_deficit'] = (df['gcs_motor'] <= 3).astype(int)

            # Jak duży jest udział ruchu w całym wyniku? (Rozbicie iluzji sumy GCS)
            df['motor_to_gcs_ratio'] = df['gcs_motor'] / (df['gcs_sum'] + 1e-5)

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

    base_svc = SVC(
        kernel='rbf',
        probability=True,
        random_state=42,
        max_iter=10000,
        cache_size=5000
    )

    specialist_model = CalibratedClassifierCV(base_svc, ensemble=False)

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
    print(f"Wynik XGB+LGBM+RF bez optymalizacji AUC: {auc_base:.4f}")

    # Optymalizacja Bayesowska silników modeli
    print("Rozpoczynam naukę Bayesowską XGB+LGBM+RF przez Turniej Halving.")
    consilium_path = 'models/XGB_LGBM_RF.pkl'

    if os.path.exists(consilium_path):
        print(f"[CACHE] Znaleziono zapisany model Konsylium. Pomijam turniej i ładuję: {consilium_path}")
        ensemble_best = joblib.load(consilium_path)
    else:

        # Przygotowujemy scorer. Beta=1.5 oznacza, że Recall jest ważniejszy od Precision.
        SAFE_SCORER = make_scorer(fbeta_score, beta=1.5)

        # Skracamy search_space do samych silników, dostosowując nazewnictwo
        consilium_search_space = {
            # LightGBM - zawężamy wokół sprawdzonych wartości
            'lgbm__n_estimators': randint(900, 1500),  # wokół 1185
            'lgbm__learning_rate': loguniform(0.001, 0.015),  # przesunięte w dół (bliżej podłogi)
            'lgbm__num_leaves': randint(40, 65),  # wokół 50
            'lgbm__scale_pos_weight': loguniform(4.0, 9.0),  # wokół 6.8

            # XGBoost - stabilizujemy wokół optimum
            'xgb__n_estimators': randint(900, 1500),  # wokół 1117
            'xgb__learning_rate': loguniform(0.001, 0.015),  # przesunięte w dół
            'xgb__max_depth': randint(5, 8),  # wokół 6
            'xgb__min_child_weight': randint(3, 6),  # wokół 4
            'xgb__scale_pos_weight': loguniform(1.2, 2.5),  # wokół 1.8

            # RandomForest - zwrot w stronę mniejszych zasobów
            'rf__n_estimators': randint(100, 250),  # wokół 135
            'rf__max_depth': randint(10, 15)  # wokół 13
        }

        # Konsylium (wszyscy widzą całe X_train_clean_df)
        geriatrist_model = xgb.XGBClassifier(
            n_estimators=750,
            max_depth=4,
            learning_rate=0.01,
            random_state=42,
            n_jobs=-1
        )

        final_consilium = VotingClassifier(
            estimators=[
                ('lgbm', lgbm_model),
                ('rf', RandomForestClassifier(n_jobs=-1, random_state=42)),
                ('xgb', xgb_model)
            ],
            voting='soft'
        )

        # Wstrzyknięcie pełnej, 57-elementowej krotki ograniczeń monotonicznych
        final_consilium.named_estimators['lgbm'].set_params(monotone_constraints=monotone_tuple)
        final_consilium.named_estimators['xgb'].set_params(monotone_constraints=monotone_tuple)

        # Uruchomienie turnieju dla Etapu 1
        opt = HalvingRandomSearchCV(
            estimator=final_consilium,
            param_distributions=consilium_search_space,
            n_candidates=500,
            factor=3,
            resource='n_samples',
            min_resources=5000,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            n_jobs=-1,
            scoring='roc_auc',
            random_state=42,
            verbose=1
        )

        with parallel_backend('threading'):
            opt.fit(X_train_clean_df, y_train)

        print(f"Najlepsze parametry Konsylium znalezione: {opt.best_params_}")
        joblib.dump(opt.best_estimator_, 'models/XGB_LGBM_RF.pkl')
        ensemble_best = opt.best_estimator_

        # Predykcje końcowe dla zbioru testowego
        y_pred_main = pd.Series(opt.predict(X_test_clean_df), index=X_test.index)
        y_prob_main = opt.predict_proba(X_test_clean_df)[:, 1]

        print(f"Ostateczny wynik Konsylium po optymalizacji AUC: {roc_auc_score(y_test, y_prob_main):.4f}")

    # Wyznaczamy pacjentów w stanie krytycznym i filtrujemy zbiór dla Specjalisty
    train_probs_main = ensemble_best.predict_proba(X_train_clean_df)[:, 1]

    # Definiujemy stan krytyczny (Konsylium daje ryzyko zgonu >= 50% lub twardy stan kliniczny GCS <= 8)
    critical_mask = (X_train['gcs_sum'] <= 8) | (train_probs_main >= 0.50)

    print(f"Liczba pacjentów przekazanych do Specjalisty: {np.sum(critical_mask)} z {len(X_train)}")

    SVM_PATH = 'models/SVM.pkl'
    if os.path.exists(SVM_PATH):
        print(f"[CACHE] Znaleziono zapisany model Specjalisty SVM. Pomijam turniej i ładuję: {SVM_PATH}")
        calibrated_spec = joblib.load(SVM_PATH)
    else:
        # Tworzymy zbiór treningowy tylko dla tych trudnych przypadków
        X_train_for_spec = X_train[critical_mask].copy()
        y_train_spec = y_train[critical_mask]

        # Dodajemy pewność modelu głównego jako nową cechę
        X_train_for_spec['model_1_prob'] = train_probs_main[critical_mask]

        # Preprocesor i standardowy Pipeline (SVM ma wbudowane class_weight='balanced')
        spec_prep = ColumnTransformer([
            ('num', Pipeline([
                ('imp', SimpleImputer(strategy='constant', fill_value=-1)),
                ('sc', RobustScaler()),
                ('noise', AdaptiveMedicalNoiseAdder())
            ]), X_train_for_spec.select_dtypes(include='number').columns),
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('oh', OneHotEncoder(handle_unknown='ignore'))
            ]), X_train_for_spec.select_dtypes(include='object').columns)
        ])

        specialist_model = SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            max_iter=10000,
            cache_size=5000,
            class_weight='balanced'
        )

        spec_pipe = Pipeline([
            ('prep', spec_prep),
            ('svm', specialist_model)
        ])

        print("Rozpoczynam naukę Specjalisty SVM na podzbiorze krytycznym.")

        # Filtrowanie pacjentów trudnych
        uncertain_mask_spec = (X_train_for_spec['model_1_prob'] > 0.25) & (X_train_for_spec['model_1_prob'] < 0.75)
        X_uncertain = X_train_for_spec[uncertain_mask_spec]
        y_uncertain = y_train_spec[uncertain_mask_spec]

        # Bierzemy wszystkich pacjentów krytycznych 3 razy, a tych najbardziej niepewnych dorzucamy jeszcze 2 razy.
        X_train_augmented = pd.concat([X_train_for_spec] * 3 + [X_uncertain] * 2, ignore_index=True)
        y_train_augmented = pd.concat([y_train_spec] * 3 + [y_uncertain] * 2, ignore_index=True)
        print(f"Rozmiar zbioru po fizycznym balansowaniu szarej strefy: {len(X_train_augmented)}")

        # Przestrzeń hiperparametrów do nauki SVM
        svm_search_space = {
            'svm__C': loguniform(0.01, 100.0),
            'svm__gamma': loguniform(0.00001, 10.0),
            'svm__kernel': ['rbf', 'poly', 'sigmoid'],
            'svm__degree': randint(2, 5),
            'svm__class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 5}]
        }

        # Obliczanie bezpiecznych zasobów startowych dla turnieju Halving
        dynamic_min_resources = max(500, int(np.sum(critical_mask) * 0.15))

        # Turniej Halving, żeby wybrać jak najlepsze parametry
        opt_spec = HalvingRandomSearchCV(
            spec_pipe,
            svm_search_space,
            n_candidates=300,
            factor=2,
            resource='n_samples',
            min_resources=dynamic_min_resources,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        with parallel_backend('threading'):
            # Czysty FIT bez parametru 'sample_weight' - model sam się dostosuje dzięki fizycznym klonom
            opt_spec.fit(X_train_augmented, y_train_augmented)

        print("Rozpoczynam automatyczną kalibrację prawdopodobieństw Specjalisty (CV=5).")
        calibrated_spec = CalibratedClassifierCV(
            estimator=opt_spec.best_estimator_,
            method='isotonic',
            cv=5
        )

        # Czysta kalibracja na zbalansowanym fizycznie zbiorze
        calibrated_spec.fit(X_train_augmented, y_train_augmented)

        print("Zapisywanie Specjalisty (SVM).")
        joblib.dump(calibrated_spec, SVM_PATH)

    print("Przygotowanie modeli do ostatecznej fuzji.")
    try:
        if 'ensemble_best' not in locals():
            ensemble_best = joblib.load('models/XGB_LGBM.pkl')
        if 'calibrated_spec' not in locals():
            calibrated_spec = joblib.load('models/SVM.pkl')
    except Exception as e:
        print(f"Błąd dostępu do modeli: {e}")
        exit()

    print("Synchronizacja ograniczeń medycznych dla XGBoosta (Ekspert Geriatryczny).")

    # Zabezpieczenie: jeśli zmienna nie istnieje w pamięci, przypisz jej listę
    if 'geriatric_constraints_list' not in locals() and 'geriatric_constraints_list' not in globals():
        geriatric_constraints_list = []
        for col in X_train_clean_df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in
                   ['gcs', 'sysbp', 'spo2', 'map_min', 'neurological_efficiency', 'adjusted_gcs', 'mobility_score',
                    'bmi_score']):
                geriatric_constraints_list.append(-1)
            elif any(x in col_lower for x in
                     ['age', 'charlson', 'cci', 'comorbidity', 'frailty', 'shock_index', 'is_hypotensive',
                      'polypharmacy']):
                geriatric_constraints_list.append(1)
            else:
                geriatric_constraints_list.append(0)

        geriatrist_model = xgb.XGBClassifier(
            n_estimators=750, max_depth=4, learning_rate=0.01,
            monotone_constraints=tuple(geriatric_constraints_list),
            scale_pos_weight=3.0, random_state=42, verbosity=0, n_jobs=-1
        )
        geriatrist_model.fit(X_train_clean_df, y_train)

        print("Generowanie predykcji OOF dla Sędziego.")

        # Wyciągnięcie przewidywań z trzech niezależnych źródeł
        y_prob_train_main_oof = \
        cross_val_predict(ensemble_best, X_train_clean_df, y_train, cv=3, method='predict_proba', n_jobs=-1)[:, 1]
        y_prob_train_ger_oof = \
        cross_val_predict(geriatrist_model, X_train_clean_df, y_train, cv=3, method='predict_proba', n_jobs=-1)[:, 1]

        y_prob_train_spec_oof = y_prob_train_main_oof.copy()
        hard_mask_oof = (X_train['gcs_sum'] <= 8) | ((y_prob_train_main_oof > 0.15) & (y_prob_train_main_oof < 0.85))

        if hard_mask_oof.sum() > 0:
            X_spec_train_oof = X_train[hard_mask_oof].copy()
            X_spec_train_oof['model_1_prob'] = y_prob_train_main_oof[hard_mask_oof]
            y_prob_train_spec_oof[hard_mask_oof] = \
            cross_val_predict(opt_spec.best_estimator_, X_spec_train_oof, y_train[hard_mask_oof], cv=3,
                              method='predict_proba', n_jobs=-1)[:, 1]

        # Zbieranie kontekstu medycznego pacjenta dla sędziego
        ctx_age = X_train_clean_df['age'].values if 'age' in X_train_clean_df.columns else np.zeros(len(X_train))
        ctx_gcs = X_train_clean_df['gcs_sum'].values if 'gcs_sum' in X_train_clean_df.columns else np.zeros(
            len(X_train))
        ctx_map = X_train_clean_df['map_min'].values if 'map_min' in X_train_clean_df.columns else np.zeros(
            len(X_train))

        # Pusta przestrzeń dla przyszłego Transformera Obrazowego
        ctx_img_placeholder = np.full_like(y_prob_train_main_oof, 0.5)

        # Nauka Sędziego na spójnej, 7-kolumnowej matrycy OOF
        X_meta_train = np.column_stack((
            y_prob_train_main_oof, y_prob_train_spec_oof, y_prob_train_ger_oof,
            ctx_img_placeholder, ctx_age, ctx_gcs, ctx_map
        ))
        final_judge = LogisticRegression(solver='lbfgs', C=0.01, max_iter=2500, class_weight='balanced',
                                         random_state=42)
        final_judge.fit(X_meta_train, y_train)

        print("Rozpoczynam ostateczne scalanie systemów (Context-Aware Blending).")

        # Inicjalizacja pełnej kaskady
        calibrated_spec_pipe = MedicalCascadePredictor(
            preprocessor=preprocessor,
            ensemble=ensemble_best,
            specialist=calibrated_spec,
            geriatrist=geriatrist_model,
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

    # Wyciągamy model bezpośrednio z Konsylium (brak wcześniejszej izolacji 'geriatrist')
    xgb_model = ensemble_best.named_estimators_['xgb']

    if hasattr(xgb_model, "get_booster"):
        booster = xgb_model.get_booster()
        # Wymuszamy nadpisanie base_score na czysty string reprezentujący float bez nawiasów
        try:
            import json

            config = json.loads(booster.save_config())
            # Wyciągamy rzeczywisty base_score z konfiguracji XGBoosta, czyszcząc nawiasy jeśli istnieją
            raw_score = config["learner"]["learner_model_param"]["base_score"]
            clean_score = raw_score.replace('[', '').replace(']', '')
            booster.set_attr(base_score=str(float(clean_score)))
        except Exception:
            # Rezerwowy fallback, jeśli struktura configu w danej wersji xgb się różni
            booster.set_attr(base_score="0.5")

    # Wyjaśnienie modelu przez SHAP dla samego XGBoosta
    print("Generuję wykres SHAP dla silnika XGBoost.")
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test_clean_df)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_clean_df, show=False)
        plt.title("Istotność cech w silniku XGBoost (Konsylium)")
        plt.tight_layout()
        plt.savefig('models/shap_xgb_global.png')
        plt.close()
        print("Wykres SHAP dla XGBoost wygenerowany pomyślnie.")
    except Exception as e:
        print(f"Błąd SHAP dla XGBoost: {e}")

    # Raportowanie i orkiestra
    tn, fp, fn, tp = cm.ravel()
    print(f"TN: {tn:>5} | FP: {fp:>5} | FN: {fn:>5} | TP: {tp:>5}")
    print(f"Czułość: {(tp / (tp + fn)) * 100:.2f}% | Swoistość (Spec): {(tn / (tn + fp)) * 100:.2f}%\n")

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
print("Raport klasyfikacji dla GCS <= 5:")
print(gcs_report)

# Wizualizacja Macierzy Pomyłek dla GCS
plt.figure(figsize=(8, 6))
sns.heatmap(gcs_cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Przeżycie', 'Zgon'],
            yticklabels=['Przeżycie', 'Zgon'])

plt.title('Macierz Pomyłek: Tylko Reguła GCS <= 5 dla sprawdzenia jakości')
plt.ylabel('Prawda')
plt.xlabel('Predykcja')
plt.savefig('models/confusion_matrix_GCS_baseline.png')
plt.show()

# Bezpośrednie porównanie z modelem
model_bal_acc = balanced_accuracy_score(y_test, (y_final_prob >= best_threshold).astype(int))
print(f"Zysk z użycia Kaskady ML: {((model_bal_acc - gcs_bal_acc) * 100):.2f}%")

# Przestawić moje HCR na analizę skanów mózgu: najpierw RSNA, potem do złączenia CQ-500
r""" cd C:\TBI\database_model
     docker-compose up -d
     docker exec -it tbi_postgres psql -U postgres -d hospital_db -c "SELECT count(*) FROM Patients;"
     ..\.venv\Scripts\python.exe TBI.py """
