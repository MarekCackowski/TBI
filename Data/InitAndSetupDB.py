import os
import logging

from minio import Minio
from minio.error import S3Error
import zipfile
import subprocess
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import time
from sqlalchemy.exc import OperationalError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ładowanie zmiennych środowiskowych
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_HOST = os.getenv("DB_HOST", "tbi_postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "tbi_database")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")

# Określenie sterownika psycopg2 w adresie URI
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_size=5,          # Utrzymuje 5 stałych połączeń do bazy
    max_overflow=10,      # Pozwala na dodatkowe 10 połączeń w szczycie
    pool_pre_ping=True    # Sprawdza, czy połączenie z chmurą nie zostało zerwane przed wykonaniem zapytania
)

train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
sql_path = os.path.join(BASE_DIR, "setup_db.sql")


class ComprehensiveDataQualitySuite:
    """ Pełny zestaw testów jakości danych. """
    def __init__(self, df: pd.DataFrame, dataset_name: str = "train"):
        self.df = df
        self.dataset_name = dataset_name
        self.errors = []
        self.warnings = []

    def run_all_checks(self):
        logger.info(f"Uruchomienie pełnych testów jakości dla: {self.dataset_name} (Wierszy: {len(self.df)})")
        
        self._check_schema_and_types()
        self._check_uniqueness()
        self._check_missing_values()
        self._check_value_ranges()
        self._check_logical_constraints()
        
        if self.errors:
            error_msg = f"Krytyczne błędy w {self.dataset_name}]:\n" + "\n".join(self.errors)
            raise ValueError(error_msg)
        
        if self.warnings:
            logger.info(f"Ostrzeżenia o jakości: \n" + "\n".join(self.warnings))
            
        logger.info(f"Wszystkie pełne testy jakości dla {self.dataset_name} przeszły pomyślnie.")

    def _check_schema_and_types(self):
        expected_types = {
            'patient_id': ['int64', 'float64', 'Int64'],
            'encounter_id': ['int64', 'float64', 'Int64'],
            'age': ['int64', 'float64', 'Int64']
        }
        for col, allowed_types in expected_types.items():
            if col in self.df.columns and str(self.df[col].dtype) not in allowed_types:
                self.errors.append(f"Kolumna '{col}' ma nieprawidłowy typ: {self.df[col].dtype} (oczekiwano: {allowed_types})")

    def _check_uniqueness(self):
        if 'encounter_id' in self.df.columns:
            duplicates = self.df['encounter_id'].duplicated().sum()
            if duplicates > 0:
                self.errors.append(f"Wykryto {duplicates} zduplikowanych wizyt ('encounter_id'). Klucz główny musi być unikalny!")

    def _check_missing_values(self):
        critical_cols = ['patient_id', 'encounter_id']
        for col in critical_cols:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                self.errors.append(f"Kluczowa kolumna '{col}' zawiera wartości NULL.")
        
        for col in self.df.columns:
            missing_pct = self.df[col].isnull().mean()
            if missing_pct > 0.5:
                self.warnings.append(f"Kolumna '{col}' ma wysoki odsetek braków danych: {missing_pct*100:.1f}%")

    def _check_value_ranges(self):
        ranges = {
            'age': (0, 120),
            'gcs_eyes_apache': (1, 4),
            'gcs_motor_apache': (1, 6),
            'gcs_verbal_apache': (1, 5),
            'd1_sysbp_min': (0, 300),
            'heart_rate_max': (0, 300),
            'spo2_min': (0, 100),
            'glucose_max': (0, 2000)
        }
        for col, (min_val, max_val) in ranges.items():
            if col in self.df.columns:
                invalid = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)].shape[0]
                if invalid > 0:
                    self.warnings.append(f"Kolumna '{col}' zawiera {invalid} wartości poza standardowym zakresem klinicznym [{min_val}, {max_val}].")

    def _check_logical_constraints(self):
        if 'glucose_min' in self.df.columns and 'glucose_max' in self.df.columns:
            invalid_glucose = (self.df['glucose_min'] > self.df['glucose_max']).sum()
            if invalid_glucose > 0:
                self.errors.append(f"Błąd logiki biznesowej: Znaleziono {invalid_glucose} rekordów, gdzie `glucose_min` > `glucose_max`.")


# Zmiana dla pliku: InitAndSetupDB_2.py
def download_kaggle_patient_survival_data():
    """ Pobiera pliki treningowe i testowe z Kaggle za pomocą szybkiego aria2c. """
    if os.path.exists(train_path) and os.path.exists(test_path):
        logger.info("Pliki train.csv oraz test.csv są obecne lokalnie. Pomijam pobieranie.")
        return

    logger.info("Brak lokalnych plików danych. Rozpoczynam pobieranie z Kaggle przez aria2c.")
    
    kaggle_user = os.getenv("KAGGLE_USERNAME")
    kaggle_token = os.getenv("KAGGLE_API_TOKEN") or os.getenv("KAGGLE_KEY")
    
    if not kaggle_user or not kaggle_token:
        logger.error("Brak danych KAGGLE_USERNAME oraz KAGGLE_API_TOKEN w pliku .env!")
        exit(1)
    
    try:
        import requests
        r = requests.get("https://www.kaggle.com/api/v1/competitions/data/download-all/patient-survival-prediction", auth=(kaggle_user, kaggle_token), allow_redirects=False)
        r.raise_for_status() # Wymuszenie błędu HTTP, jeśli status to 4xx lub 5xx
        download_url = r.headers.get("Location")
        
        if not download_url:
            logger.error("Nie otrzymano poprawnego adresu przekierowania z Kaggle API.")
            exit(1)
        
        zip_filepath = os.path.join(BASE_DIR, "patient_survival.zip")
        subprocess.run([
            "aria2c", "--conf-path=/app/aria2.conf", 
            "-d", BASE_DIR, "-o", "patient_survival.zip", download_url
        ], check=True)
        
        if os.path.exists(zip_filepath):
            logger.info("Rozpakowywanie archiwum patient_survival.zip.")
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                zip_ref.extractall(BASE_DIR)
            os.remove(zip_filepath)
            logger.info("Dane zostały pobrane i rozpakowane przez aria2c.")
            
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Błąd sieciowy podczas komunikacji z Kaggle API: {req_err}")
        exit(1)
    except (subprocess.SubprocessError, zipfile.BadZipFile, OSError) as e:
        logger.error(f"Nie udało się pobrać lub rozpakować danych przez aria2c: {e}")
        exit(1)


logger = logging.getLogger(__name__)


class CloudStorageClient:
    """ Klient do obsługi MinIO (S3-compatible Object Storage). """
    def __init__(self):
        # Konfiguracja zczytywana z pliku .env
        self.endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "admin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "admin123")
        
        # Inicjalizacja klienta MinIO
        self.client = Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False  # False wymusza użycie HTTP zamiast domyślnego HTTPS
        )

    def upload_artifact(self, file_path: str, bucket_name: str, object_name: str = None):
        """ Uploaduje plik do wskazanego bucketu, tworząc go, jeśli nie istnieje. """
        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            if not self.client.bucket_exists(bucket_name):
                logger.info(f"Bucket '{bucket_name}' nie istnieje. Tworzenie.")
                self.client.make_bucket(bucket_name)

            # Właściwy upload pliku
            logger.info(f"Wysyłanie pliku {file_path} do MinIO (bucket: {bucket_name}).")
            self.client.fput_object(bucket_name, object_name, file_path)
            logger.info(f"Sukces: Plik {object_name} zabezpieczony w Object Storage.")
        except S3Error as e:
            logger.error(
                f"Nie udało się zabezpieczyć artefaktu w MinIO. "
                f"Plik: '{object_name}', Bucket: '{bucket_name}'. "
                f"Szczegóły S3 -> Kod: {e.code}, Komunikat: {e.message}"
            )
        except Exception as e:
            logger.error(
                f"Nieoczekiwany błąd sieciowy/systemowy podczas zapisu do MinIO. "
                f"Plik: '{file_path}'. Treść błędu: {e}"
            )


if __name__ == "__main__":
    # Uruchomienie procedury sprawdzenia / pobierania
    download_kaggle_patient_survival_data()

    # Sprawdzenie i utworzenie domyślnego pliku setup_db.sql
    logger.info("Generuję najnowszy skrypt migracji bazy setup_db.sql.")
    default_sql = """
    DROP TABLE IF EXISTS Clinical_Measurements_Temporal CASCADE;
    DROP TABLE IF EXISTS Visits CASCADE;
    DROP TABLE IF EXISTS Patients CASCADE;
    DROP TABLE IF EXISTS extracted_radiology_features CASCADE; 

    CREATE TABLE Patients (
        patient_id INT PRIMARY KEY,
        age INT,
        diabetes_mellitus INT,
        immunosuppression INT,
        cirrhosis INT,
        hepatic_failure INT,
        leukemia INT,
        lymphoma INT,
        solid_tumor_with_metastasis INT
    );

    CREATE TABLE Visits (
        hadm_id INT PRIMARY KEY,
        patient_id INT REFERENCES Patients(patient_id),
        hospital_outcome_death INT
    );

    CREATE TABLE Clinical_Measurements_Temporal (
        id SERIAL PRIMARY KEY,
        patient_id INT,
        hadm_id INT REFERENCES Visits(hadm_id),
        gcs_eyes INT,
        gcs_motor INT,
        gcs_verbal INT,
        sysbp_min FLOAT,
        heart_rate_max FLOAT,
        spo2_min FLOAT,
        glucose_max FLOAT,
        glucose_min FLOAT,
        d1_glucose_max FLOAT,
        d1_glucose_min FLOAT,
        d1_sysbp_max FLOAT,
        d1_heartrate_min FLOAT,
        d1_diasbp_min FLOAT,
        age INT,
        is_intubated INT,
        ventilated_apache INT,
        apache_3j_bodysystem VARCHAR(50),
        valid_to TIMESTAMP DEFAULT '2099-12-31 00:00:00'
    );

    -- Tabela cech radiologicznych i NLP z uwzględnieniem nerwu okoruchowego
    CREATE TABLE IF NOT EXISTS extracted_radiology_features (
        hadm_id INT PRIMARY KEY,
        vision_midline_shift_mm FLOAT,
        vision_basal_cisterns FLOAT,
        vision_corpus_callosum FLOAT,
        nlp_c_spine_fracture FLOAT,
        nlp_brain_aneurysm FLOAT,
        nlp_oculomotor_palsy FLOAT
    ); """
    with open(sql_path, "w", encoding="utf-8") as f:
        f.write(default_sql)

    logger.info("Wczytywanie train.csv oraz test.csv.")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Zestaw testów jakości danych
    ComprehensiveDataQualitySuite(train_df, dataset_name="train.csv").run_all_checks()
    ComprehensiveDataQualitySuite(test_df, dataset_name="test.csv").run_all_checks()

    # Zabezpieczenie przed brakiem kolumny docelowej w zbiorze testowym Kaggle
    if 'hospital_death' not in test_df.columns:
        test_df['hospital_death'] = -1

    logger.info(f"Wymiary train.csv: {train_df.shape}")
    logger.info(f"Wymiary test.csv: {test_df.shape}")

    # Pętla oczekiwania na fizyczną gotowość PostgreSQL
    logger.info("Oczekiwanie na gotowość bazy danych PostgreSQL.")
    for attempt in range(40):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Baza danych jest gotowa i akceptuje połączenia.")
            break
        except OperationalError:
            if attempt < 39:
                logger.info(f"Baza jeszcze się uruchamia. Czekam 5 sekund. (Próba {attempt+1}/40)")
                time.sleep(5)
            else:
                raise RuntimeError("Przekroczono limit czasu oczekiwania na bazę PostgreSQL.")
            
    logger.info("Zapisywanie surowych danych do tabel staging w PostgreSQL.")
    train_df.to_sql("staging_train", engine, if_exists="replace", index=True, index_label="index", chunksize=1000)
    test_df.to_sql("staging_test", engine, if_exists="replace", index=True, index_label="index", chunksize=1000)
    logger.info("Staging przygotowany pomyślnie.")

    logger.info("Tworzenie indeksów na tabelach staging.")
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_staging_train_patient ON staging_train(patient_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_staging_train_encounter ON staging_train(encounter_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_staging_test_patient ON staging_test(patient_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_staging_test_encounter ON staging_test(encounter_id);"))
    logger.info("Indeksy staging utworzone pomyślnie.")

    logger.info("SQL i migracja")
    with open(sql_path, "r", encoding="utf-8") as f:
        sql_script = f.read()

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cursor:
            cursor.execute(sql_script)
        raw_conn.commit()
    finally:
        raw_conn.close()

    logger.info("Mapowanie danych do tabel docelowych.")

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO Patients (
                    patient_id, age, diabetes_mellitus, immunosuppression, 
                    cirrhosis, hepatic_failure, leukemia, lymphoma, solid_tumor_with_metastasis
                )
                SELECT 
                    patient_id, MAX(age)::INT as age, 
                    MAX(diabetes_mellitus)::INT, MAX(immunosuppression)::INT, MAX(cirrhosis)::INT, 
                    MAX(hepatic_failure)::INT, MAX(leukemia)::INT, MAX(lymphoma)::INT, MAX(solid_tumor_with_metastasis)::INT
                FROM staging_train
                GROUP BY patient_id
                ON CONFLICT (patient_id) DO NOTHING;

                INSERT INTO Visits (hadm_id, patient_id, hospital_outcome_death)
                SELECT encounter_id as hadm_id, patient_id, MAX(hospital_death)::INT as hospital_outcome_death 
                FROM staging_train
                GROUP BY encounter_id, patient_id
                ON CONFLICT (hadm_id) DO NOTHING;

                INSERT INTO Clinical_Measurements_Temporal (
                    patient_id, hadm_id, gcs_eyes, gcs_motor, gcs_verbal, 
                    sysbp_min, heart_rate_max, spo2_min, glucose_max, glucose_min, 
                    d1_glucose_max, d1_glucose_min, d1_sysbp_max, d1_heartrate_min, d1_diasbp_min, 
                    age, is_intubated, ventilated_apache, apache_3j_bodysystem
                )
                SELECT 
                    patient_id, encounter_id as hadm_id, 
                    gcs_eyes_apache::INT as gcs_eyes, gcs_motor_apache::INT as gcs_motor, gcs_verbal_apache::INT as gcs_verbal, 
                    d1_sysbp_min as sysbp_min, d1_heartrate_max as heart_rate_max, d1_spo2_min as spo2_min, 
                    d1_glucose_max as glucose_max, d1_glucose_min as glucose_min, 
                    d1_glucose_max, d1_glucose_min, d1_sysbp_max, d1_heartrate_min, d1_diasbp_min, 
                    age::INT, intubated_apache::INT as is_intubated, ventilated_apache::INT, apache_3j_bodysystem
                FROM staging_train;
            """)
        raw_conn.commit()
    finally:
        raw_conn.close()

    logger.info("Struktura bazy i migracja danych zakończone pomyślnie.")

    logger.info("Weryfikacja bazy:")
    with engine.connect() as connection:
        result = connection.execute(text("""
            SELECT
                (SELECT count(*) FROM Patients) as pacjenci,
                (SELECT count(*) FROM Visits) as wizyty,
                (SELECT count(*) FROM Clinical_Measurements_Temporal) as pomiary;
        """))
        row = result.fetchone()
        logger.info(f" Liczba pacjentów: {row[0]}")
        logger.info(f" Liczba wizyt:   {row[1]}")
        logger.info(f" Liczba pomiarów: {row[2]}")

    logger.info("Weryfikuję strukturę kolumn w bazie danych.")
    required_columns = {
        "patients": ["patient_id", "age", "diabetes_mellitus"],
        "visits": ["hadm_id", "patient_id", "hospital_outcome_death"],
        "clinical_measurements_temporal": ["gcs_eyes", "sysbp_min", "is_intubated"]
    }

    with engine.connect() as conn:
        for table_name, cols in required_columns.items():
            # Sprawdzanie istnienia kolumn w bazie PostgreSQL
            query = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = :tname;
            """)
            existing_cols = {row[0] for row in conn.execute(query, {"tname": table_name}).fetchall()}
            
            missing = [c for c in cols if c not in existing_cols]
            if missing:
                raise RuntimeError(f"Tabela '{table_name}' nie posiada wymaganych kolumn: {missing}!")

    logger.info("Wszystkie wymagane tabele i kolumny są poprawnie zweryfikowane w bazie.")

    logger.info("Tworzenie i mapowanie danych testowych do tabel przeznaczonych do zewnętrznej walidacji.")

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cursor:
            cursor.execute("""
                DROP TABLE IF EXISTS Clinical_Measurements_Temporal_Test CASCADE;
                DROP TABLE IF EXISTS Visits_Test CASCADE;
                DROP TABLE IF EXISTS Patients_Test CASCADE;

                CREATE TABLE Patients_Test (
                    patient_id INT PRIMARY KEY,
                    age INT,
                    diabetes_mellitus INT,
                    immunosuppression INT,
                    cirrhosis INT,
                    hepatic_failure INT,
                    leukemia INT,
                    lymphoma INT,
                    solid_tumor_with_metastasis INT
                );

                CREATE TABLE Visits_Test (
                    hadm_id INT PRIMARY KEY,
                    patient_id INT REFERENCES Patients_Test(patient_id),
                    hospital_outcome_death INT
                );

                CREATE TABLE Clinical_Measurements_Temporal_Test (
                    id SERIAL PRIMARY KEY,
                    patient_id INT,
                    hadm_id INT REFERENCES Visits_Test(hadm_id),
                    gcs_eyes INT,
                    gcs_motor INT,
                    gcs_verbal INT,
                    sysbp_min FLOAT,
                    heart_rate_max FLOAT,
                    spo2_min FLOAT,
                    glucose_max FLOAT,
                    glucose_min FLOAT,
                    d1_glucose_max FLOAT,
                    d1_glucose_min FLOAT,
                    d1_sysbp_max FLOAT,
                    d1_heartrate_min FLOAT,
                    d1_diasbp_min FLOAT,
                    age INT,
                    is_intubated INT,
                    ventilated_apache INT,
                    apache_3j_bodysystem VARCHAR(50),
                    valid_to TIMESTAMP DEFAULT '2099-12-31 00:00:00'
                );

                INSERT INTO Patients_Test (
                    patient_id, age, diabetes_mellitus, immunosuppression, 
                    cirrhosis, hepatic_failure, leukemia, lymphoma, solid_tumor_with_metastasis
                )
                SELECT 
                    patient_id, MAX(age)::INT as age, 
                    MAX(diabetes_mellitus)::INT, MAX(immunosuppression)::INT, MAX(cirrhosis)::INT, 
                    MAX(hepatic_failure)::INT, MAX(leukemia)::INT, MAX(lymphoma)::INT, MAX(solid_tumor_with_metastasis)::INT
                FROM staging_test
                GROUP BY patient_id
                ON CONFLICT (patient_id) DO NOTHING;

                INSERT INTO Visits_Test (hadm_id, patient_id, hospital_outcome_death)
                SELECT encounter_id as hadm_id, patient_id, MAX(hospital_death)::INT as hospital_outcome_death 
                FROM staging_test
                GROUP BY encounter_id, patient_id
                ON CONFLICT (hadm_id) DO NOTHING;

                INSERT INTO Clinical_Measurements_Temporal_Test (
                    patient_id, hadm_id, gcs_eyes, gcs_motor, gcs_verbal, 
                    sysbp_min, heart_rate_max, spo2_min, glucose_max, glucose_min, 
                    d1_glucose_max, d1_glucose_min, d1_sysbp_max, d1_heartrate_min, d1_diasbp_min, 
                    age, is_intubated, ventilated_apache, apache_3j_bodysystem
                )
                SELECT 
                    patient_id, encounter_id as hadm_id, 
                    gcs_eyes_apache::INT as gcs_eyes, gcs_motor_apache::INT as gcs_motor, gcs_verbal_apache::INT as gcs_verbal, 
                    d1_sysbp_min as sysbp_min, d1_heartrate_max as heart_rate_max, d1_spo2_min as spo2_min, 
                    d1_glucose_max as glucose_max, d1_glucose_min as glucose_min, 
                    d1_glucose_max, d1_glucose_min, d1_sysbp_max, d1_heartrate_min, d1_diasbp_min, 
                    age::INT, intubated_apache::INT as is_intubated, ventilated_apache::INT, apache_3j_bodysystem
                FROM staging_test;
            """)
        raw_conn.commit()
    finally:
        raw_conn.close()

    logger.info("Zbiór testowy do zewnętrznej walidacji został pomyślnie zmapowany w bazie.")

    logger.info("Weryfikacja bazy testowej:")
    with engine.connect() as connection:
        result = connection.execute(text("""
            SELECT
                (SELECT count(*) FROM Patients_Test) as pacjenci_test,
                (SELECT count(*) FROM Visits_Test) as wizyty_test,
                (SELECT count(*) FROM Clinical_Measurements_Temporal_Test) as pomiary_test;
        """))
        row = result.fetchone()
        logger.info(f" Liczba pacjentów: {row[0]}")
        logger.info(f" Liczba wizyt:   {row[1]}")
        logger.info(f" Liczba pomiarów: {row[2]}")

    logger.info("Kompletny proces ETL dla treningu i walidacji zewnętrznej zakończony sukcesem.")

    #Zapis surowych danych do Data Lake
    try:
        storage = CloudStorageClient()
        
        # Wysyłka surowych danych do bucketu Data Lake
        if os.path.exists(train_path):
            storage.upload_artifact(train_path, bucket_name="tbi-raw-data")
        if os.path.exists(test_path):
            storage.upload_artifact(test_path, bucket_name="tbi-raw-data")
            
        logger.info("Surowe pliki danych zostały pomyślnie zarchiwizowane w MinIO (Data Lake).")
    except NameError as e:
        logger.error(f"Błąd zmiennych: {e}")
    except Exception as e:
        logger.warning(f"Zapis w chmurze (MinIO) nie powiódł się: {e}")
