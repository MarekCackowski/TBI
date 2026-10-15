import sys
import os
import subprocess
import psycopg2
from dotenv import load_dotenv
from pydicom import config
config.settings.reading_validation_mode = config.IGNORE
import logging

logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)
load_dotenv()


def verify_database_schema():
    """ Fizycznie odpytuje bazę PostgreSQL, czy wymagane tabele i kolumny istnieją i są aktualne. """
    DB_HOST = os.getenv("DB_HOST", "tbi_postgres")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "tbi_database")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASS = os.getenv("DB_PASS", "postgres")
    
    # Kompleksowy słownik wymaganych tabel oraz kolumn
    required_columns = {
        "patients": ["patient_id", "age", "diabetes_mellitus"],
        "visits": ["hadm_id", "patient_id", "hospital_outcome_death"],
        "clinical_measurements_temporal": ["gcs_eyes", "gcs_motor", "gcs_verbal", "sysbp_min", "is_intubated"],
        "extracted_radiology_features": ["vision_midline_shift_mm", "nlp_brain_aneurysm"],
        "mimic_icu_features": ["icp_max", "rsbi", "mannitol_administered"]
    }

    try:
        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
        cur = conn.cursor()
        
        for table_name, cols in required_columns.items():
            # Sprawdzenie czy tabela istnieje
            cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s);", (table_name,))
            if not cur.fetchone()[0]:
                logger.error(f"Brak wymaganej tabeli: '{table_name}'.")
                cur.close()
                conn.close()
                return False
            
            # Sprawdzenie czy kolumny istnieją
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s;", (table_name,))
            existing_cols = {row[0] for row in cur.fetchall()}
            
            missing = [c for c in cols if c not in existing_cols]
            if missing:
                logger.error(f"W tabeli '{table_name}' brakuje kolumn: {missing}.")
                cur.close()
                conn.close()
                return False
                
        cur.close()
        conn.close()
        return True
    except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
        # Jeśli baza jeszcze wstaje lub wystąpił błąd połączenia/zapytania, uznajemy schemat za niegotowy
        logger.error(f"Błąd podczas weryfikacji bazy (serwer niedostępny lub błąd zapytania): {e}")
        return False


def run_step(script_path, expected_output=None, optional=False, expected_file=None):
    """ Uruchamia wskazany skrypt Pythona i monitoruje jego status. 
        Sprawdza obecność plików wyjściowych, obrazów oraz opcjonalnego pliku (np. labels.csv). """
    def has_images(folder):
        if not os.path.exists(folder): 
            return False
        for root, _, files in os.walk(folder):
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.nii')) for f in files): 
                return True
        return False

    if expected_output:
        # Sprawdzamy dodatkowo, czy wymagany plik (np. labels.csv) również istnieje
        file_check_passed = True
        if expected_file:
            target_file_path = os.path.join(expected_output, expected_file) if os.path.isdir(expected_output) else expected_file
            if not os.path.exists(target_file_path):
                file_check_passed = False

        if os.path.isdir(expected_output):
            if has_images(expected_output) and file_check_passed:
                logger.info(f"Znaleziono gotowy wynik z obrazami i plikiem '{expected_file}' w: '{expected_output}'.")
                logger.info(f"Pomijam wykonywanie skryptu: {script_path}\n")
                return
            else:
                logger.info(f"Folder '{expected_output}' istnieje, ale brakuje w nim obrazów lub pliku '{expected_file}'. Uruchamiam skrypt.")
        elif os.path.exists(expected_output) and file_check_passed:
            logger.info(f"Znaleziono gotowy plik wyjściowy: '{expected_output}'.")
            logger.info(f"Pomijam wykonywanie skryptu: {script_path}\n")
            return

    logger.info(f"Rozpoczynanie wykonywania skryptu: {script_path}")
    
    if not os.path.exists(script_path):
        if optional:
            logger.info(f"Plik '{script_path}' nie istnieje, ale jest opcjonalny. Pomijam krok.")
            return
        else:
            logger.error(f"Nie znaleziono pliku: {script_path}. Przerywam proces.")
            sys.exit(1)

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()

        result = subprocess.run([sys.executable, "-u", script_path], env=env)
        
        if result.returncode != 0:
            logger.error(f"Moduł '{script_path}' zakończył się błędem (kod: {result.returncode}). Przerywam cały proces.")
            sys.exit(result.returncode)
        else:
            logger.info(f"Zakończono sukcesem krok: {script_path}\n")
            
    except KeyboardInterrupt:
        logger.error(f"Przerwano ręcznie (Ctrl+C) podczas wykonywania {script_path}. Zatrzymuję orkiestrator.")
        sys.exit(130)
    except OSError as e:
        logger.error(f"Błąd systemu operacyjnego podczas próby uruchomienia {script_path}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    """ docker-compose up -d mongo && docker-compose run --rm tbi_app python run_pipeline.py """
    logger.info("Start orkiestratora projektu TBI.")

    # Weryfikacja schematu kolumn w PostgreSQL 
    logger.info("Weryfikuję strukturę bazy danych.")
    if verify_database_schema():
        logger.info("Pomijam wykonywanie skryptu: Data/InitAndSetupDB.py (Baza danych oraz schemat kolumn są aktualne).")
    else:
        logger.info("Wykryto brak bazy danych lub niekompletny schemat kolumn.")
        run_step(os.path.join("Data", "InitAndSetupDB.py"))
        run_step(os.path.join("Data", "PrepareMimicDB.py"))

    # Przygotowanie danych wizyjnych
    run_step(os.path.join("Data", "PrepareVisionData.py"), expected_output=os.path.join("Data", "dataset_ct_main"), expected_file="labels.csv")

    # Trening CRNN
    run_step(os.path.join("Models", "IntracranialHemorrhage_CRNN.py"), expected_output=os.path.join("output_data", "Models", "crnn_fine_uncertainty.pth"))

    # Trening UNet i EfficientNet
    run_step(os.path.join("Models", "ModzelBody_UNet_to_EfficientNet.py"), expected_output=os.path.join("output_data", "Models", "efficientnet_corpus_callosum_finetune.pth"))

    # Trening DenseNet+Transformer 3D
    run_step(os.path.join("Models", "BoneFracture_DenseNet_ViT.py"), expected_output=os.path.join("output_data", "Models", "fracture_best.pth"))

    # Główny skrypt analityczny i złączenie
    run_step(os.path.join("Models", "TBI.py"), expected_output=os.path.join("output_data", "Models", "holistic_cascade_e2e.pth"))

    logger.info("Cały proces orkiestracji zakończony pomyślnie! Wszystkie moduły wizyjne, językowe i tabelaryczne zostały zintegrowane.")
    logger.info("Ostateczny zintegrowany model kaskady oraz wagi modeli znajdują się w folderze 'output_data/Models'")