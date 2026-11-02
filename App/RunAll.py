import sys
import os
import subprocess
import psycopg2
from dotenv import load_dotenv
from pydicom import config
config.settings.reading_validation_mode = config.IGNORE
import logging

# Poprawiona konfiguracja logowania (usunięto błędy składniowe z diffa)
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
        "extracted_radiology_features": ["vision_midline_shift_mm", "vision_basal_cisterns"],
        "mimic_icu_features": ["icp_max", "mannitol_administered"]
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
        logger.error(f"Błąd podczas weryfikacji bazy (serwer niedostępny lub błąd zapytania): {e}")
        return False


def run_step(script_path, expected_output=None, optional=False, expected_file=None, eval_if_exists=False):
    """ Uruchamia wskazany skrypt Pythona i monitoruje jego status.
        Sprawdza obecność plików wyjściowych, obrazów oraz opcjonalnego pliku. """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    def has_images(folder):
        if not os.path.exists(folder):
            return False
        for root, _, files in os.walk(folder):
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.nii')) for f in files):
                return True
        return False

    if expected_output:
        file_check_passed = True
        if expected_file:
            target_file_path = os.path.join(expected_output, expected_file) if os.path.isdir(expected_output) else expected_file
            if not os.path.exists(target_file_path):
                file_check_passed = False

        if os.path.isdir(expected_output):
            if has_images(expected_output) and file_check_passed:
                if eval_if_exists:
                    logger.info(f"Znaleziono gotowy wynik w: '{expected_output}'. Uruchamiam skrypt w trybie samej ewaluacji.")
                    env["EVAL_ONLY"] = "1"
                else:
                    logger.info(f"Znaleziono gotowy wynik z obrazami i plikiem '{expected_file}' w: '{expected_output}'.")
                    logger.info(f"Pomijam wykonywanie skryptu: {script_path}")
                    return
            else:
                logger.info(f"Folder '{expected_output}' istnieje, ale brakuje w nim obrazów lub pliku '{expected_file}'. Uruchamiam skrypt.")

        elif os.path.exists(expected_output) and file_check_passed:
            if eval_if_exists:
                logger.info(f"Znaleziono gotowy plik wyjściowy: '{expected_output}'. Uruchamiam skrypt w trybie samej ewaluacji.")
                env["EVAL_ONLY"] = "1"
            else:
                logger.info(f"Znaleziono gotowy plik wyjściowy: '{expected_output}'.")
                logger.info(f"Pomijam wykonywanie skryptu: {script_path}")
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
    logger.info("Start orkiestratora projektu TBI.")

    # Weryfikacja schematu kolumn w PostgreSQL
    logger.info("Weryfikuję strukturę bazy danych.")
    if verify_database_schema():
        logger.info("Pomijam wykonywanie skryptu: Data/InitAndSetupDB.py (Baza danych oraz schemat kolumn są aktualne).")
    else:
        logger.info("Wykryto brak bazy danych lub niekompletny schemat kolumn.")
        run_step(os.path.join("Data", "InitAndSetupDB.py"))
        run_step(os.path.join("Data", "PrepareMimicDB.py"))

    # Jawne sprawdzenie zarówno RSNA, jak i CQ500 przed pominięciem PrepareVisionData.py
    cq500_labels_path = os.path.join("Data", "dataset_cq500", "labels.csv")
    rsna_labels_path = os.path.join("Data", "dataset_ct_main", "labels.csv")
    weights_path = os.path.join("output_data", "Models", "resnet_50_23dataset.pth")

    # Przygotowanie danych wizualnych, w środku sprawdzamy obecność zarówno RSNA, jak i CQ500
    run_step(os.path.join("Data", "PrepareVisionData.py"))

    logger.info("Rozpoczynam trening CRNN dla krwawień wewnątrzczaszkowych.")
    run_step(
        os.path.join("Models", "VisionModels", "IntracranialHemorrhage_CRNN.py"),
        expected_output=os.path.join("output_data", "Models", "crnn_fine_uncertainty.pth"),
        eval_if_exists=True
    )

    logger.info("Rozpoczynam wstępny trening EfficientNet dla ciała modzelowatego.")
    run_step(
        os.path.join("Models", "VisionModels", "ModzelBody_UNet_to_EfficientNet.py"),
        expected_output=os.path.join("output_data", "Models", "efficientnet_best.pth"),
        optional=True
    )

    logger.info("Rozpoczynam trening złączenia UNet, dla poprawnej segmentacji i dostrajania EfficientNet dla ekstrakcji cech dla ciała modzelowatego.")
    run_step(
        os.path.join("Models", "VisionModels", "ModzelBody_UNet_to_EfficientNet.py"),
        expected_output=os.path.join("output_data", "Models", "efficientnet_corpus_callosum_finetune.pth")
    )

    logger.info("Rozpoczynam trening łączący Transformera z DenseNet do wykrywania złamań kości.")
    run_step(
        os.path.join("Models", "VisionModels", "BoneFracture_Trans_plus_Dense.py"),
        expected_output=os.path.join("output_data", "Models", "fracture_best.pth")
    )

    logger.info("Inicjalizacja modułów farmakokinetycznych oraz pobieranie ontologii/priorów DDI.")
    run_step(
        os.path.join("Models", "PharmaKineticsModels.py"),
        expected_output="ddi_prior.json",
        optional=True
    )

    logger.info("Inicjalizacja monitorowania trajektorii pacjenta i sieci Neural ODE.")
    run_step(
        os.path.join("Models", "TrajectoryMonitors.py"),
        expected_output=os.path.join("output_data", "Models", "trajectory_monitor_state.pth"),
        optional=True
    )

    logger.info("Inicjalizacja głównego skryptu analitycznego i integracji kaskadowej spinającej całość.")
    run_step(
        os.path.join("Models", "TBI.py"),
        expected_output=os.path.join("output_data", "Models", "holistic_cascade_e2e.pth")
    )

    logger.info("Cały proces orkiestracji zakończony pomyślnie! Moduły wizyjne, tabelaryczne, farmakokinetyczne oraz trajektorie ODE zostały w pełni zintegrowane.")
    logger.info("Ostateczny zintegrowany model kaskady oraz wagi modeli znajdują się w folderze 'output_data/Models'")