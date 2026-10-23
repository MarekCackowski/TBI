import os
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
import psycopg2
from tqdm import tqdm
from dotenv import load_dotenv
from psycopg2.extras import execute_values
import requests
import logging
logger = logging.getLogger(__name__)

load_dotenv()

# Dane logowania pobierane z pliku .env
DB_HOST = os.getenv("DB_HOST", "tbi_postgres")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "tbi_database")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")

# Dynamiczne ustalenie głównego katalogu (niezależnie skąd wywołano skrypt)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Kierujemy główny folder MIMIC do podkatalogu 'Data'
MIMIC_BASE_DIR = os.getenv("MIMIC_BASE_DIR", os.path.join(BASE_DIR, "Data"))

# Do filtrowania danych, żeby wyciągnąć tylko istotne cechy dla TBI i ICH
DIAGNOSES_PATH = os.path.join(MIMIC_BASE_DIR, "hosp", "diagnoses_icd.csv.gz")
LABEVENTS_PATH = os.path.join(MIMIC_BASE_DIR, "hosp", "labevents.csv.gz")
CHARTEVENTS_PATH = os.path.join(MIMIC_BASE_DIR, "icu", "chartevents.csv.gz")
CHUNK_SIZE = 250_000

# Zaktualizowane słowniki ItemID - dodano parametry Weaning i Farmakologię
LAB_ITEM_IDS = {
    50813: 'lactate',   
    50983: 'sodium',    
    51237: 'inr',       
    51265: 'platelets',
    
    # Gazometria krwi tętniczej (ABG)
    50820: 'ph',
    50821: 'pao2',
    50818: 'paco2'
}

CHART_ITEM_IDS = {
    220765: 'icp', 227989: 'icp',             
    223835: 'fio2',                           
    224409: 'pupil_right', 224410: 'pupil_left', 
    
    # Terapia osmotyczna (Mannitol + Hipertoniczny NaCl)
    221260: 'mannitol', 227533: 'mannitol',   
    227537: 'hypertonic_saline', 227538: 'hypertonic_saline',
    
    # Wazopresory (Noradrenalina, Dopamina, Fenylefryna)
    221906: 'noradrenaline', 221662: 'dopamine', 221749: 'phenylephrine',
    
    # Sedacja (Propofol, Midazolam, Dexmedetomidine)
    228096: 'rass',
    221653: 'propofol', 222168: 'propofol',
    221668: 'midazolam',
    221744: 'dexmedetomidine',
    227529: 'flumazenil', 221752: 'flumazenil',
    
    # Leki przeciwpadaczkowe (Keppra / Lewetyracetam)
    228340: 'levetiracetam',
    
    # Parametry respiratora i oddechu
    220210: 'respiratory_rate', 224690: 'respiratory_rate',
    224685: 'tidal_volume', 224684: 'tidal_volume', 224686: 'tidal_volume',
    220339: 'peep',
    224695: 'pip'
}

def download_file_if_missing(url, dest_path, desc):
    """ Sprawdza obecność pliku na dysku. Jeśli go nie ma, pobiera go automatycznie. """
    if os.path.exists(dest_path):
        logger.info(f"Plik {dest_path} jest obecny lokalnie.")
        return True
    
    dir_path = os.path.dirname(dest_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    logger.info(f"Automatyczne pobieranie pliku: {desc}.")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=desc, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = f.write(data)
                bar.update(size)
        logger.info(f"Pobieranie zakończone pomyślnie: {dest_path}")
        return True

    except requests.exceptions.Timeout as e:
        logger.error(f"Przekroczono limit czasu (timeout 60s) podczas pobierania {desc}: {e}")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "Nieznany"
        logger.error(f"Błąd HTTP (kod {status_code}) podczas pobierania {desc}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Błąd połączenia/sieciowy podczas pobierania {desc}: {e}")
    except (IOError, OSError) as e:
        logger.error(f"Błąd zapisu pliku na dysku ({dest_path}): {e}")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas pobierania {desc}: {e}")

    # Czyszczenie uszkodzonego lub niekompletnego pliku w przypadku wystąpienia błędu
    if os.path.exists(dest_path):
        try:
            os.remove(dest_path)
            logger.info(f"Usunięto niekompletny plik: {dest_path}")
        except OSError as cleanup_err:
            logger.warning(f"Nie udało się usunąć niekompletnego pliku {dest_path}: {cleanup_err}")

    return False


def extract_from_huge_csv(filepath, item_dict, desc):
    """ Czyta pliki CSV w paczkach, wyciąga surową historię czasową. Zoptymalizowane dla dużych plików MIMIC. """
    if not os.path.exists(filepath):
        logger.info(f"Brak pliku {filepath}. Upewnij się, że znajduje się w odpowiednim folderze.")
        return pd.DataFrame()

    logger.info(f"Rozpoczynam strumieniowe przetwarzanie pliku: {filepath}")
    filtered_chunks = []
    
    # Konwertujemy klucze na set do szybkiego wyszukiwania
    target_ids = set(item_dict.keys())
    cols_to_use = ['hadm_id', 'itemid', 'valuenum', 'charttime']
    
    # 250_000 żeby uniknąć przepełnienia pamięci RAM przy gzip
    chunk_iter = pd.read_csv(
        filepath, 
        chunksize=CHUNK_SIZE, 
        compression='gzip', 
        low_memory=False, 
        usecols=lambda c: c in cols_to_use,


        dtype={'hadm_id': 'Int64', 'itemid': 'Int32', 'valuenum': 'float32', 'charttime': 'str'}
    )
    
    for chunk in tqdm(chunk_iter, desc=desc):
        # Szybsze filtrowanie z użyciem zbioru
        mask = chunk['itemid'].isin(target_ids)
        if mask.any():
            useful_data = chunk[mask].copy()
            useful_data['feature_name'] = useful_data['itemid'].map(item_dict)
            filtered_chunks.append(useful_data)
            
    if not filtered_chunks:
        return pd.DataFrame()
        
    logger.info(f"[{desc}] Zakończono ekstrakcję. Łączenie wyfiltrowanych danych.")
    return pd.concat(filtered_chunks, ignore_index=True)


def filter_tbi_encounters(df_all):
    """ Filtruje surowe dane pozostawiając pełne spektrum pacjentów z urazami głowy i mózgu:
        - ICD-10: S06 (urazy wewnątrzczaszkowe), S02 (złamania czaszki)
        - ICD-9: 850-854 (urazy wewnątrzczaszkowe), 800-804 (złamania czaszki) """
    if not os.path.exists(DIAGNOSES_PATH):
        logger.info("Brak pliku diagnoses_icd.csv.gz! Nie można przefiltrować pacjentów pod kątem TBI.")
        return df_all
        
    logger.info("Filtrowanie pacjentów pod kątem pełnego spektrum TBI (kody ICD S06, S02, 850-854, 800-804).")
    diag_df = pd.read_csv(DIAGNOSES_PATH, compression='gzip', low_memory=False)
    diag_df['icd_code'] = diag_df['icd_code'].astype(str)
    
    # Maska dla urazów mózgu i złamań czaszki
    tbi_mask = (
        diag_df['icd_code'].str.startswith('S06', na=False) | 
        diag_df['icd_code'].str.startswith('S02', na=False) |
        diag_df['icd_code'].str.startswith(('850', '851', '852', '853', '854', '800', '801', '802', '803', '804'), na=False)
    )
    tbi_hadm_ids = diag_df.loc[tbi_mask, 'hadm_id'].unique()
    
    logger.info(f"Znaleziono {len(tbi_hadm_ids)} unikalnych hospitalizacji z pełnego spektrum TBI i urazów czaszki.")
    
    filtered_df = df_all[df_all['hadm_id'].isin(tbi_hadm_ids)].copy()
    logger.info(f"Liczba rekordów po rozszerzonej filtracji TBI: {len(filtered_df)} (z {len(df_all)})")
    return filtered_df


def targeted_rare_event_oversampling(df, rare_multiplier=20, general_multiplier=5):
    """ Adaptacyjne, wielopoziomowe nadpróbkowanie celowane odporne na brak specyficznych 
        markerów TBI w ogólnym zbiorze demonstracyjnym MIMIC-IV. """
    logger.info("Uruchamiam adaptacyjne, inteligentne nadpróbkowanie rzadkich stanów.")
    np.random.seed(42)
    
    continuous_cols = [
        'icp_max', 'icp_mean', 'icp_std', 'icp_slope', 
        'fio2_max', 'inr_max', 'platelets_min', 'noradrenaline_rate', 
        'lactate_max', 'sodium_mean', 'ph_min', 'pao2_min', 'paco2_max',
        'peep_max', 'pip_max', 'rass_mean', 'rsbi'
    ]
    
    # Ogólny jittering dla całej populacji
    general_dfs = [df]
    for i in range(1, general_multiplier):
        df_clone = df.copy()
        df_clone['hadm_id'] = df_clone['hadm_id'] + (i * 1000000)
        for col in continuous_cols:
            if col in df_clone.columns:
                std = df_clone[col].std()
                if pd.notnull(std) and std > 0:
                    df_clone[col] = df_clone[col] + np.random.normal(0, std * 0.03, size=len(df_clone))
        general_dfs.append(df_clone)
    base_augmented = pd.concat(general_dfs, ignore_index=True)
    
    # Adaptacyjne poszukiwanie pacjentów wysokiego ryzyka (Tiered approach)
    rare_seeds = pd.DataFrame()
    
    # Próba 1: Pacjenci z lekami sedacyjnymi lub wazopresorami i podwyższonym mleczanem / PEEP
    tier1_mask = (
        (df['sedatives_administered'].fillna(0) == 1) | 
        (df['vasopressor_load'].fillna(0) == 1)
    ) & (
        (df['lactate_max'].fillna(0) > 1.5) | 
        (df['peep_max'].fillna(0) > 5.0)
    )
    rare_seeds = df[tier1_mask].copy()
    
    # Próba 2: Jeśli w tier 1 nic nie ma, bierzemy kogokolwiek z jakiejkolwiek terapii intensywnej (sedacja lub wazopresory)
    if len(rare_seeds) < 2:
        tier2_mask = (df['sedatives_administered'].fillna(0) == 1) | (df['vasopressor_load'].fillna(0) == 1)
        rare_seeds = df[tier2_mask].copy()
        logger.info("Tier 1: Zbyt mało próbek. Rozszerzono kryteria do ogólnej terapii ICU.")
        
    # Próba 3: Ostateczny bezpiecznik – top pacjenci z najwyższym dostępnym mleczanem lub obciążeniem
    if len(rare_seeds) < 2:
        sort_col = 'lactate_max' if 'lactate_max' in df.columns and df['lactate_max'].notnull().sum() > 0 else df.columns[1]
        rare_seeds = df.nlargest(3, sort_col).copy()
        logger.info(f"Tier 2: Brak pełnych danych lekowych. Wybrano top 3 pacjentów na podstawie kryterium: {sort_col}.")
    else:
        logger.info(f"Sukces! Organicznie zidentyfikowano {len(rare_seeds)} pacjentów wysokiego ryzyka w zbiorze demo.")
        
    logger.info(f"Generuję celowane powielenie (x{rare_multiplier}) dla wybranych ziaren.")
    
    # Oversampling z szumem
    rare_augmented_list = []
    for j in range(rare_multiplier):
        seed_clone = rare_seeds.copy()
        seed_clone['hadm_id'] = seed_clone['hadm_id'] + 80000000 + (j * 100000)
        for col in continuous_cols:
            if col in seed_clone.columns:
                std = seed_clone[col].std()
                if pd.notnull(std) and std > 0:
                    seed_clone[col] = seed_clone[col] + np.random.normal(0, std * 0.02, size=len(seed_clone))
        rare_augmented_list.append(seed_clone)
        
    if rare_augmented_list:
        rare_df = pd.concat(rare_augmented_list, ignore_index=True)
        final_augmented_df = pd.concat([base_augmented, rare_df], ignore_index=True)
    else:
        final_augmented_df = base_augmented
        
    logger.info(f"Zakończono proces. Łączna liczba rekordów po inteligentnym nadpróbkowaniu: {len(final_augmented_df)}")
    return final_augmented_df


def backup_raw_mimic_to_minio(bucket_name="tbi-raw-data"):
    """ Archiwizuje surowe, skompresowane pliki MIMIC-IV w systemie MinIO. """

    # Upewniamy się, że MinIO jest dostępne i tworzymy bucket, jeśli nie istnieje
    raw_endpoint = os.getenv("MINIO_ENDPOINT", "tbi-minio:9000")
    if not raw_endpoint.startswith("http://") and not raw_endpoint.startswith("https://"):
        endpoint_url = f"http://{raw_endpoint}"
    else:
        endpoint_url = raw_endpoint

    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        region_name="us-east-1"
    )
    
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError:
        logger.info(f"Bucket '{bucket_name}' nie istnieje. Tworzenie.")
        try:
            s3_client.create_bucket(Bucket=bucket_name)
        except Exception as bucket_err:
            logger.error(f"Błąd tworzenia bucketu: {bucket_err}")
            return
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas sprawdzania/kreowania bucketu MinIO: {e}")
        return

    files_to_backup = [DIAGNOSES_PATH, LABEVENTS_PATH, CHARTEVENTS_PATH]
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            object_name = f"mimic_raw/{os.path.basename(file_path)}"
            logger.info(f"Wysyłanie {file_path} do MinIO jako {object_name}.")
            # Zabezpieczenie pojedynczego pliku w pętli
            try:
                s3_client.upload_file(file_path, bucket_name, object_name)
                logger.info(f"Sukces: {object_name} zarchiwizowany.")
            except ClientError as e:
                logger.error(f"Błąd MinIO/AWS przy wysyłaniu pliku {file_path}: {e}")
            except OSError as e:
                logger.error(f"Błąd dyskowy przy próbie odczytu pliku {file_path}: {e}")
            except Exception as e:
                logger.error(f"Nieoczekiwany błąd przy wysyłaniu pliku {file_path}: {e}")


def main():
    # Automatyczne pobranie brakujących danych demonstracyjnych MIMIC-IV
    logger.info("Rozpoczynam przetwarzanie lokalnych skompresowanych danych źródłowych.")
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS
        )
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS mimic_icu_features;")
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Stara tabela została usunięta (o ile istniała).")
    except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
        logger.info(f"Ostrzeżenie przy czyszczeniu tabeli w bazie danych: {e}")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd przy czyszczeniu tabeli w bazie danych: {e}")
        return

    df_lab = extract_from_huge_csv(LABEVENTS_PATH, LAB_ITEM_IDS, "Ekstrakcja Labevents")
    df_chart = extract_from_huge_csv(CHARTEVENTS_PATH, CHART_ITEM_IDS, "Ekstrakcja Chartevents")

    # Zabezpieczenie przed brakiem lub wielkością liter w kolumnach
    if not df_lab.empty:
        df_lab.columns = [str(c).lower() for c in df_lab.columns]
    if not df_chart.empty:
        df_chart.columns = [str(c).lower() for c in df_chart.columns]

    # Najpierw łączymy ramki, żeby df_all faktycznie istniał
    df_all = pd.concat([df_lab, df_chart], ignore_index=True)

    if df_all.empty:
        logger.info("Nie wyekstrahowano żadnych danych. Przerywam operację.")
        return

    if 'hadm_id' not in df_all.columns:
        raise KeyError("Brak wymaganej kolumny kluczowej 'hadm_id' w wyekstrahowanych danych.")

    # Dynamiczne ustalenie kolumn do oczyszczenia z NaN
    subset_cols = ['hadm_id']
    if 'valuenum' in df_all.columns:
        subset_cols.append('valuenum')
    elif 'value_num' in df_all.columns:
        subset_cols.append('value_num')

    df_all = df_all.dropna(subset=subset_cols)

    if df_all.empty:
        logger.info("Nie wyekstrahowano żadnych danych po oczyszczeniu. Przerywam operację.")
        return

    df_all = filter_tbi_encounters(df_all)
    if df_all.empty:
        logger.info("Brak pacjentów TBI po filtracji ICD! Sprawdź plik diagnoses_icd.csv.gz.")
        return

    logger.info("Agregacja statystyczna i dynamiczna (Wariancja, Mediana, Trendy czasowe).")
    
    if 'charttime' in df_all.columns:
        df_all['charttime'] = pd.to_datetime(df_all['charttime'], errors='coerce')
        
        val_check_col = 'valuenum' if 'valuenum' in df_all.columns else (
            'value_num' if 'value_num' in df_all.columns else subset_cols[0]
        )
        df_all = df_all.dropna(subset=[col for col in ['charttime', 'hadm_id', val_check_col] if col in df_all.columns])
        df_all = df_all.sort_values(['hadm_id', 'feature_name', 'charttime'])

        df_all['time_idx'] = df_all.groupby(['hadm_id', 'feature_name'])['charttime'].transform(
            lambda x: (x - x.min()).dt.total_seconds() / 3600.0
        )
    else:
        df_all['time_idx'] = 0.0

    def compute_advanced_stats(group):
        val_col = 'valuenum' if 'valuenum' in group.columns else (
            'value_num' if 'value_num' in group.columns else group.columns[0]
        )
        vals = group[val_col]
        res = {
            'max': vals.max(),
            'min': vals.min(),
            'mean': vals.mean(),
            'std': vals.std() if len(vals) > 1 else 0.0,
            'median': vals.median(),
            'iqr': vals.quantile(0.75) - vals.quantile(0.25)
        }
        
        if len(vals) > 2 and group['time_idx'].nunique() > 1:
            try:
                slope = np.polyfit(group['time_idx'], vals, 1)[0]
            except (np.linalg.LinAlgError, ValueError):
                slope = 0.0
        else:
            slope = 0.0
        res['slope'] = slope
        return pd.Series(res)

    grouped = df_all.groupby(['hadm_id', 'feature_name']).apply(compute_advanced_stats, include_groups=False).unstack(level='feature_name')
    grouped.columns = [f"{feat}_{stat}" for stat, feat in grouped.columns]
    grouped.reset_index(inplace=True)
    aggregated = grouped
    aggregated['hadm_id'] = aggregated['hadm_id'].astype(int)

    logger.info("Aplikowanie pogłębionych reguł patofizjologicznych TBI (czyste dane MIMIC).")
    
    final_df = pd.DataFrame()
    final_df['hadm_id'] = aggregated['hadm_id']
    
    # Neurologia i Hemodynamika
    final_df['icp_max'] = aggregated.get('icp_max', np.nan)
    final_df['icp_mean'] = aggregated.get('icp_mean', np.nan)
    final_df['icp_std'] = aggregated.get('icp_std', np.nan)
    final_df['icp_slope'] = aggregated.get('icp_slope', 0.0)
    
    final_df['fio2_max'] = aggregated.get('fio2_max', np.nan)
    final_df['inr_max'] = aggregated.get('inr_max', np.nan)
    final_df['platelets_min'] = aggregated.get('platelets_min', np.nan)
    final_df['noradrenaline_rate'] = aggregated.get('noradrenaline_max', pd.Series(0.0, index=aggregated.index)).fillna(0.0)
    final_df['lactate_max'] = aggregated.get('lactate_max', np.nan)
    final_df['sodium_mean'] = aggregated.get('sodium_mean', np.nan)
    
    # Mechanika Płuc i Gazometria
    final_df['ph_min'] = aggregated.get('ph_min', np.nan)
    final_df['pao2_min'] = aggregated.get('pao2_min', np.nan)
    final_df['paco2_max'] = aggregated.get('paco2_max', np.nan)
    final_df['peep_max'] = aggregated.get('peep_max', np.nan)
    final_df['pip_max'] = aggregated.get('pip_max', np.nan)
    final_df['rass_mean'] = aggregated.get('rass_mean', np.nan)

    rr_mean = aggregated.get('respiratory_rate_mean', np.nan)
    vt_mean_ml = aggregated.get('tidal_volume_mean', np.nan)
    final_df['rsbi'] = np.where((pd.notnull(rr_mean)) & (pd.notnull(vt_mean_ml)) & (vt_mean_ml > 0),
                                rr_mean / (vt_mean_ml / 1000.0), np.nan)
    
    mannitol = aggregated.get('mannitol_max', pd.Series(0, index=aggregated.index))
    h_saline = aggregated.get('hypertonic_saline_max', pd.Series(0, index=aggregated.index))
    final_df['mannitol_administered'] = np.where((mannitol > 0) | (h_saline > 0), 1, 0)
    
    norad = aggregated.get('noradrenaline_max', pd.Series(0, index=aggregated.index))
    dopa = aggregated.get('dopamine_max', pd.Series(0, index=aggregated.index))
    pheny = aggregated.get('phenylephrine_max', pd.Series(0, index=aggregated.index))
    final_df['vasopressor_load'] = np.where((norad > 0) | (dopa > 0) | (pheny > 0), 1, 0)
    
    prop = aggregated.get('propofol_max', pd.Series(0, index=aggregated.index))
    mida = aggregated.get('midazolam_max', pd.Series(0, index=aggregated.index))
    dex = aggregated.get('dexmedetomidine_max', pd.Series(0, index=aggregated.index))
    final_df['sedatives_administered'] = np.where((prop > 0) | (mida > 0) | (dex > 0), 1, 0)

    # Profilaktyka przeciwpadaczkowa
    levet = aggregated.get('levetiracetam_max', pd.Series(0, index=aggregated.index))
    final_df['anticonvulsant_administered'] = np.where(levet > 0, 1, 0)

    # Celowane nadpróbkowanie rzadkich stanów na bazie realnych trajektorii MIMIC
    final_df = targeted_rare_event_oversampling(final_df, rare_multiplier=20, general_multiplier=5)

    flum = aggregated.get('flumazenil_max', pd.Series(0, index=aggregated.index))
    final_df['flumazenil_administered'] = np.where(flum > 0, 1, 0)

    # Ocena reaktywności źrenic
    pupil_right = aggregated.get('pupil_right_max', pd.Series(0, index=aggregated.index))
    pupil_left = aggregated.get('pupil_left_max', pd.Series(0, index=aggregated.index))

    """ Anizokria: różnica średnicy > 1.0 mm między prawym a lewym okiem (przy dostępnych obu pomiarach), oznacza, że ucisk jest jednostronny,
        najczęściej w wyniku krwiaka podtwardówkowego/nadtwardówkowegp po jednej stronie i narastającego efektu masy. """
    pupil_diff = (pupil_right - pupil_left).abs()
    final_df['anizokria'] = np.where(
        (pd.notnull(pupil_right)) & (pd.notnull(pupil_left)) & (pupil_diff > 1.0), 1, 0
    )
    
    # Mydriaza: nadmierne rozszerzenie którejkolwiek źrenicy (> 5.0 mm), jest groźne, ponieważ oznacza głębokie niedokrwienie lub uszkodzenie pnia mózgu.
    final_df['mydriaza'] = np.where(
        (pupil_right > 5.0) | (pupil_left > 5.0), 1, 0
    )

    logger.info("Tworzenie tabeli w PostgreSQL i masowy zapis (zaugmentowane czyste dane).")
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS
        )
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS mimic_icu_features (
                hadm_id INT PRIMARY KEY,
                icp_max FLOAT,
                icp_mean FLOAT,
                icp_std FLOAT,
                icp_slope FLOAT,
                fio2_max FLOAT,
                inr_max FLOAT,
                platelets_min FLOAT,
                noradrenaline_rate FLOAT,
                lactate_max FLOAT,
                sodium_mean FLOAT,
                ph_min FLOAT,
                pao2_min FLOAT,
                paco2_max FLOAT,
                peep_max FLOAT,
                pip_max FLOAT,
                rass_mean FLOAT,
                rsbi FLOAT,
                mannitol_administered INT,
                sedatives_administered INT,
                flumazenil_administered INT,
                anizokria INT,
                mydriaza INT
            );
        """)
        
        insert_query = """
            INSERT INTO mimic_icu_features (
                hadm_id, icp_max, icp_mean, icp_std, icp_slope, 
                fio2_max, inr_max, platelets_min, noradrenaline_rate, 
                lactate_max, sodium_mean, ph_min, pao2_min, paco2_max,
                peep_max, pip_max, rass_mean, rsbi,
                mannitol_administered, sedatives_administered, flumazenil_administered, anizokria, mydriaza
            ) VALUES %s
            ON CONFLICT (hadm_id) DO UPDATE SET
                icp_max = EXCLUDED.icp_max,
                icp_mean = EXCLUDED.icp_mean,
                icp_std = EXCLUDED.icp_std,
                icp_slope = EXCLUDED.icp_slope,
                fio2_max = EXCLUDED.fio2_max,
                inr_max = EXCLUDED.inr_max,
                platelets_min = EXCLUDED.platelets_min,
                noradrenaline_rate = EXCLUDED.noradrenaline_rate,
                lactate_max = EXCLUDED.lactate_max,
                sodium_mean = EXCLUDED.sodium_mean,
                ph_min = EXCLUDED.ph_min,
                pao2_min = EXCLUDED.pao2_min,
                paco2_max = EXCLUDED.paco2_max,
                peep_max = EXCLUDED.peep_max,
                pip_max = EXCLUDED.pip_max,
                rass_mean = EXCLUDED.rass_mean,
                rsbi = EXCLUDED.rsbi,
                mannitol_administered = EXCLUDED.mannitol_administered,
                sedatives_administered = EXCLUDED.sedatives_administered,
                flumazenil_administered = EXCLUDED.flumazenil_administered,
                anizokria = EXCLUDED.anizokria,
                mydriaza = EXCLUDED.mydriaza;
        """
        
        values_list = []
        for _, row in final_df.iterrows():
            values_list.append((
                int(row['hadm_id']),
                float(row['icp_max']) if pd.notnull(row['icp_max']) else None,
                float(row['icp_mean']) if pd.notnull(row['icp_mean']) else None,
                float(row['icp_std']) if pd.notnull(row['icp_std']) else None,
                float(row['icp_slope']) if pd.notnull(row['icp_slope']) else 0.0,
                float(row['fio2_max']) if pd.notnull(row['fio2_max']) else None,
                float(row['inr_max']) if pd.notnull(row['inr_max']) else None,
                float(row['platelets_min']) if pd.notnull(row['platelets_min']) else None,
                float(row['noradrenaline_rate']) if pd.notnull(row['noradrenaline_rate']) else 0.0,
                float(row['lactate_max']) if pd.notnull(row['lactate_max']) else None,
                float(row['sodium_mean']) if pd.notnull(row['sodium_mean']) else None,
                float(row['ph_min']) if pd.notnull(row['ph_min']) else None,
                float(row['pao2_min']) if pd.notnull(row['pao2_min']) else None,
                float(row['paco2_max']) if pd.notnull(row['paco2_max']) else None,
                float(row['peep_max']) if pd.notnull(row['peep_max']) else None,
                float(row['pip_max']) if pd.notnull(row['pip_max']) else None,
                float(row['rass_mean']) if pd.notnull(row['rass_mean']) else None,
                float(row['rsbi']) if pd.notnull(row['rsbi']) else None,
                int(row.get('mannitol_administered', 0)) if 'mannitol_administered' in row and pd.notna(row.get('mannitol_administered')) else 0,
                int(row.get('sedatives_administered', 0)) if 'sedatives_administered' in row and pd.notna(row.get('sedatives_administered')) else 0,
                int(row.get('flumazenil_administered', 0)) if 'flumazenil_administered' in row and pd.notna(row.get('flumazenil_administered')) else 0,
                int(row.get('anizokria', 0)) if 'anizokria' in row and pd.notna(row.get('anizokria')) else 0,
                int(row.get('mydriaza', 0)) if 'mydriaza' in row and pd.notna(row.get('mydriaza')) else 0
            ))
            
        execute_values(cur, insert_query, values_list, page_size=5000)
            
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Pomyślnie załadowano {len(final_df)} czystych rekordów do Postgresa.")

        logger.info("Rozpoczynam synchronizację surowych danych szpitalnych z MinIO Data Lake.")
        backup_raw_mimic_to_minio()
        
    except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
        logger.info(f"BŁĄD SQL podczas zapisu cech MIMIC: {e}")

    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas przetwarzania danych MIMIC: {e}")

       
if __name__ == "__main__":
    main()