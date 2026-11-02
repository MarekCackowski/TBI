from collections import defaultdict
from dataclasses import field
import subprocess
import sys
import os
import shutil
from typing import List, Optional, Tuple
import zipfile

from pydantic.dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
import requests
import cv2 as cv
import pydicom
from dotenv import load_dotenv
from tqdm import tqdm
import warnings
from pydicom import config
config.settings.reading_validation_mode = config.IGNORE
import logging
logger = logging.getLogger(__name__)

# Wyłączenie ostrzeżeń o niepoprawnych metadanych w plikach DICOM (RSNA i CQ500)
warnings.filterwarnings('ignore', category=UserWarning, module='pydicom')

# Wymuszamy zaczytanie zmiennych przed importem Kaggle API
load_dotenv()
os.environ['KAGGLE_KEY'] = os.getenv("KAGGLE_API_TOKEN", "")
os.environ['KAGGLE_USERNAME'] = os.getenv("KAGGLE_USERNAME", "api")

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    import nibabel as nib
except (OSError, ImportError, ModuleNotFoundError) as e:
    logger.error(f"Błąd importu! Upewnij się, że pakiety są zainstalowane, a w pliku .env jest KAGGLE_USERNAME i KAGGLE_API_TOKEN. Szczegóły: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Nieoczekiwany błąd podczas importu pakietów: {e}")
    sys.exit(1)


DATA_DIR = os.path.expanduser("~/TBI/Data")


def download_kaggle_item(item_name, download_path, is_competition=False):
    """ Uniwersalna funkcja pobierająca zbiór danych lub konkurs z Kaggle przez aria2c. """
    os.makedirs(download_path, exist_ok=True)
    
    try:
        username = os.getenv("KAGGLE_USERNAME", "api")
        key = os.getenv("KAGGLE_API_TOKEN", "")
        
        if is_competition:
            zip_path = os.path.join(download_path, f"{item_name}.zip")
            if os.path.exists(zip_path):
                logger.info(f"Plik archiwum '{zip_path}' już istnieje na dysku. Pomijam pobieranie.")
                return True
            
            # Pobranie bezpośredniego adresu URL przekierowania dla konkursu przez Kaggle API
            r = requests.get(f"https://www.kaggle.com/api/v1/competitions/data/download-all/{item_name}", auth=(username, key), allow_redirects=False)
            download_url = r.headers.get("Location")
            
            logger.info(f"Pobieranie konkursu {item_name} przez wielowątkowe aria2c.")
            subprocess.run(["aria2c", "-x", "16", "-s", "16", "-d", download_path, "-o", f"{item_name}.zip", download_url], check=True)
        else:
            if len(os.listdir(download_path)) > 0:
                logger.info(f"Folder datasetu '{download_path}' nie jest pusty. Pomijam pobieranie.")
                return True
                
            r = requests.get(f"https://www.kaggle.com/api/v1/datasets/download/{item_name}", auth=(username, key), allow_redirects=False)
            download_url = r.headers.get("Location")
            
            logger.info(f"Pobieranie zbioru danych {item_name} z Kaggle przez aria2c.")
            subprocess.run(["aria2c", "-x", "16", "-s", "16", "-d", download_path, "-o", "dataset.zip", download_url], check=True)
            
            # Rozpakowanie datasetu po pobraniu
            zip_file_path = os.path.join(download_path, "dataset.zip")
            if os.path.exists(zip_file_path):
                # Pasek postępu dla rozpakowywania pliku ZIP
                with zipfile.ZipFile(zip_file_path, 'r') as zf:
                    for member in tqdm(zf.infolist(), desc='Rozpakowywanie zbioru Kaggle'):
                        zf.extract(member, download_path)

                # Po rozpakowaniu usuwamy plik ZIP, aby zaoszczędzić miejsce
                os.remove(zip_file_path)
                
    except subprocess.CalledProcessError as cpe:
        logger.error(f"Błąd wykonania aria2c dla {item_name}: {cpe}")
        return False
    except Exception as oe:
        logger.error(f"Nieoczekiwany błąd podczas pobierania {item_name}: {oe}")
        return False
        
    return True


def apply_edge_boost(img_windowed: np.ndarray) -> np.ndarray:
    """ Podbija sygnał brzegowy za pomocą filtru Sobela, by uwydatnić krwiaki podtwardówkowe. """
    sobelx = cv.Sobel(img_windowed, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(img_windowed, cv.CV_64F, 0, 1, ksize=3)
    edges = cv.magnitude(sobelx, sobely)
    edges = cv.normalize(edges, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return cv.addWeighted(img_windowed, 0.7, edges, 0.3, 0)


def morphological_skull_strip(image_hu):
    """ Izoluje tkankę miękką mózgu, odcinając kości czaszki za pomocą operacji morfologicznych. """
    brain_mask = np.logical_and(image_hu > -50, image_hu < 250).astype(np.uint8) * 255

    kernel = np.ones((5,5), np.uint8)
    brain_mask = cv.erode(brain_mask, kernel, iterations=1)
    brain_mask = cv.dilate(brain_mask, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(brain_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        final_mask = (labels == largest_label).astype(np.uint8)
        final_mask = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, kernel, iterations=2)
    else:
        final_mask = np.ones_like(image_hu, dtype=np.uint8)

    stripped_image = np.where(final_mask, image_hu, -1000)
    return stripped_image, final_mask


def apply_ct_window(img_array, dicom_data, window_center, window_width, apply_skull_strip=False):
    """ Konwertuje surowe piksele na jednostki Hounsfielda z opcjonalnym skull strippingiem oraz obsługą MONOCHROME1. """
    if img_array is None or img_array.size == 0 or img_array.ndim != 2:
        return None
        
    # Obsługa inwersji fotometrycznej (MONOCHROME1 vs MONOCHROME2), dla zgodności, ponieważ niektóre pliki DICOM mogą mieć odwróconą skalę jasności, zależnie od aparatu CT.
    photometric = getattr(dicom_data, 'PhotometricInterpretation', 'MONOCHROME2')
    if photometric == 'MONOCHROME1':
        img_array = np.max(img_array) - img_array

    # Pobranie parametrów Rescale
    try:
        intercept = float(getattr(dicom_data, 'RescaleIntercept', 0.0))
        slope = float(getattr(dicom_data, 'RescaleSlope', 1.0))
    except (ValueError, TypeError):
        intercept = 0.0
        slope = 1.0
    
    img_hu = img_array * slope + intercept
    
    # Selektywne usunięcie czaszki przed nałożeniem okna
    if apply_skull_strip:
        img_hu, _ = morphological_skull_strip(img_hu)
        
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    
    img_windowed = np.clip(img_hu, img_min, img_max)
    return ((img_windowed - img_min) / window_width * 255).astype(np.uint8)


def window_image(img_hu, window_center=40, window_width=80, apply_skull_strip=False):
    """ Filtruje Hounsfield Units dla skanów NIfTI (CQ500) z opcjonalnym skull strippingiem. """
    if apply_skull_strip:
        img_hu, _ = morphological_skull_strip(img_hu)
        
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img_clipped = np.clip(img_hu, img_min, img_max)
    if img_max > img_min:
        img_clipped = (img_clipped - img_min) / (img_max - img_min) * 255.0
    return img_clipped.astype(np.uint8)



@dataclass
class MedicalImagingConfig:
    window_brain: Tuple[float, float] = (40.0, 80.0)
    window_blood: Tuple[float, float] = (80.0, 200.0)
    window_bone: Tuple[float, float] = (600.0, 2800.0)
    num_slices: int = 30
    img_size: int = 224
    target_classes: List[str] = field(
        default_factory=lambda: [
            'epidural',
            'subdural',
            'subarachnoid',
            'intraparenchymal',
            'intraventricular'
        ]
    )


def extract_nifti_to_pngs(nii_path: str, output_dir: str, imaging_config: Optional[MedicalImagingConfig] = None):
    """ Ekstrahuje plik NIfTI do obrazów PNG, nakładając potrójne okno radiologiczne CT (Podtwardówkowe z Sobelem, Krew, Mózg) 
        z zabezpieczeniem przed brakiem atrybutów w konfiguracji. """
    if imaging_config is None:
        imaging_config = MedicalImagingConfig()
        
    os.makedirs(output_dir, exist_ok=True)
    try:
        nii = nib.load(nii_path)
        img_data = nii.get_fdata() 
        
        # W plikach NIfTI (CQ500) oś Z jest na 3 wymiarze
        num_slices = img_data.shape[2]
        
        def apply_ct_window_nifti(img_hu, window_center, window_width):
            img_min = window_center - window_width // 2
            img_max = window_center + window_width // 2
            img_windowed = np.clip(img_hu, img_min, img_max)
            return ((img_windowed - img_min) / window_width * 255).astype(np.uint8)

        # Bezpieczne pobranie parametrów okien z MedicalImagingConfig (z domyślnymi bezpiecznymi fallbackami, nie używamy kości)
        wb = getattr(imaging_config, 'window_brain', (40, 80))
        wbl = getattr(imaging_config, 'window_blood', (80, 200))
        wsd = getattr(imaging_config, 'window_subdural', (70, 150))

        # Pasek postępu dla ekstrakcji warstw NIfTI na PNG
        for z in tqdm(range(num_slices), desc=f'Ekstrakcja warstw z pliku {os.path.basename(nii_path)}', leave=False):
            img_slice = img_data[:, :, z]
            
            # Obrót o 90 stopni (standardowo NIfTI różni się orientacją od DICOM)
            img_slice = np.rot90(img_slice)
            
            # NIfTI zawiera woksele już przeliczone na HU, nakładamy okna radiologiczne
            img_brain = apply_ct_window_nifti(img_slice, wb[0], wb[1])
            img_blood = apply_ct_window_nifti(img_slice, wbl[0], wbl[1])
            
            # Wyizolowanie okna podtwardówkowego z krawędziami (Sobel boost)
            img_subdural = apply_ct_window_nifti(img_slice, wsd[0], wsd[1])
            img_subdural = apply_edge_boost(img_subdural)
            
            # Format BGR zgodny z modelem CRNN
            img_bgr = np.stack([img_subdural, img_blood, img_brain], axis=-1)
            
            out_path = os.path.join(output_dir, f"slice_{z:04d}.png")
            cv.imwrite(out_path, img_bgr)
            
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
        logger.error(f"Błąd we/wy lub uszkodzone dane podczas przetwarzania NIfTI {nii_path}: {e}")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas przetwarzania NIfTI {nii_path}: {e}")


def prepare_rsna_training_data(zip_path, csv_path):
    """ Przetwarza zbiór RSNA jako bazę treningową z uwzględnieniem grubości slajdów (Slice Thickness / Delta Z). """    
    if not os.path.exists(zip_path) or not os.path.exists(csv_path):
        logger.error(f"Brak pliku RSNA zip ({zip_path}) lub csv ({csv_path}).")
        return False

    labels_target_path = os.path.join(DATA_DIR, 'dataset_ct_main', 'labels.csv')
    if os.path.exists(labels_target_path):
        logger.info("Zbiór treningowy RSNA jest już przetworzony i gotowy. Pomijam ekstrakcję z pliku ZIP.")
        return True

    TARGET_CLASSES = ['epidural', 'subdural', 'subarachnoid', 'intraparenchymal', 'intraventricular']
    
    crnn_main_img = os.path.join('Data', 'dataset_ct_main', 'images')
    crnn_ft_img = os.path.join('Data', 'dataset_ct_fine', 'images')
    caps_main_norm = os.path.join('Data', 'dataset_capsnet', '0_normal')
    caps_main_dai = os.path.join('Data', 'dataset_capsnet', '1_corpus_callosum_dai')
    caps_ft_norm = os.path.join('Data', 'dataset_capsnet_finetune', '0_normal')
    caps_ft_dai = os.path.join('Data', 'dataset_capsnet_finetune', '1_corpus_callosum_dai')

    for d in [crnn_main_img, crnn_ft_img, caps_main_norm, caps_main_dai, caps_ft_norm, caps_ft_dai]:
        os.makedirs(d, exist_ok=True)

    logger.info("Ładowanie i transformacja metadanych RSNA ze słownika.")
    df = pd.read_csv(csv_path)
    
    df['ImageID'] = df['ID'].apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else x)
    df['Subtype'] = df['ID'].apply(lambda x: x.split('_')[-1] if len(x.split('_')) > 1 else 'any')
    
    df_pathology = df[df['Subtype'].isin(TARGET_CLASSES)]
    pivot_df = df_pathology.pivot_table(index='ImageID', columns='Subtype', values='Label', aggfunc='max').fillna(0)
    labels_dict = pivot_df.to_dict('index')

    # Tymczasowe struktury do grupowania według pacjenta przed sortowaniem po osi Z
    patients_raw_slices = defaultdict(list)
    patients_labels = defaultdict(list)
    current_patients = set()

    logger.info("Rozpoczynam wolumetryczną ekstrakcję DICOM z ZIP (z 3 oknami HU).")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            dcm_files = [f for f in z.namelist() if f.endswith('.dcm')]
            
            for i, dcm_file in enumerate(tqdm(dcm_files, desc="Przetwarzanie plików DICOM", leave=False, file=sys.stdout, mininterval=2.0)):
                try:
                    filename = os.path.basename(dcm_file)
                    image_id = filename.replace('ID_', '').replace('.dcm', '')
                    
                    if image_id not in labels_dict:
                        continue
                        
                    with z.open(dcm_file) as f:
                        dicom_data = pydicom.dcmread(f, stop_before_pixels=False)
                        patient_id = str(getattr(dicom_data, 'StudyInstanceUID', getattr(dicom_data, 'PatientID', 'Unknown')))
                        
                        if patient_id not in current_patients:
                            current_patients.add(patient_id)
                        
                        z_loc = float(getattr(dicom_data, 'SliceLocation', getattr(dicom_data, 'InstanceNumber', 0)))
                        img_array = dicom_data.pixel_array
                        if img_array.size == 0: 
                            continue
                        
                        img_brain = apply_ct_window(img_array, dicom_data, window_center=40, window_width=80, apply_skull_strip=True)
                        img_blood = apply_ct_window(img_array, dicom_data, window_center=80, window_width=200, apply_skull_strip=True)
                        img_bone = apply_ct_window(img_array, dicom_data, window_center=600, window_width=2000, apply_skull_strip=False)
                        
                        img_bgr = np.stack([img_bone, img_blood, img_brain], axis=-1)
                        
                        slice_labels = labels_dict[image_id]
                        vec = [
                            slice_labels.get('epidural', 0),
                            slice_labels.get('subdural', 0),
                            slice_labels.get('subarachnoid', 0),
                            slice_labels.get('intraparenchymal', 0),
                            slice_labels.get('intraventricular', 0)
                        ]
                        
                        # Zapisujemy do bufora pacjenta, dla łatwiejszego odczytu
                        patients_raw_slices[patient_id].append({
                            'z_loc': z_loc,
                            'image_id': image_id,
                            'img_bgr': img_bgr,
                            'vec': vec
                        })
                        
                except Exception as dicom_err:
                    logger.error(f"Pomijam uszkodzony plik {dcm_file}: {dicom_err}")
                    continue
                    
    except zipfile.BadZipFile as bz:
        logger.error(f"Błąd: Plik ZIP archiwum RSNA jest uszkodzony: {bz}")
        return False
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas przetwarzania RSNA: {e}")
        return False

    logger.info("Sortowanie wolumenu, obliczanie grubości slajdów (Delta Z) i zapis na dysk.")
    crnn_labels_main = []
    crnn_labels_ft = []
    
    patient_id_list = list(current_patients)
    
    for idx, pid in enumerate(tqdm(patient_id_list, desc="Dystrybucja CapsNet/Fine-Tune i Slice Thickness")):
        slices = patients_raw_slices[pid]
        if not slices:
            continue
            
        # Sortowanie slajdów rosnąco po pozycji Z (dla spójności sekwencji, ważna dla analizy 3D)
        slices.sort(key=lambda x: x['z_loc'])
        
        # Obliczenie odległości pierwszego rzędu (Delta Z / Slice Thickness)
        z_coords = [s['z_loc'] for s in slices]
        thicknesses = [0.0] + [abs(z_coords[i] - z_coords[i-1]) for i in range(1, len(z_coords))]
        
        # Normalizacja grubości do zakresu [0, 1] (zabezpieczenie przed dzieleniem przez zero)
        max_t = max(thicknesses) if thicknesses and max(thicknesses) > 0 else 1.0
        norm_thicknesses = [t / max_t for t in thicknesses]

        patient_dir_crnn = os.path.join(crnn_main_img, pid)
        os.makedirs(patient_dir_crnn, exist_ok=True)
        
        patient_vecs = []
        patient_thickness_meta = {}

        for s_idx, s in enumerate(slices):
            image_id = s['image_id']
            z_loc = s['z_loc']
            thick = norm_thicknesses[s_idx]
            
            safe_z_loc = z_loc + 5000.0 
            out_filename = f"{safe_z_loc:08.2f}_{image_id}.png"
            out_path = os.path.join(patient_dir_crnn, out_filename)
            
            cv.imwrite(out_path, s['img_bgr'])
            patient_vecs.append(s['vec'])
            
            # Mapowanie nazwy pliku PNG na jego wyliczoną znormalizowaną grubość
            patient_thickness_meta[out_filename] = thick

        pat_vec = np.max(patient_vecs, axis=0)
        thickness_str = ",".join([f"{t:.4f}" for t in norm_thicknesses]) # Formatujemy znormalizowane grubości slajdów do postaci zapisu w jednej kolumnie CSV, to ważne dla późniejszej analizy w modelu CRNN
        label_dict = {
            'PatientID': pid, 
            'epidural': pat_vec[0], 
            'subdural': pat_vec[1],
            'subarachnoid': pat_vec[2], 
            'intraparenchymal': pat_vec[3], 
            'intraventricular': pat_vec[4],
            'slice_thicknesses': thickness_str
        }
        crnn_labels_main.append(label_dict)
        
        has_pathology = any(pat_vec)
        target_caps_main = caps_main_dai if has_pathology else caps_main_norm
        
        src_dir = os.path.join(crnn_main_img, pid)
        if os.path.exists(src_dir):
            shutil.copytree(src_dir, os.path.join(target_caps_main, pid), dirs_exist_ok=True)
            
            if idx < len(patient_id_list) // 4:
                crnn_labels_ft.append(label_dict)
                shutil.copytree(src_dir, os.path.join(crnn_ft_img, pid), dirs_exist_ok=True)
                target_caps_ft = caps_ft_dai if has_pathology else caps_ft_norm
                shutil.copytree(src_dir, os.path.join(target_caps_ft, pid), dirs_exist_ok=True)

    pd.DataFrame(crnn_labels_main).to_csv(os.path.join('Data', 'dataset_ct_main', 'labels.csv'), index=False)
    if crnn_labels_ft:
        pd.DataFrame(crnn_labels_ft).to_csv(os.path.join('Data', 'dataset_ct_fine', 'labels.csv'), index=False)

    logger.info("Pomyślnie wygenerowano uporządkowane wolumetrycznie dane z uwzględnieniem grubości slajdów.")
    return True


def prepare_cq500_evaluation_data(info_csv, source_root):
    """ Przetwarza zbiór CQ500, używany w celu walidacji modeli. """
    if not os.path.exists(info_csv) or not os.path.exists(source_root):
        logger.error(f"Brak pliku info.csv ({info_csv}) lub folderu źródłowego CQ500 ({source_root}).")
        return False

    cq500_crnn_dir = os.path.join(DATA_DIR, 'dataset_cq500')
    cq500_caps_dir = os.path.join(DATA_DIR, 'dataset_cq500_capsnet')
    labels_target_path = os.path.join(cq500_crnn_dir, 'labels.csv')
    
    # Zabezpieczenie przed ponownym przetworzeniem
    if os.path.exists(labels_target_path):
        logger.info("Zbiór walidacyjny CQ500 jest już przetworzony i gotowy. Pomijam ekstrakcję.")
        return True

    df = pd.read_csv(info_csv)
    
    cq500_crnn_img = os.path.join(cq500_crnn_dir, 'images')
    
    os.makedirs(cq500_crnn_img, exist_ok=True)
    os.makedirs(os.path.join(cq500_caps_dir, '0_normal'), exist_ok=True)
    os.makedirs(os.path.join(cq500_caps_dir, '1_corpus_callosum_dai'), exist_ok=True)

    logger.info("Przetwarzanie zbioru CQ500 na zewnętrzny zbiór testowy.")
    
    crnn_labels = []
    count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Przetwarzanie CQ500", file=sys.stdout, mininterval=2.0):
        patient_folder = str(row['name'])
        nii_full_path = None
        
        # Dynamiczne wyszukiwanie pliku NIfTI dla danego pacjenta
        for root, _, files in os.walk(source_root):
            if patient_folder in root:
                for f in files:
                    if f.endswith(('.nii', '.nii.gz')):
                        nii_full_path = os.path.join(root, f)
                        break
            if nii_full_path:
                break
                
        if not nii_full_path or not os.path.exists(nii_full_path):
            continue

        crnn_labels.append({
            'PatientID': patient_folder,
            'epidural': float(row.get('EDH', 0)),
            'subdural': float(row.get('SDH', 0)),
            'subarachnoid': float(row.get('SAH', 0)),
            'intraparenchymal': float(row.get('IPH', 0)),
            'intraventricular': float(row.get('IVH', 0))
        })
        
        crnn_target = os.path.join(cq500_crnn_img, patient_folder)
        extract_nifti_to_pngs(nii_full_path, crnn_target)

        iph_score = float(row.get('IPH', 0))
        target_caps_sub = '1_corpus_callosum_dai' if iph_score >= 0.5 else '0_normal'
        caps_target = os.path.join(cq500_caps_dir, target_caps_sub, patient_folder)
        
        if not os.path.exists(caps_target):
            shutil.copytree(crnn_target, caps_target, dirs_exist_ok=True)
            
        count += 1

    pd.DataFrame(crnn_labels).to_csv(os.path.join(cq500_crnn_dir, 'labels.csv'), index=False)
    logger.info(f"Przygotowano {count} pacjentów CQ500 do niezależnych testów.")
    return True


def prepare_physionet_from_kaggle(target_dir=os.path.join('Data', 'dataset_physionet')):
    """ Pobiera zbiór PhysioNet CT-ICH z Kaggle i konwertuje pliki NIfTI na PNG. """
    png_dir = os.path.join(target_dir, 'images')
    
    if os.path.exists(png_dir) and len(os.listdir(png_dir)) > 0:
        logger.info("Zbiór PhysioNet z Kaggle już istnieje w katalogu Data/.")
        return True

    logger.info("Pobieranie zbioru PhysioNet CT-ICH z Kaggle.")
    if download_kaggle_item("snikhilrao/physionet-ich-dataset", target_dir, is_competition=False):
        success_count = 0
        logger.info("Rozpoczynam konwersję skanów NIfTI z PhysioNet na PNG.")
        
        nifti_files = []
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith('.nii'):
                    nifti_files.append((root, file))
        
        # Pasek postępu dla konwersji wszystkich plików NIfTI
        for root, file in tqdm(nifti_files, desc="Konwersja NIfTI z PhysioNet na PNG"):
            nii_path = os.path.join(root, file)
            patient_id = file.replace('.nii', '')
            patient_png_dir = os.path.join(png_dir, f"patient_{patient_id}")
            
            try:
                extract_nifti_to_pngs(nii_path, patient_png_dir)
                success_count += 1
            except (FileNotFoundError, OSError, RuntimeError) as conv_err:
                logger.error(f"Błąd konwersji pliku {file}: {conv_err}")
            except Exception as e:
                logger.error(f"Nieoczekiwany błąd podczas konwersji {file}: {e}")
                        
        logger.info(f"Zakończono. Pomyślnie przetworzono {success_count} pacjentów PhysioNet.")
        return True
    return False


def download_pretrained_models():
    """ Pobiera bazowe wagi modelu, omijając problemy Dockera z pobieraniem w locie. """
    models_dir = os.path.join(BASE_DIR, "output_data", "Models")
    os.makedirs(models_dir, exist_ok=True)
    
    weights_path = os.path.join(models_dir, "resnet_50_23dataset.pth")
    weights_url = "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50/resolve/main/resnet_50_23dataset.pth"
    
    if not os.path.exists(weights_path):
        logger.info("Brak pretrenowanych wag MedicalNet. Rozpoczynam bezpośrednie pobieranie.")
        try:
            # Timeout wymusza błąd zamiast nieskończonego zawieszenia
            response = requests.get(weights_url, stream=True, timeout=15)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(weights_path, 'wb') as file, tqdm(
                desc="Pobieranie resnet_50_23dataset.pth",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=8192):
                    size = file.write(data)
                    bar.update(size)
            logger.info("Zakończono pobieranie wag bazowych.")
        except (requests.exceptions.RequestException, OSError, ValueError) as e:
            logger.error(f"Błąd sieci lub zapisu podczas pobierania wag bazowych: {e}")
            if os.path.exists(weights_path):
                os.remove(weights_path) 
            sys.exit(1)
        except Exception as e:
            logger.error(f"Nieoczekiwany błąd podczas pobierania wag bazowych: {e}")
            if os.path.exists(weights_path):
                os.remove(weights_path) 
            sys.exit(1)
    else:
        logger.info("Pretrenowane wagi MedicalNet (resnet_50_23dataset.pth) już istnieją w output_data/Models.")


def upload_directory_to_minio(bucket_name, source_dir):
    """ Rekursywnie wysyła lokalny wygenerowany folder z obrazami do MinIO (Data Lake). """
    raw_endpoint = os.getenv("MINIO_ENDPOINT", "tbi-minio:9000")
    
    # Boto3 wymaga podania protokołu http:// lub https://
    if not raw_endpoint.startswith("http://") and not raw_endpoint.startswith("https://"):
        endpoint_url = f"http://{raw_endpoint}"
    else:
        endpoint_url = raw_endpoint

    # Pobieramy klucze do zmiennych, aby móc je zlogować
    access_key = os.getenv("MINIO_ACCESS_KEY", "admin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "SuperSecretPassword")
    
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1"
    )
    
    try:
        s3_client = boto3.client('s3', endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name="us-east-1")
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except ClientError:
            s3_client.create_bucket(Bucket=bucket_name)
            
        logger.info(f"Rozpoczynam wgrywanie plików z {source_dir} do bucketa {bucket_name}.")
        
        # Tworzenie listy wszystkich plików przed rozpoczęciem wysyłania do MinIO
        all_files = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                all_files.append(os.path.join(root, file))
                
        # Pasek postępu dla wgrywania plików do MinIO
        for file_path in tqdm(all_files, desc=f"Wgrywanie plików do MinIO ({bucket_name})"):
            s3_key = os.path.relpath(file_path, start=DATA_DIR)
            try:
                s3_client.upload_file(file_path, bucket_name, s3_key)
            except (ClientError, boto3.exceptions.Boto3Error, OSError) as e:
                logger.warning(f"Błąd wgrywania pliku {file_path}: {e}")
    except (ClientError, boto3.exceptions.Boto3Error, ValueError, OSError) as e:
        logger.error(f"Nieoczekiwany błąd inicjalizacji klienta/bucketa MinIO: {e}")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas wgrywania do MinIO: {e}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    download_pretrained_models()
    
    CQ500_RAW = os.path.join(DATA_DIR, 'cq500_raw')
    INFO_CSV = os.path.join(CQ500_RAW, 'Seg-CQ500', 'data', 'volumes', 'info.csv')
    RSNA_DIR = os.path.join(DATA_DIR, 'rsna_raw')
    RSNA_ZIP = os.path.join(RSNA_DIR, 'stage_2_train.zip')
    RSNA_CSV = os.path.join(RSNA_DIR, 'stage_2_train.csv')
    
    main_ct_path = os.path.join(DATA_DIR, 'dataset_ct_main')
    cq500_crnn_path = os.path.join(DATA_DIR, 'dataset_cq500')
    cq500_caps_path = os.path.join(DATA_DIR, 'dataset_cq500_capsnet')
    labels_main = os.path.join(main_ct_path, 'labels.csv')
    
    if os.path.exists(labels_main):
        logger.info("Zbiory treningowe z RSNA już istnieją w katalogu Data/.")
    else:
        logger.info("Inicjalizacja zbioru treningowego z RSNA w katalogu Data/.")
        if os.path.exists(main_ct_path):
            shutil.rmtree(main_ct_path, ignore_errors=True)
            
        os.makedirs(RSNA_DIR, exist_ok=True)
        
        # Zablokowane pobieranie z Kaggle
        if not os.path.exists(RSNA_ZIP) or not os.path.exists(RSNA_CSV):
            logger.error(f"KRYTYCZNY BŁĄD: Brakuje pliku {RSNA_ZIP} lub {RSNA_CSV}. Upewnij się, że są w folderze!")
            sys.exit(1)

        extracted_images_check = os.path.join(main_ct_path, 'images')
        if not os.path.exists(extracted_images_check) or len(os.listdir(extracted_images_check)) == 0:
            logger.info("Rozpakowywanie fizycznej struktury folderów RSNA ze zbioru ZIP.")
            os.makedirs(extracted_images_check, exist_ok=True)

        # Skrypt od razu przechodzi do wycinania obrazów z pliku ZIP
        success = prepare_rsna_training_data(RSNA_ZIP, RSNA_CSV)
        
        # Usuwanie gigantycznego pliku ZIP po zakończeniu przetwarzania, jeśli wszystko poszło dobrze
        if success and os.path.exists(RSNA_ZIP):
            logger.info(f"Przetwarzanie RSNA zakończone sukcesem. Usuwam gigantyczny plik ZIP: {RSNA_ZIP}")
            try:
                os.remove(RSNA_ZIP)
                logger.info("Plik ZIP został pomyślnie usunięty z dysku.")
            except Exception as e:
                logger.warning(f"Nie udało się usunąć pliku ZIP {RSNA_ZIP}: {e}")

    # Przygotowanie zewnętrznego zbioru testowego na CQ500
    labels_cq500 = os.path.join(cq500_crnn_path, 'labels.csv')
    if os.path.exists(labels_cq500) and os.path.exists(cq500_caps_path):
        logger.info("Zewnętrzny zbiór testowy CQ500 już istnieje w katalogu Data/.")
    else:
        logger.info("Inicjalizacja zewnętrznego zbioru testowego z CQ500 w katalogu Data/.")
        if os.path.exists(cq500_crnn_path):
            shutil.rmtree(cq500_crnn_path, ignore_errors=True)
        if os.path.exists(cq500_caps_path):
            shutil.rmtree(cq500_caps_path, ignore_errors=True)
            
        if download_kaggle_item("fooxmoon/cq500-dataset", CQ500_RAW, is_competition=False):
            if os.path.exists(INFO_CSV):
                prepare_cq500_evaluation_data(INFO_CSV, CQ500_RAW)
            else:
                logger.error(f"Brak pliku analitycznego '{INFO_CSV}' dla CQ500!")

    # Przygotowanie dodatkowego zbioru z PhysioNet (z Kaggle)
    physionet_path = os.path.join(DATA_DIR, 'dataset_physionet', 'images')
    if os.path.exists(physionet_path) and len(os.listdir(physionet_path)) > 0:
        logger.info("Zbiór PhysioNet już istnieje w katalogu Data/.")
    else:
        logger.info("Inicjalizacja pobierania i przetwarzania zbioru PhysioNet z Kaggle.")
        prepare_physionet_from_kaggle()

    logger.info("Rozpoczynam synchronizację wygenerowanych danych obrazowych z MinIO Data Lake.")
    vision_bucket = "tbi-vision-data"
    
    if os.path.exists(main_ct_path):
        upload_directory_to_minio(vision_bucket, main_ct_path)
    if os.path.exists(cq500_crnn_path):
        upload_directory_to_minio(vision_bucket, cq500_crnn_path)
    if os.path.exists(physionet_path):
        upload_directory_to_minio(vision_bucket, physionet_path)