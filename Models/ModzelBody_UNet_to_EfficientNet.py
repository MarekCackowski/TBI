# Konfiguracja środowiska i ograniczenie wątków backendów C++
import json
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2 as cv
cv.setNumThreads(0)  # Wyłączenie konfliktowej wielowątkowości OpenCV

# Wbudowane biblioteki standardowe Pythona
from contextlib import nullcontext
from dataclasses import dataclass, field
import gc
import logging
import math
from pathlib import Path
import sys
import zipfile
from typing import List, Optional, Tuple

# Przetwarzanie danych i formaty medyczne
import numpy as np
import pandas as pd
from PIL import Image
import pydicom

# Silnik Deep Learningu
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.ops import DeformConv2d
import torchvision.transforms.v2 as transforms

# Uczenie maszynowe, optymalizacja i metryki
import optuna
from sklearn.metrics import (
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

# Narzędzia pomocnicze i konfiguracja
from dotenv import load_dotenv
from tqdm import tqdm

# Inicjalizacja logowania i środowiska
load_dotenv()

# Inicjalizacja systemu logowania
def setup_logger(name: str = "crnn"):
    """ Centralny logger: konsole kieruje przez tqdm.write (nie rozbija paskow postepu),
        a kopie zapisuje do pliku. Zabezpieczony przed podwojna rejestracja handlerow. """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    class _TqdmStream:
        """ Strumien przekazujacy logi przez tqdm.write, zeby pasek sie nie dublowal. """
        @classmethod
        def write(cls, msg):
            logger.info(msg.rstrip())
        def flush(cls):
            sys.stdout.flush()

    ch = logging.StreamHandler(_TqdmStream)
    ch.setFormatter(fmt)
    fh = logging.FileHandler("crnn_training.log", encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

# Inicjalizacja systemu logowania
def setup_logger(name="CRNN", level=logging.INFO):
    class _TqdmStream:
        def write(self, buf):
            tqdm.write(buf, end="")

        def flush(self):
            pass

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Używamy instancji _TqdmStream(), żeby logi nie kolidowały z paskiem postępu tqdm
    handler = logging.StreamHandler(stream=_TqdmStream())
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    if not logger.handlers:
        logger.addHandler(handler)

    return logger

logger = setup_logger()


# Połączenia z infrastrukturą
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Globalne urządzenie dla całego skryptu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Standardowe parametry dla DataLoaderów eliminujące boilerplate
LOADER_KWARGS = {
    'num_workers': 4,
    'pin_memory': (DEVICE.type == 'cuda')
}

# Stałe globalne dla całego modułu
WINDOW_LEVEL = 40
WINDOW_WIDTH = 80
DEFAULT_BATCH_SIZE = 4
ACCUMULATION_STEPS = 8
SOFT_VOTING_ALPHA = 0.7
SOFT_VOTING_BETA = 0.3


def get_scaler():
    """ Obsługuje kompatybilność wsteczną dla torch.amp.GradScaler w zależności od wersji PyTorch. """
    try:
        return torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == 'cuda'))
    except (AttributeError, TypeError):
        logger.debug("Brak modułu torch.amp.GradScaler. Używam kompatybilnego wstecznie torch.cuda.amp.GradScaler.")
        return torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))


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

@dataclass
class EfficientNetTrainingConfig:
    batch_size: int = 4
    learning_rate: float = 1e-4
    routing_iters: int = 3
    num_epochs_main: int = 12
    num_epochs_fine: int = 6
    accumulation_steps: int = 8


def _has_images(folder: str) -> bool:
    """ Sprawdza, czy w podanym folderze lub jego podkatalogach znajdują się obrazy. """
    folder_path = Path(folder)
    if not folder_path.exists():
        return False
    return any(f.suffix.lower() in ('.png', '.jpg') for f in folder_path.rglob('*'))


def prepare_rsna_data(zip_path, csv_path, output_dir, limit_patients=10000, imaging_config: Optional[MedicalImagingConfig] = None):
    """ Odczytuje etykiety z pliku CSV i wypakowuje w locie określone plastry z archiwum ZIP RSNA 
        z uwzględnieniem okienkowania HU z MedicalImagingConfig (czyste logowanie). """
    if imaging_config is None:
        imaging_config = MedicalImagingConfig()

    if not os.path.exists(zip_path) or not os.path.exists(csv_path):
        logger.error(f"Brak pliku archiwum {zip_path} lub etykiet {csv_path}.")
        return False
        
    os.makedirs(os.path.join(output_dir, '0_normal'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '1_corpus_callosum_dai'), exist_ok=True)
    
    logger.info("Ładowanie metadanych RSNA z CSV.")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    if 'name' in df.columns and 'PatientID' not in df.columns:
        df = df.rename(columns={'name': 'PatientID'})
    elif 'index' in df.columns and 'PatientID' not in df.columns:
        df = df.rename(columns={'index': 'PatientID'})
        
    cq500_to_rsna = {
        'EDH': 'epidural', 'SDH': 'subdural', 'SAH': 'subarachnoid',
        'IPH': 'intraparenchymal', 'IVH': 'intraventricular'
    }
    df = df.rename(columns=cq500_to_rsna)
    
    patients = df['PatientID'].unique()
    if len(patients) > limit_patients:
        patients = patients[:limit_patients]
        
    logger.info(f"Rozpoczynam ekstrakcję surowych skanów dla {len(patients)} pacjentów.")
    
    wl, ww = imaging_config.window_brain
    vmin, vmax = wl - (ww / 2), wl + (ww / 2)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        zip_files = z.namelist()
        with tqdm(patients, desc="Ekstrakcja RSNA", leave=False) as pbar:
            for patient_id in pbar:
                is_dai = df[(df['PatientID'] == patient_id) & (df['Label'] == 1)].shape[0] > 0
                target_folder = '1_corpus_callosum_dai' if is_dai else '0_normal'
                patient_dir = os.path.join(output_dir, target_folder, patient_id)
                os.makedirs(patient_dir, exist_ok=True)
                
                patient_dicoms = [f for f in zip_files if patient_id in f and f.endswith('.dcm')]
                
                for dcm_file in patient_dicoms:
                    file_name = os.path.basename(dcm_file).replace('.dcm', '.png')
                    out_path = os.path.join(patient_dir, file_name)
                    
                    if not os.path.exists(out_path):
                        with z.open(dcm_file) as f:
                            dicom_data = pydicom.dcmread(f)
                            img_array = dicom_data.pixel_array.astype(np.float32)
                            
                            intercept = getattr(dicom_data, 'RescaleIntercept', 0.0)
                            slope = getattr(dicom_data, 'RescaleSlope', 1.0)
                            hu_image = img_array * slope + intercept
                            
                            img_norm = np.clip(hu_image, vmin, vmax)
                            img_norm = ((img_norm - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
                            cv.imwrite(out_path, img_norm)
                        
    logger.info(f"Zakończono ekstrakcję do {output_dir}")
    return True


def squash(x, dim=-1):
    """ Nieliniowa funkcja aktywacji dla kapsuł z zabezpieczeniem numerycznym. """
    x_fp32 = x.float()
    squared_norm = (x_fp32 ** 2).sum(dim=dim, keepdim=True)

    # Bezpieczniejsze skalowanie zapobiegające niedokładnościom zmiennoprzecinkowym
    scale = squared_norm / (1.0 + squared_norm)
    return (scale * x_fp32 / torch.sqrt(squared_norm + 1e-7)).to(x.dtype)


class PrimaryCaps(nn.Module):
    """ Warstwa Kapsuł Podstawowych w architekturze EfficientNet.
        Zadaniem tej warstwy jest przekształcenie niskopoziomowych map cech (wyciągniętych przez 
        bloki splotowe i deformowalne) w wektorowe reprezentacje. 
        Kluczowe mechanizmy:
        1. Spatial Awareness: Przed przejściem przez sploty kapsułowe, do map cech wstrzykiwana jest dwukanałowa siatka
        współrzędnych przestrzennych. Dzięki temu sieć nie traci informacji o położeniu struktur wewnątrz skanu tomograficznego.
        2. Przekształcenie wektorowe: Moduł operuje za pomocą listy niezależnych warstw splotowych, 
        z których każda generuje jeden kanał wewnętrzny wektora kapsuły, a całość kończy się 
        nieliniową aktywacją funkcji squash. """
    def __init__(self, num_capsules=32, in_channels=256, out_channels=8, kernel_size=3, padding=1, stride=2):
        super(PrimaryCaps, self).__init__()
        # Dodajemy +2 kanały na siatkę współrzędnych X i Y
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
            for _ in range(num_capsules)
        ])
        self.out_channels = out_channels

    def forward(self, x):
        batch_size, _, h, w = x.size()

        # Generowanie siatki z koordynatami (Spatial Awareness)
        grid_x = torch.linspace(-1, 1, w, device=x.device)
        grid_y = torch.linspace(-1, 1, h, device=x.device)
        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        
        coords = torch.stack([mesh_x, mesh_y], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Wstrzyknięcie koordynat przed wejściem do Kapsuł
        x = torch.cat([x, coords], dim=1)

        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), len(self.capsules) * u.size(3) * u.size(4), self.out_channels)
        return squash(u)

    
class DigitCaps(nn.Module):
    """ Warstwa Kapsuł Wysokopoziomowych z mechanizmem Attention Routing.
        Odpowiada za finalną klasyfikację w ramach podmodułu wizyjnego dla spoidła wielkiego 
        (rozróżnienie zdrowego ciała modzelowatego od patologicznego uszkodzenia aksonalnego / krwotoku).
        Zastępuje tradycyjny, bardzo powolny proces iteracyjnego dopasowywania podejściem Attention Routing,
        które jest znacznie szybsze i bardziej stabilne w kontekście wolumetrycznych danych medycznych. 
        Sprawdza ono również, które kapsuły podstawowe są najbardziej istotne dla danej klasyfikacji,
        co pozwala na lepszą interpretowalność modelu. """
    def __init__(self, num_capsules=2, num_routes=1568, in_channels=8, out_channels=16, routing_iters=3):
        super(DigitCaps, self).__init__()
        self.num_capsules = num_capsules
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_routes = num_routes
        
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels) * 0.01)

        # Skalowanie do Attention Routing
        self.scale = (out_channels ** 0.5)
        
        """ Dzięki temu sieć sama uczy się szukać anomalii (np. małych krwotoków),
            zamiast je wygaszać przez przyrównywanie do zdrowej większości obrazu. """
        self.query = nn.Parameter(torch.randn(1, 1, num_capsules, out_channels))

    def forward(self, x):
        batch_size = x.size(0)
        num_routes = x.size(1)

        # Dynamiczne dopasowanie wymiaru W, jeśli liczba tras wejściowych różni się od domyślnej
        if self.W.size(1) != num_routes:
            dtype = self.W.dtype
            self.W = nn.Parameter(torch.randn(1, num_routes, self.num_capsules, self.out_channels, self.in_channels, device=DEVICE, dtype=dtype) * 0.01)

        x = x.unsqueeze(2).unsqueeze(4)
        W = self.W.repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(W, x).squeeze(-1)

        # Wektor kontekstu modelujący globalną specyfikę anatomii ciała modzelowatego
        context = self.query.repeat(batch_size, 1, 1, 1)

        # Wykorzystanie context do wyliczania wag uwagi
        scores = (u_hat * context).sum(dim=-1, keepdim=True) / self.scale
        c_ij = torch.softmax(scores, dim=2)
        s_j = (c_ij * u_hat).sum(dim=1, keepdim=False)

        return squash(s_j, dim=-1)


class SpatialAttention(nn.Module):
    """ Tworzy maskę przestrzenną izolującą struktury mózgu od szumu tła i kości. """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Kompresja informacji kanałowej poprzez wyciągnięcie średniej i maksimum dla każdego piksela
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Złączenie map i przepuszczenie przez splot w celu wygenerowania jednokanałowej maski
        res = torch.cat([avg_out, max_out], dim=1)
        res = self.conv(res)
        
        # Sigmoid normalizuje maskę przestrzenną do przedziału [0, 1]
        return self.sigmoid(res)


class DualAttentionBlock(nn.Module):
    """ Łączy wygaszanie zbędnych filtrów ze skupianiem wzroku na uszkodzeniach (przestrzeń). """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(1, channels // reduction)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        # Aplikacja uwagi kanałowej (wyciszenie szumnych filtrów, podbicie tych z anomalią)
        x_ca = x * self.channel_attention(x)
        
        # Generowanie i zapisanie maski przestrzennej
        self.last_mask = self.spatial_attention(x_ca)
        
        # Ostateczne skupienie uwagi sieci na patologicznych fragmentach tkanki przez pomnożenie przez maskę przestrzenną
        return x_ca * self.last_mask


class DeformableBlock(nn.Module):
    """ Konwolucja dopasowująca się do patologicznie wygiętych i zmiażdżonych struktur mózgu. """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DeformableBlock, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * 3 * 3, kernel_size=3, padding=1, stride=stride)
        self.dcn = DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        # Obliczenie mapy przesunięć dla standardowego splotu
        offsets = self.offset_conv(x)
        
        # Splot odkształcalny dynamicznie dopasowujący siatkę próbkowania do krzywizn struktury mózgu
        out = self.dcn(x, offsets)
        
        # Normalizacja instancji i aktywacja
        out = self.bn(out)
        return self.prelu(out)


class VolumetricEfficientNet(nn.Module):
    """ Wolumetryczny EfficientNet do wykrywania uszkodzeń Ciała Modzelowatego w skanach CT.
        Zoptymalizowany pod kątem stabilności i zgodności wymiarów z kaskadą UNet. """
    def __init__(self, imaging_config: Optional[MedicalImagingConfig] = None, num_classes=2, backbone='efficientnet'):
        super(VolumetricEfficientNet, self).__init__()
        if imaging_config is None:
            imaging_config = MedicalImagingConfig()

        # 30 plastrów * 3 kanały BGR = 90 kanałów wejściowych
        input_channels = imaging_config.num_slices * 3
        
        # weights=None ustawia wagi na domyślne losowe inicjalizacje (efekt "szarych/pustych" wag przed trenowaniem)
        efficientnet = models.efficientnet_b0(weights=None)
        
        # Modyfikacja pierwszej warstwy pod 90 kanałów wolumetrycznych
        efficientnet.features[0][0] = nn.Conv2d(
            input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        self.features = efficientnet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Stabilna głowica kapsułowa / klasyfikująca
        self.primary_caps = PrimaryCaps(num_capsules=16, in_channels=1280, out_channels=8, kernel_size=1, padding=0, stride=1)
        self.digit_caps = DigitCaps(num_capsules=num_classes, num_routes=16 * 7 * 7, in_channels=8, out_channels=16, routing_iters=3)

    def forward(self, x):
        # x oczekuje kształtu: [Batch, Slices, Channels, H, W]
        batch_size, num_slices, C, H, W = x.size()
        
        # Spłaszczenie wymiarów Z i C, aby Conv2d otrzymało tensor 4D [Batch, Channels*Slices, H, W]
        x = x.view(batch_size, num_slices * C, H, W)
        
        x = self.features(x)
        
        # Przepuszczenie przez kapsuły w celu zachowania reprezentacji wektorowej
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        
        # Obliczenie prawdopodobieństw klas z normy wektorowej kapsuł
        classes_len = torch.sqrt((x ** 2).sum(dim=-1) + 1e-7)
        
        return classes_len


def preprocess_raw_ct_to_roi(raw_dir="raw_ct_pngs", out_dir="dataset_efficientnet", limit_patients_per_class=None, unet_model=None):
    """ Skanuje surowy folder, wycina Ciało Modzelowate i zapisuje do folderu treningowego. 
        Jeśli podano unet_model, używa sieci neuronowej do precyzyjnego wycięcia, 
        w przeciwnym razie spada na heurystykę geometryczną OpenCV. """
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    
    images_dir = raw_path / 'images'
    labels_file = raw_path / 'labels.csv'
    
    if not images_dir.exists() or not labels_file.exists():
        logger.info(f"Brak folderu 'images' lub pliku 'labels.csv' w '{raw_dir}'. Pomijam ekstrakcję ROI.")
        return False

    logger.info(f"Rozpoczynam wycinanie Ciała Modzelowatego z '{raw_dir}'.")
    processed_count = 0
    
    try:
        df = pd.read_csv(labels_file)
        df.columns = df.columns.str.strip()
        
        if 'name' in df.columns and 'PatientID' not in df.columns:
            df = df.rename(columns={'name': 'PatientID'})
        elif 'index' in df.columns and 'PatientID' not in df.columns:
            df = df.rename(columns={'index': 'PatientID'})
            
        cq500_to_rsna = {
            'EDH': 'epidural', 'SDH': 'subdural', 'SAH': 'subarachnoid',
            'IPH': 'intraparenchymal', 'IVH': 'intraventricular'
        }
        df = df.rename(columns=cq500_to_rsna)
        
    except Exception as e:
        logger.error(f"Nie udało się wczytać etykiet z pliku {labels_file}: {e}")
        return False

    label_cols = [c for c in df.columns if c != 'PatientID']
    df['is_sick'] = df[label_cols].sum(axis=1) > 0
    
    class_mapping = {
        '0_normal': df[~df['is_sick']]['PatientID'].astype(str).tolist(),
        '1_corpus_callosum_dai': df[df['is_sick']]['PatientID'].astype(str).tolist()
    }
    
    if unet_model is not None:
        unet_model.eval()
    
    for cls_name, patients in class_mapping.items():
        cls_out_dir = out_path / cls_name
        cls_out_dir.mkdir(parents=True, exist_ok=True)
        
        available_patients = [p for p in patients if (images_dir / p).exists()]
        
        if limit_patients_per_class is not None and len(available_patients) > limit_patients_per_class:
            available_patients = available_patients[:limit_patients_per_class]

        if not available_patients:
            continue
            
        with tqdm(available_patients, desc=f"Wycinanie UNet ({cls_name})", leave=False) as pbar:
            for patient_id in pbar:
                patient_raw_path = images_dir / patient_id
                patient_out_path = cls_out_dir / patient_id
                patient_out_path.mkdir(exist_ok=True)
                
                for slice_file in os.listdir(patient_raw_path):
                    if not slice_file.lower().endswith(('.png', '.jpg')):
                        continue
                        
                    img_path = str(patient_raw_path / slice_file)
                    out_file_path = str(patient_out_path / slice_file)

                    img_bgr = cv.imread(img_path, cv.IMREAD_COLOR)
                    if img_bgr is None:
                        continue

                    brain_channel = img_bgr[:, :, 2]
                    
                    try:
                        if unet_model is not None:
                            img_tensor = torch.from_numpy(brain_channel).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
                            
                            with torch.no_grad():
                                mask_pred = unet_model(img_tensor)
                                mask_pred = torch.sigmoid(mask_pred).squeeze().cpu().numpy()
                            
                            mask_binary = (mask_pred > 0.5).astype(np.uint8) * 255
                            contours, _ = cv.findContours(mask_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                            
                            if not contours:
                                continue
                                
                            largest_contour = max(contours, key=cv.contourArea)
                            x, y, w, h = cv.boundingRect(largest_contour)
                            
                            padding = 10
                            x1 = max(0, x - padding)
                            y1 = max(0, y - padding)
                            x2 = min(img_bgr.shape[1], x + w + padding)
                            y2 = min(img_bgr.shape[0], y + h + padding)
                        else:
                            _, thresh = cv.threshold(brain_channel, 5, 255, cv.THRESH_BINARY)
                            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                            
                            if not contours:
                                continue
                                
                            largest_contour = max(contours, key=cv.contourArea)
                            x, y, w, h = cv.boundingRect(largest_contour)
                            
                            if w < 20 or h < 20:
                                continue
                            
                            center_x = x + w // 2
                            cc_center_y = y + int(h * 0.42)
                            roi_w, roi_h = int(w * 0.28), int(h * 0.22)
                            
                            x1 = max(0, center_x - roi_w // 2)
                            y1 = max(0, cc_center_y - roi_h // 2)
                            x2 = min(img_bgr.shape[1], center_x + roi_w // 2)
                            y2 = min(img_bgr.shape[0], cc_center_y + roi_h // 2)
                    
                        cc_crop_bgr = img_bgr[y1:y2, x1:x2]
                        
                        if cc_crop_bgr.shape[0] < 5 or cc_crop_bgr.shape[1] < 5:
                            continue
                            
                        efficientnet_ready = cv.resize(cc_crop_bgr, (224, 224), interpolation=cv.INTER_CUBIC)
                        cv.imwrite(out_file_path, efficientnet_ready)
                        processed_count += 1
                    except cv.error as cve:
                        logger.debug(f"Błąd OpenCV podczas wycinania ROI dla pliku {slice_file}: {cve}")
                    except RuntimeError as re:
                        logger.error(f"Błąd operacji tensorowej/GPU podczas ekstrakcji ROI: {re}")
                        if "out of memory" in str(re).lower():
                            if DEVICE.type == 'cuda':
                                torch.cuda.empty_cache()
                            logger.warning("Zwolniono pamięć GPU po błędzie OOM.")
                    except Exception as inner_e:
                        logger.debug(f"Pominięto plaster z powodu nieoczekiwanego błędu: {inner_e}")
                    
    logger.info(f"Wycięto i przetworzono {processed_count} trójkanałowych przekrojów Ciała Modzelowatego.")
    return processed_count > 0


def calculate_hemispheric_symmetry(volumes):
    """ Oblicza indeks zachowania symetrii mózgu. """
    brain_slices = volumes[:, :, 2, :, :]
    W = brain_slices.shape[-1]
    mid = W // 2
    
    left_hemisphere = brain_slices[:, :, :, :mid]
    right_hemisphere = brain_slices[:, :, :, mid:]
    
    # Lustrzane odbicie prawej półkuli do porównania
    right_flipped = torch.flip(right_hemisphere, dims=[-1])
    
    # MSE jako szybka aproksymacja błędu SSIM
    mse = torch.nn.functional.mse_loss(left_hemisphere, right_flipped)
    return torch.exp(-mse * 10.0).item()


class RLAutoAugmentAgent(nn.Module):
    """ Agent steruje polityką zniekształceń, optymalizując dwie sprzeczne nagrody:
        1. Task Reward: Poprawa F1Score zmusza do trudnych augmentacji, przy jednoczesnym pozostawieniu modelowi szans podjęcia dobrej decyzji.
        2. Struct Reward: Zachowanie anatomii, które zapobiega niszczeniu skanów i umożliwia tworzenie augmentacji skoncentrowanych w okolicach aksonów. """
    def __init__(self, state_dim=3, action_dim=5, lr=1e-3, struct_lambda=0.2):
        super().__init__()
        self.struct_lambda = struct_lambda  # Waga określająca jak bardzo chronimy biologię

        # Aktor: zwraca parametry rozkładu Beta (eksploracja/eksploatacja)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim * 2) 
        )
        
        # Krytyk: ocenia wartość kontekstu, co służy do nauki redukcji wariancji
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        
        self.last_log_prob = None
        self.last_entropy = None
        self.last_state = None
        
    def select_augmentations(self, current_loss, struct_score=1.0, symmetry_score=1.0, deterministic=False):
        # Stan dla Krytyka poszerzony o ocenę "Sędziego Anatomii" oraz Symetrię
        state = torch.tensor([[current_loss, struct_score, symmetry_score]], dtype=torch.float32, device=DEVICE)
        self.last_state = state
        
        params = self.actor(state)
        
        # Rozkład Beta (parametry > 1.0 gwarantują bezpieczny, dzwonowy kształt rozkładu)
        alpha = func.softplus(params[:, :5]) + 1.0
        beta = func.softplus(params[:, 5:]) + 1.0
        
        dist = Beta(alpha, beta)
        
        if deterministic:
            # Faza wyzysku
            actions = alpha / (alpha + beta)
        else:
            # Faza eksploracji
            actions = dist.sample()
            self.last_log_prob = dist.log_prob(actions).sum(dim=-1)

            # Entropia posłuży do premiowania eksploracji w trudnych momentach
            self.last_entropy = dist.entropy().sum(dim=-1)
            
        return actions.detach().cpu().numpy()[0]
        
    def update(self, task_reward, struct_reward=0.0):
        if self.last_log_prob is None or self.last_state is None:
            return
        
        # Obliczenie nagrody całkowitej: Główny cel + Ochrona mikrostruktury
        total_reward = task_reward + (self.struct_lambda * struct_reward)
        reward_tensor = torch.tensor([total_reward], dtype=torch.float32, device=DEVICE)
        
        # Krytyk szacuje spodziewaną całkowitą nagrodę w zadanym kontekście
        value = self.critic(self.last_state).view(-1)
        
        # Weryfikujemy, o ile wylosowana akcja była lepsza niż zakładał Krytyk
        advantage = reward_tensor - value.detach()
        
        # Strata Aktora z bonusem entropijnym (kara za nadmierną pewność siebie na wczesnym etapie)
        actor_loss = -(self.last_log_prob * advantage + 0.05 * self.last_entropy).mean() 
        
        # Strata Krytyka (zbieganie do rzeczywistej połączonej wartości nagrody)
        critic_loss = func.mse_loss(value, reward_tensor)
        
        loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.last_log_prob = None
        self.last_entropy = None
        self.last_state = None


class CTCapsuleDataset(Dataset):
    """ Ładuje wolumeny wyciętego Ciała Modzelowatego. Aplikuje ukierunkowane augmentacje medyczne, aby symulować asymetrię
        pourazową, obrzęki i artefakty skanerów CT. Wzbogacone o interfejs dla Agenta RL. """
    def __init__(self, root_dir, imaging_config: Optional[MedicalImagingConfig] = None, is_train=True):
        if imaging_config is None:
            imaging_config = MedicalImagingConfig()
            
        self.root_dir = root_dir
        self.num_slices = imaging_config.num_slices
        self.is_train = is_train
        self.img_size = imaging_config.img_size
        self.patient_folders = []
        self.labels = []
        
        for label, class_name in enumerate(['0_normal', '1_corpus_callosum_dai']):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for patient in os.listdir(class_dir):
                patient_path = os.path.join(class_dir, patient)
                if os.path.isdir(patient_path):
                    self.patient_folders.append(patient_path)
                    self.labels.append(label)

        # Domyślne parametry polityki RL dla treningu
        self.p_affine = 1.0
        self.p_elastic = 0.5
        self.p_blur = 0.5
        self.p_jitter = 1.0
        self.p_erase = 0.35
        
        self.transform = self._get_transforms()

    def _get_base_normalization(self):
        """ Wspólny, powtarzalny blok końcowy dla każdych transformacji. """
        return [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

    def _get_transforms(self):
        """ Fabryka transformacji obsługująca tryb treningowy, ewaluacyjny oraz politykę RL. """
        if not self.is_train:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                *self._get_base_normalization()
            ])
        
        safe_p_erase = float(self.p_erase) * 0.20
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05))], p=self.p_affine),
            transforms.RandomApply([transforms.ElasticTransform(alpha=(5.0, 15.0), sigma=(4.0, 6.0))], p=self.p_elastic),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=self.p_blur),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=self.p_jitter),
            *self._get_base_normalization(),
            transforms.RandomErasing(p=safe_p_erase, scale=(0.01, 0.05), value='random')
        ])

    def set_resolution(self, img_size):
        """ Pozwala dynamicznie zmieniać rozdzielczość obrazków za pomocą fabryki. """
        self.img_size = img_size
        self.transform = self._get_transforms()

    def update_rl_policy(self, p_affine, p_elastic, p_blur, p_jitter, p_erase):
        """ Metoda dla Agenta RL aktualizująca parametry i przebudowująca potok transformacji w locie. """
        self.p_affine = p_affine
        self.p_elastic = p_elastic
        self.p_blur = p_blur
        self.p_jitter = p_jitter
        self.p_erase = p_erase
        self.transform = self._get_transforms()

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        patient_path = self.patient_folders[idx]
        label = self.labels[idx]
        
        slices = sorted([f for f in os.listdir(patient_path) if f.endswith(('.png', '.jpg'))])
        
        # Pusty tensor wolumenu
        volume_tensor = torch.zeros((self.num_slices, 3, self.img_size, self.img_size))
        
        if len(slices) == 0:
            # Brak zdjęć w folderze pacjenta, zwracamy wyzerowany wolumen, żeby nie przerywać treningu
            return volume_tensor, torch.tensor(label, dtype=torch.long)
        
        if len(slices) > self.num_slices:
            indices = np.linspace(0, len(slices) - 1, self.num_slices, dtype=int)
            slices = [slices[i] for i in indices]
        elif len(slices) < self.num_slices:
            padding_needed = self.num_slices - len(slices)
            slices = slices + [slices[-1]] * padding_needed 
        
        volumetric_seed = int(torch.randint(0, 2**32, (1,)).item())
        rng_state = torch.get_rng_state()
        
        for i, slice_file in enumerate(slices):
            img_path = os.path.join(patient_path, slice_file)
            
            with Image.open(img_path) as img:
                img_rgb = img.convert('RGB')
            img_bgr_array = np.array(img_rgb)[:, :, ::-1].copy()
            img = Image.fromarray(img_bgr_array)
            
            torch.manual_seed(volumetric_seed)
            volume_tensor[i] = self.transform(img)

        torch.set_rng_state(rng_state)
            
        return volume_tensor, torch.tensor(label, dtype=torch.long)


class DynamicErrorMarginLoss(nn.Module):
    """ Funkcja straty oparta na dynamicznym skupianiu na błędach z poprawną obsługą wektora pos_weight. """
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_val=0.5, base_gamma=2.0, pos_weight=None):
        super(DynamicErrorMarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_val = lambda_val
        self.base_gamma = base_gamma
        self.pos_weight = pos_weight

    def forward(self, capsule_probs, labels):
        capsule_probs = torch.clamp(capsule_probs, min=1e-7, max=1.0 - 1e-7)
        labels_one_hot = func.one_hot(labels.to(torch.int64), num_classes=2).float()
        
        present_error_loss = func.relu(self.m_plus - capsule_probs) ** 2
        absent_error_loss = func.relu(capsule_probs - self.m_minus) ** 2
        base_loss = labels_one_hot * present_error_loss + self.lambda_val * (1.0 - labels_one_hot) * absent_error_loss

        if self.pos_weight is not None:
            pw = self.pos_weight.to(capsule_probs.device)
            if pw.dim() == 1 and pw.shape[0] == 2:
                base_loss = base_loss * pw.unsqueeze(0)
            else:
                base_loss = base_loss * pw
        
        p_t = labels_one_hot * capsule_probs + (1.0 - labels_one_hot) * (1.0 - capsule_probs)
        error_magnitude = (1.0 - p_t)
        
        dynamic_gamma = self.base_gamma + (error_magnitude * 2.0)
        focal_weight = error_magnitude ** dynamic_gamma
        loss = base_loss * focal_weight
        
        return loss.sum(dim=1).mean() 


def objective(trial): 
    dataset_path = os.path.join(BASE_DIR, 'Data', 'dataset_efficientnet')
    
    imaging_config = MedicalImagingConfig()
    
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    routing_iters = trial.suggest_int('routing_iters', 2, 4)
    batch_size = trial.suggest_categorical('batch_size', [2, 4])
    
    dataset = CTCapsuleDataset(root_dir=dataset_path, imaging_config=imaging_config, is_train=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **LOADER_KWARGS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **LOADER_KWARGS)
    
    model = VolumetricEfficientNet(imaging_config=imaging_config, backbone='efficientnet')
    model.digit_caps.routing_iters = routing_iters
    model.to(DEVICE)

    criterion = DynamicErrorMarginLoss(base_gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

    scaler = get_scaler()
    
    num_epochs = 5
    best_val_loss = float('inf')
    
    with tqdm(range(num_epochs), desc=f"Optuna Trial {trial.number}") as optuna_pbar:
        for epoch in optuna_pbar:
            model.train()
            
            for volumes, labels in train_loader:
                volumes, labels = volumes.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                autocast_context = torch.amp.autocast(device_type='cuda') if DEVICE.type == 'cuda' else nullcontext()
                with autocast_context:
                    capsule_probs = model(volumes)
                    loss = criterion(capsule_probs, labels)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for volumes, labels in val_loader:
                    volumes, labels = volumes.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                    with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                        capsule_probs = model(volumes)
                        loss = criterion(capsule_probs, labels)
                    val_loss += loss.item()
                    
            val_loss /= len(val_loader)
            
            optuna_pbar.set_postfix({'Val Loss': f"{val_loss:.4f}"})
            
            trial.report(val_loss, epoch)
            if trial.should_prune():
                del model, optimizer, train_loader, val_loader
                if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
    return best_val_loss


class UNet(nn.Module):
    """ Lekki model UNet do precyzyjnej segmentacji Ciała Modzelowatego. """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = conv_block(128, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Enkoder odpowiadający za ekstrakcję cech
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        # Dekoder odpowiadający za rekonstrukcję maski z uwzględnieniem cech z enkodera (skip connections)
        d2 = self.up2(b)
        if d2.shape != e2.shape:
            d2 = func.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = func.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out_conv(d1)


class UNetDataset(Dataset):
    """ Dataset generujący maski za pomocą heurystyki OpenCV w locie do treningu UNetu. """
    def __init__(self, raw_dir, img_size=224, limit_samples=1500):
        self.image_paths = []
        images_dir = Path(raw_dir) / 'images'
        if images_dir.exists():
            for patient_dir in images_dir.iterdir():
                if patient_dir.is_dir():
                    for f in patient_dir.glob('*.[jp][pn]g'):
                        self.image_paths.append(str(f))

                        # Natychmiastowe przerwanie po zebraniu wystarczającej próbki
                        if limit_samples and len(self.image_paths) >= limit_samples:
                            break
                if limit_samples and len(self.image_paths) >= limit_samples:
                    break
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_bgr = cv.imread(img_path, cv.IMREAD_COLOR)
        if img_bgr is None:
            return torch.zeros((1, self.img_size, self.img_size)), torch.zeros((1, self.img_size, self.img_size))
        
        brain_channel = img_bgr[:, :, 2]
        brain_resized = cv.resize(brain_channel, (self.img_size, self.img_size))
        
        # Maska bazująca na heurystyce OpenCV (wzorzec dla UNetu)
        _, thresh = cv.threshold(brain_resized, 5, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(largest_contour)
            if w >= 20 and h >= 20:
                center_x = x + w // 2
                cc_center_y = y + int(h * 0.42)
                roi_w, roi_h = int(w * 0.28), int(h * 0.22)
                x1 = max(0, center_x - roi_w // 2)
                y1 = max(0, cc_center_y - roi_h // 2)
                x2 = min(self.img_size, center_x + roi_w // 2)
                y2 = min(self.img_size, cc_center_y + roi_h // 2)
                mask[y1:y2, x1:x2] = 1.0

        img_tensor = torch.from_numpy(brain_resized).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        return img_tensor, mask_tensor


def train_unet_model(raw_dir=None, epochs=3):
    """ Trenuje model UNet na maskach, aby precyzyjnie lokalizować spoidło. """
    if raw_dir is None:
        raw_dir = os.path.join(BASE_DIR, "Data", "dataset_ct_main")
        
    model_path = os.path.join(BASE_DIR, 'output_data', 'Models', 'unet_corpus_callosum.pth')
    if os.path.exists(model_path):
        logger.info("Znaleziono wytrenowany model UNet. Wczytuję wagi.")
        unet = UNet().to(DEVICE)
        unet.load_state_dict(torch.load(model_path, map_location=DEVICE))
        return unet

    logger.info("Rozpoczynam trening UNet przy użyciu labelingu z OpenCV.")
    dataset = UNetDataset(raw_dir)
    if len(dataset) == 0:
        logger.error("Brak danych do treningu UNet.")
        return None
        
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    unet = UNet().to(DEVICE)
    optimizer = optim.Adam(unet.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    unet.train()
    
    with tqdm(range(epochs), desc="Trening UNet") as unet_pbar:
        for epoch in unet_pbar:
            epoch_loss = 0.0
            epoch_dice = 0.0
            
            for imgs, masks in dataloader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                optimizer.zero_grad()
                preds = unet(imgs)
                loss = criterion(preds, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                preds_bin = (torch.sigmoid(preds) > 0.5).float()
                intersection = (preds_bin * masks).sum(dim=(2, 3))
                union = preds_bin.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                dice = (2. * intersection + 1e-5) / (union + 1e-5)
                epoch_dice += dice.mean().item()
                
            unet_pbar.set_postfix({
                'Strata': f"{epoch_loss/len(dataloader):.4f}", 
                'Dice': f"{epoch_dice/len(dataloader):.3f}"
            })

    os.makedirs(os.path.join(BASE_DIR, 'output_data', 'Models'), exist_ok=True)
    torch.save(unet.state_dict(), model_path)
    logger.info(" -> Zapisano ostateczne wagi modelu UNet.")
    return unet


class RawCTVolumeDataset(Dataset):
    """ Ładuje surowe wolumeny z folderu fine wraz z etykietami, aby w fazie Fine Tune wprowadzić złączony trening UNet z EfficientNet. """
    def __init__(self, raw_dir, num_slices=30, img_size=224):
        self.raw_dir = Path(raw_dir)
        self.num_slices = num_slices
        self.img_size = img_size
        self.patient_folders = []
        self.labels = []
        
        images_dir = self.raw_dir / 'images'
        labels_file = self.raw_dir / 'labels.csv'
        
        if images_dir.exists() and labels_file.exists():
            df = pd.read_csv(labels_file)
            df.columns = df.columns.str.strip()
            
            if 'name' in df.columns and 'PatientID' not in df.columns:
                df = df.rename(columns={'name': 'PatientID'})
            elif 'index' in df.columns and 'PatientID' not in df.columns:
                df = df.rename(columns={'index': 'PatientID'})
                
            cq500_to_rsna = {
                'EDH': 'epidural', 'SDH': 'subdural', 'SAH': 'subarachnoid',
                'IPH': 'intraparenchymal', 'IVH': 'intraventricular'
            }
            df = df.rename(columns=cq500_to_rsna)

            label_cols = [c for c in df.columns if c != 'PatientID']
            df['is_sick'] = df[label_cols].sum(axis=1) > 0
            
            patient_to_label = {str(row['PatientID']): int(row['is_sick']) for _, row in df.iterrows()}
            
            for p_dir in images_dir.iterdir():
                if p_dir.is_dir() and p_dir.name in patient_to_label:
                    self.patient_folders.append(p_dir)
                    self.labels.append(patient_to_label[p_dir.name])

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        patient_path = self.patient_folders[idx]
        label = self.labels[idx]
        
        slices = sorted([f for f in os.listdir(patient_path) if f.endswith(('.png', '.jpg'))])
        
        # Pusty tensor wolumenu
        volume_tensor = torch.zeros((self.num_slices, 3, self.img_size, self.img_size))
        
        if len(slices) == 0:
            return volume_tensor, torch.tensor(label, dtype=torch.long)
            
        if len(slices) > self.num_slices:
            indices = np.linspace(0, len(slices) - 1, self.num_slices, dtype=int)
            slices = [slices[i] for i in indices]
        elif len(slices) < self.num_slices:
            padding_needed = self.num_slices - len(slices)
            slices = slices + [slices[-1]] * padding_needed 

        for i, slice_file in enumerate(slices):
            img_path = os.path.join(patient_path, slice_file)
            img_bgr = cv.imread(img_path, cv.IMREAD_COLOR)
            if img_bgr is not None:
                img_resized = cv.resize(img_bgr, (self.img_size, self.img_size))
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                volume_tensor[i] = img_tensor
                
        return volume_tensor, torch.tensor(label, dtype=torch.long)
    

def _prepare_datasets_and_unet(raw_ct_path: str, dataset_path: str, raw_ft_path: str, dataset_ft_path: str):
    """ Weryfikuje strukturę katalogów, trenuje/wczytuje model UNet oraz ekstrakuje ROI. """

    if not _has_images(raw_ct_path):
        raise FileNotFoundError(f"Brak folderu z danymi CT: '{raw_ct_path}'. Uruchom najpierw przygotowanie danych.")

    unet_model = train_unet_model(raw_dir=raw_ct_path, epochs=3)

    if not _has_images(dataset_path):
        logger.info(f"Brak ROI w '{dataset_path}'. Wycinam Ciało Modzelowate za pomocą UNetu.")
        success = preprocess_raw_ct_to_roi(raw_dir=raw_ct_path, out_dir=dataset_path, limit_patients_per_class=None, unet_model=unet_model)
        if not success:
            raise RuntimeError("Nie udało się wykonać preprocessingu danych CT.")
    else:
        logger.info(f"Znaleziono gotowe pola ROI w '{dataset_path}'.")
        
    if _has_images(raw_ft_path) and not _has_images(dataset_ft_path):
        preprocess_raw_ct_to_roi(raw_dir=raw_ft_path, out_dir=dataset_ft_path, limit_patients_per_class=500, unet_model=unet_model)

    return unet_model


def _run_cross_validation_training(train_raw_dataset, val_raw_dataset, unet_model, best_params: dict, path_main: str):
    """ Zarządza procesem walidacji krzyżowej (StratifiedKFold), treningiem wstępnym RL i głównym treningiem EfficientNet. """
    dataset_len = len(train_raw_dataset)
    main_epochs = 12
    main_lr = best_params['lr']
    main_batch_size = best_params['batch_size']
    routing_iters = best_params['routing_iters']
    
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_labels = train_raw_dataset.labels
    
    os.makedirs('output_data/Models', exist_ok=True)
    best_val_f1 = 0.0
    
    for fold, (train_indices, val_indices) in enumerate(skf.split(np.zeros(dataset_len), all_labels)):
        logger.info(f"Rozpoczęto Fold {fold+1}/{n_splits}")
        
        train_dataset = torch.utils.data.Subset(train_raw_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_raw_dataset, val_indices)
        
        fold_train_labels = [all_labels[i] for i in train_indices]
        class_counts = [fold_train_labels.count(0), fold_train_labels.count(1)]
        sample_weights = [1.0 / max(1, class_counts[label]) for label in fold_train_labels]
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=main_batch_size, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)
        
        fold_val_labels = [all_labels[i] for i in val_indices]
        val_is_sick = np.array(fold_val_labels)
        pos_indices_rel = np.where(val_is_sick == 1)[0]
        neg_indices_rel = np.where(val_is_sick == 0)[0]
        
        if len(pos_indices_rel) > 0 and len(neg_indices_rel) > 0:
            n_total = len(val_indices)
            rng_sampler = np.random.default_rng(42)
            pos_sampled = rng_sampler.choice(pos_indices_rel, size=int(n_total * 0.75), replace=True)
            neg_sampled = rng_sampler.choice(neg_indices_rel, size=n_total - int(n_total * 0.75), replace=True)
            val_dataset_oit = torch.utils.data.Subset(val_dataset, np.concatenate([pos_sampled, neg_sampled]))
            val_loader = DataLoader(val_dataset_oit, batch_size=main_batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        else:
            val_loader = DataLoader(val_dataset, batch_size=main_batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        imaging_config = MedicalImagingConfig()
        model = VolumetricEfficientNet(imaging_config=imaging_config, backbone='efficientnet')
        model.digit_caps.routing_iters = routing_iters
        model = model.to(DEVICE)

        if hasattr(torch, 'compile') and DEVICE.type == 'cuda':
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                pass

        neg_count = max(1, class_counts[0])
        pos_count = max(1, class_counts[1])
        raw_w_pos = neg_count / pos_count
        smooth_w_pos = float(np.clip(np.log1p(raw_w_pos) + 1.0, 1.0, 10.0))
        weights = torch.tensor([1.0, smooth_w_pos], dtype=torch.float32).to(DEVICE)

        criterion = DynamicErrorMarginLoss(base_gamma=1.5)
        rl_agent = RLAutoAugmentAgent(state_dim=3, action_dim=5, lr=1e-3).to(DEVICE)
        previous_f1, previous_struct_score, previous_sym_score = 0.0, 1.0, 1.0

        optimizer = optim.Adam(model.parameters(), lr=main_lr, weight_decay=1e-2, fused=(DEVICE.type == 'cuda'))
        scaler = get_scaler()

        warmup_epochs = 2
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, main_epochs - warmup_epochs))
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        patience, patience_counter = 4, 0

        for epoch in range(main_epochs):
            current_res = 112 if epoch < 4 else 224
            train_raw_dataset.set_resolution(current_res)
            val_raw_dataset.set_resolution(current_res)

            rl_probs = rl_agent.select_augmentations(current_loss=previous_f1, struct_score=previous_struct_score, symmetry_score=previous_sym_score)
            train_raw_dataset.update_rl_policy(*rl_probs)

            model.train()
            train_loss = 0.0
            
            with tqdm(train_loader, desc=f"F[{fold+1}] E[{epoch+1}/{main_epochs}] Train", leave=False) as train_pbar:
                for volumes, labels in train_pbar:
                    volumes, labels = volumes.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    
                    with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                        capsule_probs = model(volumes)
                        loss = criterion(capsule_probs, labels)
                        
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                    train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            train_loss /= len(train_loader)
            scheduler.step()
            
            model.eval()
            val_loss, val_epoch_labels, val_epoch_probs = 0.0, [], []
            unet_confidences, symmetry_scores = [], []
            
            with tqdm(val_loader, desc=f"F[{fold+1}] E[{epoch+1}/{main_epochs}] Val", leave=False) as val_pbar:
                with torch.no_grad():
                    for volumes, labels in val_pbar:
                        volumes, labels = volumes.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                        autocast_context = torch.amp.autocast(device_type='cuda') if DEVICE.type == 'cuda' else nullcontext()
                        with autocast_context:
                            capsule_probs = model(volumes)
                            loss = criterion(capsule_probs, labels)
                            
                            B, S, C, H, W = volumes.shape
                            brain_slices = volumes[:, :, 2:3, :, :].contiguous().view(B * S, 1, H, W)
                            mask_pred = torch.sigmoid(unet_model(brain_slices))
                            unet_confidences.append(mask_pred.mean().item())
                            
                            symmetry_scores.append(calculate_hemispheric_symmetry(volumes))
                            
                        val_loss += loss.item()
                        val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                        val_epoch_labels.extend(labels.cpu().numpy())
                        val_epoch_probs.extend(capsule_probs[:, 1].cpu().numpy())
                    
            val_loss /= len(val_loader)

            current_unet_score = sum(unet_confidences) / max(1, len(unet_confidences))
            current_sym_score = sum(symmetry_scores) / max(1, len(symmetry_scores))
            current_struct_score = (current_unet_score * 0.5) + (current_sym_score * 0.5)
            
            all_labels_np = np.array(val_epoch_labels)
            all_probs_np = np.array(val_epoch_probs)
            
            try:
                precisions, recalls, thresholds = precision_recall_curve(all_labels_np, all_probs_np)
                if len(thresholds) <= 1:
                    best_threshold = float(np.median(all_probs_np)) if len(all_probs_np) > 0 else 0.5
                else:
                    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
                    best_threshold = float(np.clip(thresholds[np.argmax(f1_scores)], 0.1, 0.9))
            except Exception:
                best_threshold = float(np.median(all_probs_np)) if len(all_probs_np) > 0 else 0.5
                
            all_preds = (all_probs_np >= best_threshold).astype(int)
            f1 = fbeta_score(all_labels_np, all_preds, beta=1, average='macro', zero_division=0)
            recall = recall_score(all_labels_np, all_preds, pos_label=1, zero_division=0) * 100
            precision = precision_score(all_labels_np, all_preds, pos_label=1, zero_division=0) * 100
            auc = roc_auc_score(all_labels_np, all_probs_np) if len(np.unique(all_labels_np)) > 1 else 0.5
                
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()

            logger.info(f"Fold [{fold+1}/{n_splits}] Epoka [{epoch+1}/{main_epochs}] | Próg: {best_threshold:.2f} | "
                        f"Loss T/V: {train_loss:.4f}/{val_loss:.4f} | "
                        f"AUC: {auc:.3f} | F1: {f1:.3f} | Prec: {precision:.1f}% | Sens: {recall:.1f}%")

            if f1 > best_val_f1:
                best_val_f1 = f1
                os.makedirs(os.path.join(BASE_DIR, 'output_data', 'Models'), exist_ok=True)
                torch.save(model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(), path_main)
                logger.info(f" ---> Zapisano nowy najlepszy model MAIN (F1: {f1:.3f})")
                patience_counter = 0 
            else:
                patience_counter += 1

            if patience_counter >= patience:
                current_lr = optimizer.param_groups[0]['lr']
                if current_lr > 1e-7:
                    logger.info(f" ---> Redukcja LR. Kontynuacja Fold {fold+1}.")
                    for param_group in optimizer.param_groups: param_group['lr'] *= 0.5
                    patience_counter = 0
                else:
                    logger.info(f" ---> Early stopping. Przerywam Fold {fold+1}.")
                    break

            task_reward = f1 - previous_f1
            rl_agent.update(task_reward=task_reward, struct_reward=current_struct_score)

            previous_f1 = f1
            previous_struct_score = current_struct_score
            previous_sym_score = current_sym_score


def _run_fine_tune_phase(path_main: str, path_fine: str, raw_ct_path: str, raw_ft_path: str, train_raw_dataset, val_raw_dataset):
    """ Realizuje złączony Fine Tuning (UNet + EfficientNet) z gradient accumulation oraz generuje XAI. """
    fine_epochs = 6
    fine_lr = 1e-5
    logger.info(f"Rozpoczynam złączony Fine Tuning (UNet + EfficientNet) przez {fine_epochs} epok.")

    gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    ft_batch_size = 1
    accumulation_steps = 8
    
    if os.path.exists(raw_ft_path):
        fine_raw_dataset = RawCTVolumeDataset(raw_dir=raw_ft_path, num_slices=30, img_size=224)
        fine_train_loader = DataLoader(fine_raw_dataset, batch_size=ft_batch_size, shuffle=True, num_workers=2, pin_memory=True) if len(fine_raw_dataset) > 0 else DataLoader(train_raw_dataset, batch_size=ft_batch_size, shuffle=True)
    else:
        fine_train_loader = DataLoader(train_raw_dataset, batch_size=ft_batch_size, shuffle=True)
        
    val_loader = DataLoader(val_raw_dataset, batch_size=ft_batch_size, shuffle=False)

    imaging_config = MedicalImagingConfig()
    model = VolumetricEfficientNet(imaging_config=imaging_config, backbone='efficientnet').to(DEVICE)
    model.load_state_dict(torch.load(path_main, map_location=DEVICE))
    
    unet_model = train_unet_model(raw_dir=raw_ct_path)
    
    optimizer_joint = optim.Adam(list(model.parameters()) + list(unet_model.parameters()), lr=fine_lr, weight_decay=1e-2)
    scaler = get_scaler()
    
    neg_ft = train_raw_dataset.labels.count(0)
    pos_ft = train_raw_dataset.labels.count(1)
    pos_weight_val = neg_ft / max(1, pos_ft)
    criterion_ft = DynamicErrorMarginLoss(base_gamma=2.0, pos_weight=torch.tensor([1.0, pos_weight_val]))
    
    best_val_f2 = 0.0
    optimizer_joint.zero_grad()

    for epoch in range(fine_epochs):
        model.train()
        unet_model.train()
        train_loss = 0.0
        
        with tqdm(fine_train_loader, desc=f"FT E[{epoch+1}/{fine_epochs}] Train", leave=False) as train_pbar:
            for i, (volumes, labels) in enumerate(train_pbar):
                volumes, labels = volumes.to(DEVICE), labels.to(DEVICE)
                
                with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                    B, S, C, H, W = volumes.shape
                    brain_slices = volumes[:, :, 2:3, :, :].contiguous().view(B * S, 1, H, W)
                    mask_pred = torch.sigmoid(unet_model(brain_slices)).view(B, S, 1, H, W)
                    processed_volumes = volumes * mask_pred
                    
                    capsule_probs = model(processed_volumes)
                    loss = criterion_ft(capsule_probs, labels)
                    loss = loss / accumulation_steps
                    
                scaler.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(fine_train_loader):
                    scaler.unscale_(optimizer_joint)
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(unet_model.parameters()), max_norm=1.0)
                    scaler.step(optimizer_joint)
                    scaler.update()
                    optimizer_joint.zero_grad()

                real_loss = loss.item() * accumulation_steps
                train_loss += real_loss
                train_pbar.set_postfix({'loss': f"{real_loss:.4f}"})
            
        train_loss /= max(1, len(fine_train_loader))
        
        model.eval()
        unet_model.eval()
        val_loss = 0.0
        all_labels_ft, all_probs_ft = [], []
        
        with tqdm(val_loader, desc=f"FT E[{epoch+1}/{fine_epochs}] Val", leave=False) as val_pbar:
            with torch.no_grad():
                for volumes, labels in val_pbar:
                    volumes, labels = volumes.to(DEVICE), labels.to(DEVICE)
                    with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                        B, S, C, H, W = volumes.shape
                        brain_slices = volumes[:, :, 2:3, :, :].contiguous().view(B * S, 1, H, W)
                        mask_pred = torch.sigmoid(unet_model(brain_slices)).view(B, S, 1, H, W)
                        processed_volumes = volumes * mask_pred
                        
                        capsule_probs = model(processed_volumes)
                        loss = criterion_ft(capsule_probs, labels)
                        
                    val_loss += loss.item()
                    val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                    all_labels_ft.extend(labels.cpu().numpy())
                    all_probs_ft.extend(capsule_probs[:, 1].cpu().numpy())
                
        val_loss /= max(1, len(val_loader))
        
        if len(all_labels_ft) > 0:
            all_labels_np = np.array(all_labels_ft)
            all_probs_np = np.array(all_probs_ft)
            try:
                precisions, recalls, thresholds = precision_recall_curve(all_labels_np, all_probs_np)
                if len(thresholds) <= 1:
                    best_threshold = float(np.median(all_probs_np)) if len(all_probs_np) > 0 else 0.5
                else:
                    f2_scores = (5 * precisions[:-1] * recalls[:-1]) / (4 * precisions[:-1] + recalls[:-1] + 1e-10)
                    best_threshold = float(np.clip(thresholds[np.argmax(f2_scores)], 0.1, 0.9))
            except Exception:
                best_threshold = float(np.median(all_probs_np)) if len(all_probs_np) > 0 else 0.5
                
            all_preds = (all_probs_np >= best_threshold).astype(int)
            accuracy = (all_preds == all_labels_np).mean() * 100
            f2 = fbeta_score(all_labels_np, all_preds, beta=2, average='macro', zero_division=0)
            recall = recall_score(all_labels_np, all_preds, pos_label=1, zero_division=0) * 100
            precision = precision_score(all_labels_np, all_preds, pos_label=1, zero_division=0) * 100
            auc = roc_auc_score(all_labels_np, all_probs_np) if len(np.unique(all_labels_np)) > 1 else 0.5
        else:
            best_threshold, f2, precision, recall, auc, accuracy = 0.5, 0.0, 0.0, 0.0, 0.5, 0.0
        
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()

        logger.info(f'FT Epoka [{epoch+1}/{fine_epochs}] | Próg: {best_threshold:.2f} | '
                    f'Loss T/V: {train_loss:.4f}/{val_loss:.4f} | '
                    f'BAcc: {accuracy:.1f}% | AUC: {auc:.3f} | F2: {f2:.3f} | Prec: {precision:.1f}% | Sens: {recall:.1f}%')
        
        if f2 > best_val_f2:
            best_val_f2 = f2
            torch.save(model.state_dict(), path_fine)
            torch.save(unet_model.state_dict(), os.path.join(BASE_DIR, 'output_data', 'Models', 'unet_corpus_callosum_finetune.pth'))
            logger.info(f" ---> Zapisano nowe najlepsze złączone wagi kaskady (F2: {f2:.3f})")

    # Po zakończeniu fine tuningu i zapisaniu wag modeli, trenujemy Agenta RL od polityki augmentacji
    val_loader = DataLoader(val_raw_dataset, batch_size=4, shuffle=False)
    
    if os.path.exists(path_main) and os.path.exists(path_fine):
        model_main = VolumetricEfficientNet(imaging_config=MedicalImagingConfig()).to(DEVICE)
        model_main.load_state_dict(torch.load(path_main, map_location=DEVICE))
        
        model_fine = VolumetricEfficientNet(imaging_config=MedicalImagingConfig()).to(DEVICE)
        model_fine.load_state_dict(torch.load(path_fine, map_location=DEVICE))
        
        train_routing_agent(model_main, model_fine, val_loader, thresh_fine=0.50, thresh_main=0.55, epochs=15)
    else:
        logger.warning("Brak kompletnych wag Main/Fine na dysku. Pomijam osobny trening Agenta RL.")
    
    # Trening agenta na zbiorze walidacyjnym
    train_routing_agent(model_main, model_fine, val_loader, thresh_fine=0.50, thresh_main=0.55, epochs=15)

    # Generowanie XAI
    logger.info("Generowanie wizualizacji decyzyjnej XAI dla przykładowego pacjenta.")
    heatmaps_dir = os.path.join(BASE_DIR, 'output_data', 'Models', 'heatmaps_efficientnet')
    os.makedirs(heatmaps_dir, exist_ok=True)
    model.eval()

    if len(val_loader) > 0:
        with torch.no_grad():
            volumes, _ = next(iter(val_loader))
            volumes = volumes.to(DEVICE)
            
            B, S, C, H, W = volumes.shape
            brain_slices = volumes[:, :, 2:3, :, :].contiguous().view(B * S, 1, H, W)
            mask_pred = torch.sigmoid(unet_model(brain_slices)).view(B, S, 1, H, W)
            processed_volumes = volumes * mask_pred
            
            model(processed_volumes)
            
            attention_map = None
            for module in model.modules():
                if hasattr(module, 'last_mask') and getattr(module, 'last_mask') is not None:
                    attention_map = module.last_mask[0, 0].detach().cpu().numpy()
                    break
                    
            if attention_map is None:
                attention_map = np.zeros((14, 14))
            
            vol_np = processed_volumes[0].cpu().numpy()
            vol_np = np.squeeze(vol_np)
            if vol_np.ndim == 3:
                img = vol_np[15 if vol_np.shape[0] > 15 else 0]
            elif vol_np.ndim == 4:
                img = vol_np[0, 15 if vol_np.shape[1] > 15 else 0]
            else:
                img = vol_np
                
            img = np.clip(((img * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-5)
            heatmap = cv.applyColorMap(np.uint8(255 * attention_map), cv.COLORMAP_JET)
            superimposed = cv.addWeighted(cv.cvtColor(img, cv.COLOR_GRAY2BGR), 0.5, heatmap, 0.5, 0)
            cv.imwrite(os.path.join(heatmaps_dir, 'xai_corpus_callosum_overlay.jpg'), superimposed)


def train_efficientnet():
    """ Główny orkiestrator kaskady EfficientNet (Optuna -> K-Fold z RL -> Fine Tune). """
    raw_ct_path = os.path.join(BASE_DIR, 'Data', 'dataset_ct_main')
    dataset_path = os.path.join(BASE_DIR, 'Data', 'dataset_efficientnet')
    raw_ft_path = os.path.join(BASE_DIR, 'Data', 'raw_ct_pngs_finetune')
    dataset_ft_path = os.path.join(BASE_DIR, 'Data', 'dataset_efficientnet_finetune')

    path_main = os.path.join(BASE_DIR, 'output_data', 'Models', 'efficientnet_corpus_callosum_main.pth')
    path_fine = os.path.join(BASE_DIR, 'output_data', 'Models', 'efficientnet_corpus_callosum_finetune.pth')

    logger.info(f"Uruchamianie kaskady na urządzeniu: {DEVICE}")

    if DEVICE.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Poprawione wywołanie (bez podawania DEVICE na końcu)
    unet_model = _prepare_datasets_and_unet(raw_ct_path, dataset_path, raw_ft_path, dataset_ft_path)

    imaging_config = MedicalImagingConfig()
    train_raw_dataset = CTCapsuleDataset(root_dir=dataset_path, imaging_config=imaging_config, is_train=True)
    val_raw_dataset = CTCapsuleDataset(root_dir=dataset_path, imaging_config=imaging_config, is_train=False)
    
    if len(train_raw_dataset) == 0:
        logger.info("Dataset dla fazy main jest pusty!")
        raise RuntimeError("Dataset dla fazy main jest pusty!")

    if not os.path.exists(path_main):
        logger.info("Optymalizacja Bayesowska hiperparametrów dla EfficientNet.")
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3, interval_steps=1)
        )
        study.optimize(objective, n_trials=5)
        best_params = study.best_params
        logger.info(f"Znaleziono optymalne hiperparametry: {best_params}")

        _run_cross_validation_training(train_raw_dataset, val_raw_dataset, unet_model, best_params, path_main)

    if not os.path.exists(path_fine):
        _run_fine_tune_phase(path_main, path_fine, raw_ct_path, raw_ft_path, train_raw_dataset, val_raw_dataset)

    logger.info("Wyliczanie optymalnych progów Youden's J na zbiorze walidacyjnym (RSNA).")
    
    val_loader = DataLoader(val_raw_dataset, batch_size=4, shuffle=False, **LOADER_KWARGS)
    model_main = VolumetricEfficientNet(imaging_config=imaging_config, backbone='efficientnet').to(DEVICE)
    model_main.load_state_dict(torch.load(path_main, map_location=DEVICE))
    model_main.eval()

    model_fine = VolumetricEfficientNet(imaging_config=imaging_config, backbone='efficientnet').to(DEVICE)
    model_fine.load_state_dict(torch.load(path_fine, map_location=DEVICE))
    model_fine.eval()
    
    calib_labels, calib_probs_fine, calib_probs_main = [], [], []
    with torch.no_grad():
        for volumes, labels in val_loader:
            volumes = volumes.to(DEVICE)
            # Aplikacja ROI jeśli UNet jest dostępny
            if unet_model is not None:
                B, S, C, H, W = volumes.shape
                brain_slices = volumes[:, :, 2:3, :, :].contiguous().view(B * S, 1, H, W)
                mask_pred = torch.sigmoid(unet_model(brain_slices)).view(B, S, 1, H, W)
                volumes = volumes * mask_pred
                
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                calib_probs_fine.extend(model_fine(volumes)[:, 1].cpu().numpy())
                calib_probs_main.extend(model_main(volumes)[:, 1].cpu().numpy())
            calib_labels.extend(labels.numpy())

    calib_labels = np.array(calib_labels)
    opt_threshold_fine, opt_threshold_main = 0.5, 0.5
    
    if len(np.unique(calib_labels)) > 1:
        fpr_f, tpr_f, thresh_f = roc_curve(calib_labels, calib_probs_fine)
        if len(thresh_f) > 0: opt_threshold_fine = float(thresh_f[np.argmax(tpr_f - fpr_f)])
        
        fpr_m, tpr_m, thresh_m = roc_curve(calib_labels, calib_probs_main)
        if len(thresh_m) > 0: opt_threshold_main = float(thresh_m[np.argmax(tpr_m - fpr_m)])

    params_path = os.path.join(BASE_DIR, 'output_data', 'Models', 'efficientnet_cascade_params.json')
    with open(params_path, 'w') as f:
        json.dump({'THRESH_FINE': opt_threshold_fine, 'THRESH_MAIN': opt_threshold_main}, f)
        
    logger.info(f"Zamrożono optymalne progi decyzyjne z RSNA: Fine={opt_threshold_fine:.3f}, Main={opt_threshold_main:.3f}")

    logger.info("Trening kaskadowy zakończony pomyślnie!")


def calibrate_experimental_soft_voting_weights(model_main: nn.Module, model_fine: nn.Module, val_loader: DataLoader) -> Tuple[float, float]:
    """Eksperymentalnie dobiera wagi miękkiego głosowania (alpha, beta) na zbiorze walidacyjnym metodą Grid Search."""
    model_main.eval()
    model_fine.eval()
    
    best_alpha, best_f2 = 0.5, -1.0
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    all_labels, all_p_caps, all_c_unet = [], [], []
    with torch.no_grad():
        for volumes, labels in val_loader:
            volumes = volumes.to(DEVICE)
            B, S, C, H, W = volumes.shape
            
            p_caps = model_main(volumes)[:, 1].cpu().numpy()
            all_labels.extend(labels.numpy())
            all_p_caps.extend(p_caps)
            all_c_unet.extend([0.8] * B)  # Aproksymacja pewności segmentacji
            
    all_labels_np = np.array(all_labels)
    all_p_caps_np = np.array(all_p_caps)
    all_c_unet_np = np.array(all_c_unet)
    
    for alpha in alphas:
        beta = 1.0 - alpha
        p_final = (alpha * all_p_caps_np) + (beta * all_c_unet_np)
        f2 = fbeta_score(all_labels_np, (p_final >= 0.5).astype(int), beta=2, average='macro', zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_alpha = alpha
            
    logger.info(f"Eksperymentalnie dobrane wagi soft voting -> Alpha: {best_alpha:.1f}, Beta: {1.0 - best_alpha:.1f} (F2: {best_f2:.3f})")
    return best_alpha, 1.0 - best_alpha


def uncertainty_aware_soft_voting(unet_model, efficientnet_model, volumes_tensor):
    """ Wykonuje miękkie głosowanie łącząc pewność segmentacji UNetu z prawdopodobieństwem klasyfikacji EfficientNetu. """
    unet_model = unet_model.to(DEVICE)
    efficientnet_model = efficientnet_model.to(DEVICE)
    
    unet_model.eval()
    efficientnet_model.eval()
    
    volumes_tensor = volumes_tensor.to(DEVICE)
    B, S, C, H, W = volumes_tensor.shape
    unet_confidences = []
    
    with torch.no_grad():
        for s in range(S):
            slice_bgr = volumes_tensor[:, s, :, :, :]
            slice_gray = slice_bgr[:, 2:3, :, :].to(DEVICE)
            
            mask_logits = unet_model(slice_gray)
            mask_probs = torch.sigmoid(mask_logits)
            
            # Tworzymy maskę obszarów potencjalnie zmienionych chorobowo
            roi_mask = mask_probs > 0.1
            
            # Liczymy średnią pewność tylko dla wyłapanych struktur
            if roi_mask.sum() > 0:
                conf = mask_probs[roi_mask].mean().item()
            else:
                conf = 0.0
                
            unet_confidences.append(conf)
            
        c_unet = sum(unet_confidences) / len(unet_confidences)
        
        capsule_probs = efficientnet_model(volumes_tensor)
        p_caps = capsule_probs[:, 1]
        
        alpha = 0.7  
        beta = 0.3   
        
        if p_caps.ndim > 0 and len(p_caps) > 1:
            p_final = (alpha * p_caps.cpu().numpy()) + (beta * c_unet)
        else:
            p_final = (alpha * p_caps.item()) + (beta * c_unet)
            
    return p_final, c_unet


class TwoStageRoutingAgent(nn.Module):
    """ Agent RL dla kaskady medycznej CT: dynamicznie decyduje na podstawie stanu,
        czy przepuścić przypadek do modelu Main, czy zakończyć predykcję na modelu Fine Tune. """
    def __init__(self, state_dim=3, action_dim=2, lr=1e-3):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)  # [0: Zakończ na Fine Tune, 1: Wyślij do modelu Main]
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def forward(self, state):
        return self.policy(state)

    def update(self, log_prob, reward):
        """ Aktualizacja wag agenta metodą Policy Gradient. Dba o to, żeby promować akcje skutkujące poprawną diagnozą
            oraz karać sieć za decyzje skutkujące błędami predykcji. """
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
        loss = -log_prob * reward_tensor
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train_routing_agent(model_main: nn.Module, model_fine: nn.Module, val_loader: DataLoader, 
                        thresh_fine: float, thresh_main: float, epochs: int = 15) -> TwoStageRoutingAgent:
    """ Trenuje Agenta RL na zbiorze RSNA.  Agent uczy się kiedy ufać modelowi Fine Tune wokół progu J-Youdena, 
        a kiedy przekazać przypadek do modelu Main. """
    logger.info("Rozpoczynam trening Agenta RL na zbiorze walidacyjnym.")
    
    agent = TwoStageRoutingAgent(state_dim=3, action_dim=2, lr=1e-3).to(DEVICE)
    agent.train()
    model_main.eval()
    model_fine.eval()
    
    val_data = []
    with torch.no_grad():
        for volumes, labels in val_loader:
            volumes = volumes.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                p_fine = model_fine(volumes)[:, 1].cpu()
                p_main = model_main(volumes)[:, 1].cpu()
            val_data.append((p_fine, p_main, labels))

    with tqdm(range(epochs), desc="Trening Agenta RL") as agent_pbar:
        for epoch in agent_pbar:
            total_reward = 0.0
            for p_fine_batch, p_main_batch, labels_batch in val_data:
                for i in range(len(labels_batch)):
                    pf = p_fine_batch[i].item()
                    pm = p_main_batch[i].item()
                    lbl = labels_batch[i].item()
                    
                    p_cls = torch.tensor([1.0 - pf, pf])
                    entropy = -torch.sum(p_cls * torch.log(p_cls + 1e-10)).item()
                    state = torch.tensor([[pf - thresh_fine, pm - thresh_main, entropy]], dtype=torch.float32, device=DEVICE)
                    
                    probs = agent.policy(state)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    pred = (pf >= thresh_fine) if action.item() == 0 else (pm >= thresh_main)
                    
                    if pred == lbl:
                        reward = 1.0
                    else:
                        reward = -2.0 if pred == 1 and lbl == 0 else -1.5 
                        
                    agent.update(log_prob, reward)
                    total_reward += reward
                    
            agent_pbar.set_postfix({'Średnia Nagroda': f"{total_reward / len(val_loader.dataset):.3f}"})

    agent_path = os.path.join(BASE_DIR, 'output_data', 'Models', 'routing_agent.pth')
    torch.save(agent.state_dict(), agent_path)
    logger.info(" -> Zapisano wytrenowane wagi Agenta RL.")
    return agent


def sequential_two_stage_predict(volumes, model_main, model_fine, routing_agent=None, safety_threshold_fine=0.5, fallback_threshold_main=0.6):
    """ Kaskada dwustopniowa sterowana przez Agenta RL:
        - Model Fine Tune wstępnie ocenia skan.
        - Agent RL decyduje, czy niepewność wymaga weryfikacji przez model Main, który jest bardziej nagorliwy,
          ale działa z wyższą specyficznością. """
    model_main.eval()
    model_fine.eval()
    if routing_agent is not None:
        routing_agent.eval()
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            probs_fine_tensor = model_fine(volumes)
            probs_main_tensor = model_main(volumes)
            
            probs_fine = probs_fine_tensor[:, 1].cpu().numpy()
            probs_main = probs_main_tensor[:, 1].cpu().numpy()
    
    final_preds = np.zeros_like(probs_fine, dtype=int)
    final_probs = np.zeros_like(probs_fine)
    log_probs_batch = []
    
    for i in range(len(volumes)):
        p_fine = probs_fine[i]
        p_main = probs_main[i]
        
        # Obliczenie entropii Shannona jako miary niepewności predykcji Fine Tune
        p_cls = probs_fine_tensor[i].float()
        entropy = -torch.sum(p_cls * torch.log(p_cls + 1e-10)).item()
        
        if routing_agent is not None:
            # Wektor stanu zgodny z reprezentacją używaną podczas treningu agenta
            state = torch.tensor([[p_fine - safety_threshold_fine, p_main - fallback_threshold_main, entropy]], dtype=torch.float32, device=DEVICE)
            
            action_probs = routing_agent.policy(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs_batch.append(log_prob)
            
            action_item = action.item()
            
            # Akcja 0: Ufamy modelowi Fine Tune
            if action_item == 0:
                final_preds[i] = 1 if p_fine >= safety_threshold_fine else 0
                final_probs[i] = p_fine
            # Akcja 1: Skierowanie do modelu Main
            else:
                final_preds[i] = 1 if p_main >= fallback_threshold_main else 0
                final_probs[i] = p_main
        else:
            log_probs_batch.append(None)
            if p_fine < safety_threshold_fine:
                final_preds[i] = 0
                final_probs[i] = p_fine
            else:
                final_preds[i] = 1 if p_main >= fallback_threshold_main else 0
                final_probs[i] = p_main
                    
    return final_preds, final_probs, log_probs_batch


def evaluate_efficientnet(raw_cq500_path=None, dataset_cq500_path=None):
    """ Przeprowadzam ewaluację kaskadową z udziałem Agenta RL na zbiorze CQ500 w trybie ślepego testu. """
    if raw_cq500_path is None:
        raw_cq500_path = os.path.join(BASE_DIR, 'Data', 'dataset_cq500')
    if dataset_cq500_path is None:
        dataset_cq500_path = os.path.join(BASE_DIR, 'Data', 'dataset_cq500_efficientnet')
    
    def has_images(folder):
        if not os.path.exists(folder): return False
        for _, _, files in os.walk(folder):
            if any(f.lower().endswith(('.png', '.jpg')) for f in files): return True
        return False
        
    unet_path = os.path.join(BASE_DIR, 'output_data', 'Models', 'unet_corpus_callosum.pth')
    unet_model = None
    if os.path.exists(unet_path):
        unet_model = UNet().to(DEVICE)
        unet_model.load_state_dict(torch.load(unet_path, map_location=DEVICE))
        unet_model.eval()
        
    if not has_images(dataset_cq500_path):
        logger.info(f"Brak gotowych wycinków ROI dla CQ500. Rozpoczynam ekstrakcję.")
        if not os.path.exists(raw_cq500_path): return
        preprocess_raw_ct_to_roi(raw_dir=raw_cq500_path, out_dir=dataset_cq500_path, limit_patients_per_class=None, unet_model=unet_model)
        
    imaging_config = MedicalImagingConfig()
    
    # Do ślepego testu wrzucamy cały zbiór testowy, bez odkrawania sztucznego podzbioru do kalibracji w locie
    val_raw_dataset = CTCapsuleDataset(root_dir=dataset_cq500_path, imaging_config=imaging_config, is_train=False)
    val_loader_test = DataLoader(val_raw_dataset, batch_size=4, shuffle=False)
        
    path_main = os.path.join(BASE_DIR, 'output_data', 'Models', 'efficientnet_corpus_callosum_main.pth')
    path_fine = os.path.join(BASE_DIR, 'output_data', 'Models', 'efficientnet_corpus_callosum_finetune.pth')
    path_agent = os.path.join(BASE_DIR, 'output_data', 'Models', 'routing_agent.pth')
    params_path = os.path.join(BASE_DIR, 'output_data', 'Models', 'efficientnet_cascade_params.json')
    
    if not os.path.exists(path_main) or not os.path.exists(path_fine): 
        logger.error("Brak wytrenowanych modeli kaskady w folderze docelowym.")
        return

    model_main = VolumetricEfficientNet(imaging_config=imaging_config, backbone='efficientnet').to(DEVICE)
    model_main.load_state_dict(torch.load(path_main, map_location=DEVICE))
    model_main.eval()

    model_fine = VolumetricEfficientNet(imaging_config=imaging_config, backbone='efficientnet').to(DEVICE)
    model_fine.load_state_dict(torch.load(path_fine, map_location=DEVICE))
    model_fine.eval()
    
    routing_agent = None
    if os.path.exists(path_agent):
        routing_agent = TwoStageRoutingAgent(state_dim=3, action_dim=2).to(DEVICE)
        routing_agent.load_state_dict(torch.load(path_agent, map_location=DEVICE))
        routing_agent.eval()
        logger.info("Wczytano wytrenowanego Agenta RL do zarządzania kaskadą.")

    # Wczytywanie wyliczonych na RSNA progów decyzyjnych Youdena dla obu modeli
    opt_threshold_fine, opt_threshold_main = 0.5, 0.5
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
            opt_threshold_fine = params.get('THRESH_FINE', 0.5)
            opt_threshold_main = params.get('THRESH_MAIN', 0.5)
        logger.info(f"Wczytano zamrożone progi decyzyjne z RSNA: Fine={opt_threshold_fine:.3f}, Main={opt_threshold_main:.3f}")
    else:
        logger.warning("Brak pliku z progami Youdena. Używam domyślnych (0.50).")

    # Faza ewaluacji na zbiorze testowym CQ500
    all_labels, all_probs, all_preds = [], [], [] 
    
    logger.info("Przeprowadzam ostateczną ewaluację na zbiorze testowym przy użyciu Agenta RL.")
    for volumes, labels in val_loader_test:
        volumes = volumes.to(DEVICE)
        labels_np = labels.cpu().numpy()
        
        if unet_model is not None:
            B, S, C, H, W = volumes.shape
            brain_slices = volumes[:, :, 2:3, :, :].contiguous().view(B * S, 1, H, W)
            mask_pred = torch.sigmoid(unet_model(brain_slices)).view(B, S, 1, H, W)
            processed_volumes = volumes * mask_pred
        else:
            processed_volumes = volumes

        preds_batch, probs_batch, _ = sequential_two_stage_predict(
            processed_volumes, 
            model_main, 
            model_fine, 
            routing_agent=routing_agent,
            safety_threshold_fine=opt_threshold_fine, 
            fallback_threshold_main=opt_threshold_main
        )

        all_probs.extend(probs_batch)
        all_preds.extend(preds_batch)
        all_labels.extend(labels_np)
        
    if len(all_labels) > 0:
        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)
        preds = np.array(all_preds)
        
        sens = recall_score(all_labels_np, preds, pos_label=1, zero_division=0) * 100
        spec = recall_score(all_labels_np, preds, pos_label=0, zero_division=0) * 100
        bacc = (sens + spec) / 2.0
        
        if len(np.unique(all_labels_np)) > 1: auc = roc_auc_score(all_labels_np, all_probs_np)
        else: auc = 0.5
            
        logger.info(f"Wyniki Ewaluacji Kaskady z Agentem RL - BAcc: {bacc:.1f}% | AUC: {auc:.3f} | Czułość (Sens): {sens:.1f}% | Specyficzność (Spec): {spec:.1f}%")


def run_pipeline():
    """ Główny orkiestrator kaskady EfficientNet i UNet (Przygotowanie danych -> Trening -> Ewaluacja). """
    raw_ct_path = os.path.join(BASE_DIR, 'Data', 'dataset_ct_main')
    rsna_zip = os.path.join(BASE_DIR, 'Data', 'rsna_dataset.zip')
    rsna_csv = os.path.join(BASE_DIR, 'Data', 'stage_2_train.csv')

    if not _has_images(raw_ct_path):
        if os.path.exists(rsna_zip) and os.path.exists(rsna_csv):
            logger.info("Brak surowych danych CT. Uruchamiam automatyczne przygotowanie z RSNA.")
            prepare_rsna_data(rsna_zip, rsna_csv, raw_ct_path, limit_patients=1000)

    model_unet = os.path.join(BASE_DIR, 'output_data', 'Models', 'unet_corpus_callosum.pth')
    model_efficientnet = os.path.join(BASE_DIR, 'output_data', 'Models', 'efficientnet_corpus_callosum_main.pth')
    model_efficientnet_fine = os.path.join(BASE_DIR, 'output_data', 'Models', 'efficientnet_corpus_callosum_finetune.pth')

    if os.path.exists(model_efficientnet) and os.path.exists(model_efficientnet_fine) and os.path.exists(model_unet):
        logger.info("Znaleziono zapisane wagi modeli. Pomijam trening i przechodzę do ewaluacji.")
    else:
        logger.info("Brakuje plików wag. Uruchamianie głównego treningu kaskady.")
        os.makedirs(os.path.join(BASE_DIR, 'output_data', 'Models'), exist_ok=True)
        train_efficientnet()
        
    logger.info("Ewaluacja modelu na zbiorze testowym CQ500.")
    evaluate_efficientnet()


if __name__ == '__main__':
    run_pipeline()