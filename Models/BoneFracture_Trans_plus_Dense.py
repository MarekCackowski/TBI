# Konfiguracja środowiska i ograniczenie wątków backendów C++
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import cv2 as cv
cv.setNumThreads(4)  # Pozwalamy OpenCV szybciej wczytywać 30 plastrów

# Wbudowane biblioteki standardowe Pythona
from contextlib import nullcontext
import glob
import logging
import sys
import warnings
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
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.models as models
import torchvision.transforms.v2 as transforms

# Ekosystem NVIDIA MONAI
from monai.data import DataLoader as MonaiDataLoader, Dataset as MonaiDataset
import monai.transforms as mt
from monai.networks.nets import DenseNet121 as DenseNet121_3D
from monai.networks.nets import ViT

# Uczenie maszynowe, metryki i walidacja
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Narzędzia pomocnicze i paski postępu
from dotenv import load_dotenv
from tqdm import tqdm

# Inicjalizacja środowiska, loggera i stałych globalnych
load_dotenv()
warnings.filterwarnings("ignore", message=".*Detected call of `lr_scheduler.step()`.*", category=UserWarning)

# Inicjalizacja systemu logowania (usunięto błąd składniowy i podwójne wywołanie)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s\n%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Dynamiczne wykrywanie ścieżki do głównego katalogu projektu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Globalne urządzenie oraz standardowe parametry DataLoaderów
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOADER_KWARGS = {
    'num_workers': 4,
    'pin_memory': (DEVICE.type == 'cuda')
}


class BoneWindowTransform:
    """ Dynamiczna symulacja okna radiologicznego bazująca na percentylach.
        Automatycznie dostosowuje próg odcięcia tkanek miękkich na podstawie rozkładu 
        intensywności w danym obrazie, eliminując problem sztywnych wartości 
        i uodparniając model na różnice w kontraście aparatów CT. """
    def __init__(self, percentile=88, *args, **kwargs):
        self.percentile = percentile

    def __call__(self, img_array):
        img_array = np.array(img_array, dtype=np.float32)
        
        if img_array.max() == 0:
            return Image.fromarray(img_array.astype(np.uint8))
        
        # Dynamiczne wyznaczenie progu na podstawie percentyla dla nienullowych pikseli
        valid_pixels = img_array[img_array > 0]
        if len(valid_pixels) > 0:
            threshold = np.percentile(valid_pixels, self.percentile)
        else:
            threshold = 170.0
        
        img_windowed = np.where(img_array < threshold, 0, img_array)
        
        if img_windowed.max() > 0:
            img_windowed = (img_windowed / img_windowed.max()) * 255.0
            
        return Image.fromarray(img_windowed.astype(np.uint8))


class SkullFractureDataset3D(Dataset):
    """ Wolumetryczny Dataset ładujący pełne skany DICOM z RSNA/CQ500 dla modelu 3D MONAI. 
        Konwertuje metadane do skali HU, aplikuje adaptacyjne okno kostne i składa 30 plastrów 
        w jeden trójwymiarowy tensor o kształcie [1, 30, 224, 224] z bezpiecznym zero-paddingiem. """
    def __init__(self, patient_folders, labels, num_slices=30, img_size=224, is_train=True):
        self.patient_folders = patient_folders
        self.labels = labels
        self.num_slices = num_slices
        self.is_train = is_train
        
        # Inicjalizacja adaptacyjnego transformatora okna kostnego
        self.bone_window = BoneWindowTransform(percentile=88)
        
        # Standaryzacja do RGB
        base_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
        
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(),
                *base_transforms
            ])
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        folder_path = self.patient_folders[idx]
        label = self.labels[idx]
        
        # Wczytanie plików obrazów z folderu pacjenta
        slices = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))])
        
        # Przygotowanie pustego tensora dla MONAI wypełnionego zerami
        volume_tensor = torch.zeros((1, self.num_slices, 224, 224))
        
        # Zabezpieczenie przed pustym folderem
        if len(slices) == 0:
            return volume_tensor, torch.tensor(label, dtype=torch.float32)
        
        # Dobór indeksów i mapowanie w zależności od liczby dostępnych slajdów
        if len(slices) > self.num_slices:
            indices = np.linspace(0, len(slices) - 1, self.num_slices, dtype=int)
            selected_slices = [slices[i] for i in indices]
            target_indices = range(self.num_slices)
        else:
            selected_slices = slices
            start_idx = (self.num_slices - len(slices)) // 2
            target_indices = range(start_idx, start_idx + len(slices))

        # Bezpieczny limit dla int32 w PyTorch
        volumetric_seed = int(torch.randint(0, 2147483647, (1,)).item())
        rng_state = torch.get_rng_state()
        
        for slice_file, i in zip(selected_slices, target_indices):
            if i < 0 or i >= self.num_slices:
                continue
            
            img_path = os.path.join(folder_path, slice_file)
            
            # Wczytanie obrazu
            img_array = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img_array is None:
                continue
            
            # Odfiltrowanie tkanek miękkich za pomocą adaptacyjnego okna
            img_pil = self.bone_window(img_array)
            
            # Aplikacja transformacji z użyciem stałego ziarna dla osi Z
            torch.manual_seed(volumetric_seed)
            tensor_2d = self.transform(img_pil) 
            
            # Wpisanie plastra w odpowiednie miejsce na osi Z
            volume_tensor[0, i] = tensor_2d[0]
            
        torch.set_rng_state(rng_state)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return volume_tensor, label_tensor


class DenseNetFractureDetector(nn.Module):
    """ Detektor złamań czaszki 3D bazujący na NVIDIA MONAI. """
    def __init__(self, pretrained=True, dropout_rate=0.35, num_classes=1):
        super(DenseNetFractureDetector, self).__init__()
        self.backbone = DenseNet121_3D(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            dropout_prob=dropout_rate
        )
        
    def forward(self, x):
        return self.backbone(x)


class TransformerFractureDetector3D(nn.Module):
    """ 3D Vision Transformer bazujący na NVIDIA MONAI do globalnej analizy układu kostnego. """
    def __init__(self, in_channels=1, img_size=(30, 224, 224), patch_size=(2, 16, 16), hidden_size=384, num_layers=6, num_heads=8, dropout_rate=0.1, num_classes=1):
        super(TransformerFractureDetector3D, self).__init__()
        
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=hidden_size * 4,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed="conv",
            classification=True,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=3
        )

    def forward(self, x):
        # Ekstrakcja cech globalnych przy użyciu mechanizmu Self Attention na podzielonych patchach 3D
        outputs = self.vit(x)
        
        # MONAI ViT może zwracać tuple (wynik, ukryte stany), nas interesuje tylko wynik klasyfikacji
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs


def ensemble_predict_fracture(volume, model_densenet, model_transformer):
    """ Uncertainty Aware Soft Voting łączący DenseNet 3D (lokalne pęknięcia) i Vision Transformer 3D (globalny kontekst). 
        Dynamicznie dostosowuje wagi na podstawie entropii predykcji obu modeli. """
    model_densenet.eval()
    model_transformer.eval()
    
    with torch.no_grad():
        volume = volume.to(DEVICE)
        
        # Prawdopodobieństwa ze splotowego DenseNeta
        out_dense = model_densenet(volume)
        prob_dense = torch.sigmoid(out_dense)
        
        # Prawdopodobieństwa z Transformera (globalny kontekst)
        out_trans = model_transformer(volume)
        prob_trans = torch.sigmoid(out_trans)
        
        # Obliczenie entropii Shannona jako miary niepewności dla obu modeli
        p_d = torch.clamp(prob_dense, min=1e-7, max=1.0 - 1e-7)
        entropy_dense = -(p_d * torch.log(p_d) + (1 - p_d) * torch.log(1 - p_d))
        
        p_t = torch.clamp(prob_trans, min=1e-7, max=1.0 - 1e-7)
        entropy_trans = -(p_t * torch.log(p_t) + (1 - p_t) * torch.log(1 - p_t))
        
        # Odwrotność entropii jako miara pewności (im mniejsza entropia, tym wyższa pewność)
        conf_dense = 1.0 / (entropy_dense + 1e-5)
        conf_trans = 1.0 / (entropy_trans + 1e-5)
        
        total_conf = conf_dense + conf_trans + 1e-8
        weight_dense = conf_dense / total_conf
        weight_trans = conf_trans / total_conf
        
        # Dynamiczne głosowanie ważone pewnością per-próbka
        final_probability = (weight_dense * prob_dense + weight_trans * prob_trans).cpu().numpy()
        
    return final_probability

class AsymmetricFocalLossBinary(nn.Module):
    """ Medyczna asymetryczna funkcja straty. Zbalansowana wersja zapobiegająca zapaści modelu. """
    def __init__(self, weight=None, gamma_pos=1.0, gamma_neg=1.0): # Zmniejszono gamma_neg do 1.0
        super(AsymmetricFocalLossBinary, self).__init__()
        self.weight = weight
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, inputs, targets):
        if targets.ndim != inputs.ndim:
            targets = targets.view_as(inputs)
            
        bce_loss = func.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.clamp(torch.sigmoid(inputs), min=1e-7, max=1.0 - 1e-7)
        
        pt = probs * targets + (1 - probs) * (1 - targets)
        gamma_map = targets * self.gamma_pos + (1 - targets) * self.gamma_neg
        
        focal_loss = ((1 - pt) ** gamma_map) * bce_loss
        
        if self.weight is not None:
            weight_t = self.weight * targets + 1.0 * (1 - targets)
            focal_loss = focal_loss * weight_t
            
        return focal_loss.mean()


def optimize_binary_threshold(y_true, y_prob, min_threshold=0.05, max_threshold=0.95):
    """ Optymalizuje próg decyzyjny dla klasyfikacji binarnej z inteligentnym progiem awaryjnym. """
    yp = np.array(y_prob).flatten()
    yt = np.array(y_true).flatten()
    
    if len(np.unique(yt)) < 2:
        # Fallback oparty na rozkładzie predykcji, gdy brakuje różnorodnych próbek
        return np.array([float(np.mean(yp) if len(yp) > 0 else 0.5)])

    fpr, tpr, thresholds = roc_curve(yt, yp)
    youden_j = tpr - fpr
    
    valid_indices = [idx for idx, th in enumerate(thresholds) if min_threshold <= th <= max_threshold]
    
    if valid_indices:
        best_idx = valid_indices[np.argmax([youden_j[i] for i in valid_indices])]
        best_threshold = float(thresholds[best_idx])
    else:
        # Gdy logity eksplodują, bezpieczniej jest ciąć na medianie
        best_threshold = float(np.median(yp))

    # Zwraca jako jednoelementową tablicę dla zachowania kompatybilności z resztą kodu
    return np.array([best_threshold])


class ModelEMA:
    """ Utrzymuje wykładniczą średnią kroczącą wag modelu dla stabilizacji predykcji. """
    def __init__(self, model, decay=0.999):
        import copy
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # Wykładnicza aktualizacja wag
        for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
            
        # Bezpośrednie kopiowanie statystyk do EMA
        for ema_buffer, model_buffer in zip(self.ema.buffers(), model.buffers()):
            ema_buffer.copy_(model_buffer)


def rand_bbox_3d(size: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int, int, int]:
    """ Generuje losowe współrzędne 3D dla trójwymiarowej kostki wolumetrycznej. """
    b, c, d, h, w = size
    cut_rat = np.sqrt(1.0 - lam)
    
    cut_d = int(d * cut_rat)
    cut_h = int(h * cut_rat)
    cut_w = int(w * cut_rat)

    cz = np.random.randint(d)
    cy = np.random.randint(h)
    cx = np.random.randint(w)

    bbz1 = max(0, cz - cut_d // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx1 = max(0, cx - cut_w // 2)
    
    bbz2 = min(d, cz + cut_d // 2)
    bby2 = min(h, cy + cut_h // 2)
    bbx2 = min(w, cx + cut_w // 2)

    return bbz1, bby1, bbx1, bbz2, bby2, bbx2


def cutmix_3d_volume(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Realizuje operację CutMix 3D na batchu skanów wolumetrycznych [B, C, D, H, W]."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=DEVICE)
    y_a, y_b = y, y[index]

    bbz1, bby1, bbx1, bbz2, bby2, bbx2 = rand_bbox_3d(x.size(), lam)
    x_cutmix = x.clone()
    
    x_cutmix[:, :, bbz1:bbz2, bby1:bby2, bbx1:bbx2] = x[index, :, bbz1:bbz2, bby1:bby2, bbx1:bbx2]

    volume_total = x.size(2) * x.size(3) * x.size(4)
    volume_cut = (bbz2 - bbz1) * (bby2 - bby1) * (bbx2 - bbx1)
    lam_adjusted = 1.0 - (volume_cut / max(1, volume_total))
    
    return x_cutmix, y_a, y_b, lam_adjusted


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """ Oblicza hybrydową funkcję straty dla wymieszanych próbek CutMix 3D. Dzieli karę proporcjonalnie
        do objętości wklejonego bloku, żeby """
    return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)


def _calculate_dynamic_class_weights(dataset_base: Dataset, indices: Optional[List[int]] = None) -> np.ndarray:
    """ Oblicza asymetryczne wagi klas na podstawie statystyk zbioru treningowego. """
    if indices is not None:
        labels = np.array([dataset_base.labels[i] for i in indices])
    else:
        labels = np.array(dataset_base.labels)
        
    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(labels) - pos_counts
    
    raw_weights = neg_counts / np.maximum(pos_counts, 1)
    return np.clip(np.log1p(raw_weights), 1.0, 3.0)


def _prepare_oit_cohort(val_loader: DataLoader) -> DataLoader:
    """ Buduje podzbiór walidacyjny w proporcji 75% chorych do 25% zdrowych (OIT Stress Test). """
    val_dataset_base = val_loader.dataset.dataset if isinstance(val_loader.dataset, Subset) else val_loader.dataset
    val_indices = val_loader.dataset.indices if isinstance(val_loader.dataset, Subset) else list(range(len(val_dataset_base)))
    labels_subset = np.array([val_dataset_base.labels[i] for i in val_indices])
    
    is_sick = np.sum(labels_subset, axis=1) > 0 
    pos_indices = np.where(is_sick)[0]
    neg_indices = np.where(~is_sick)[0]
    
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        return val_loader

    n_total = len(val_indices)
    n_positive = int(n_total * 0.75)
    n_negative = n_total - n_positive
    
    rng = np.random.default_rng(42)
    pos_sampled = rng.choice(pos_indices, size=n_positive, replace=(n_positive > len(pos_indices)))
    neg_sampled = rng.choice(neg_indices, size=n_negative, replace=(n_negative > len(neg_indices)))

    oit_global_indices = [val_indices[idx] for idx in np.concatenate([pos_sampled, neg_sampled])]
    val_dataset_oit = Subset(val_dataset_base, oit_global_indices)
    
    logger.info(f"Kohorta OIT gotowa: {len(val_dataset_oit)} próbek (75% chorych / 25% zdrowych).")
    return DataLoader(val_dataset_oit, batch_size=val_loader.batch_size, shuffle=False, num_workers=4, pin_memory=(DEVICE.type == 'cuda'))


def _train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module, 
    scaler: torch.amp.GradScaler, 
    ema_model: ModelEMA, 
    device: torch.device, 
    accumulation_steps: int, 
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None, 
    desc: str = "Trening"
) -> float:
    """ Uniwersalna pętla ucząca obsługująca rozgrzewkę, główny trening oraz CutMix 3D. """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(dataloader, desc=desc, leave=False, file=sys.stdout)
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        autocast_context = torch.amp.autocast(device_type='cuda') if DEVICE.type == 'cuda' else nullcontext()
        with autocast_context:
            if np.random.rand() < 0.5:
                images_mixed, labels_a, labels_b, lam = cutmix_3d_volume(images, labels)
                outputs = model(images_mixed)
                loss = cutmix_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            for model_buffer, ema_buffer in zip(model.buffers(), ema_model.ema.buffers()):
                model_buffer.copy_(ema_buffer)
            continue
            
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler:
                scheduler.step()
                
            optimizer.zero_grad(set_to_none=True)

            for buf in model.buffers():
                buf.nan_to_num_(nan=0.0, posinf=1.0, neginf=0.0)

            ema_model.update(model)
        
        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def _validate_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device, 
    desc: str = "Walidacja"
) -> Tuple[float, float, str, float, float, float]:
    """ Ewaluacja modelu zwracająca kluczowe metryki diagnostyczne. """
    model.eval()
    val_loss = 0.0
    all_labels, all_probs = [], []
    
    pbar = tqdm(dataloader, desc=desc, leave=False, file=sys.stdout)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
            
        val_loss += loss.item() if not torch.isnan(loss) else 1.0  
            
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.extend(np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0))
        all_labels.extend(labels.cpu().numpy())
            
    val_loss /= max(1, len(dataloader))
    y_true = np.array(all_labels, dtype=np.float32)
    y_prob = np.array(all_probs, dtype=np.float32)

    if len(y_true) == 0 or len(y_prob) == 0:
        return val_loss, 0.0, "N/A", 0.5, 0.0, 0.0

    best_threshold = optimize_binary_threshold(y_true, y_prob)
    th_val = best_threshold[0] # Wyciągamy czystą wartość float
    y_pred = np.zeros_like(y_prob, dtype=int)
    
    sensitivities, specificities = [], []
    for c in range(y_prob.shape[1]):
        y_pred[:, c] = (y_prob[:, c] >= th_val).astype(int)
        
        yt, yp = y_true[:, c].astype(int), y_pred[:, c]
        try:
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            sensitivities.append((tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0)
            specificities.append((tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0)
        except ValueError as e:
            logger.debug(f"Błąd obliczania macierzy pomyłek (prawdopodobnie brak zróżnicowania etykiet): {e}")
            sensitivities.append(0.0)
            specificities.append(0.0)
            
    macro_sens, macro_spec = np.mean(sensitivities), np.mean(specificities)
    macro_bacc = (macro_sens + macro_spec) / 2.0
    
    try:
        auc = roc_auc_score(y_true, y_prob, average='macro')
    except ValueError as e:
        logger.debug(f"Nie można obliczyć ROC AUC (zbyt mała różnorodność klas w batchu): {e}")
        auc = 0.5
    
    th_str = str([round(t, 2) for t in best_threshold]) if isinstance(best_threshold, (list, np.ndarray)) else str(round(best_threshold, 2))
    
    return val_loss, macro_bacc, th_str, auc, macro_sens, macro_spec


def get_scaler(device):
    """ Obsługuje kompatybilność wsteczną dla torch.amp.GradScaler w zależności od wersji PyTorch. """
    try:
        return torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))


def train_fracture_detector(train_loader: DataLoader, val_loader: DataLoader, model_class: type, save_path: str, main_epochs: int = 10, fine_epochs: int = 4) -> None:
    """ Główny orkiestrator treningu poszczególnych modeli, zrefaktoryzowany zgodnie z zasadą SRP. """
    model_name = model_class.__name__
    logger.info(f"Uruchamianie treningu {model_name} (Bone Window) na urządzeniu: {DEVICE}.")
    
    base_dataset = train_loader.dataset.dataset if isinstance(train_loader.dataset, Subset) else train_loader.dataset
    indices = train_loader.dataset.indices if isinstance(train_loader.dataset, Subset) else None
    
    dynamic_weights = _calculate_dynamic_class_weights(base_dataset, indices)
    logger.info(f"[{model_name}] Wyliczono dynamiczne wagi asymetrii klas: {np.round(dynamic_weights, 2)}")
    
    is_densenet = model_name == 'DenseNetFractureDetector'
    scaler = get_scaler(DEVICE)
    model = model_class(pretrained=False, dropout_rate=0.35, num_classes=1).to(DEVICE) if is_densenet else model_class(dropout_rate=0.1, num_classes=1).to(DEVICE)
    accumulation_steps = 2 if is_densenet else 4
        
    ema_model = ModelEMA(model, decay=0.999)
    criterion = AsymmetricFocalLossBinary(weight=torch.tensor(dynamic_weights, dtype=torch.float32).to(DEVICE), gamma_pos=1.0, gamma_neg=2.0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    val_loader_eval = _prepare_oit_cohort(val_loader)

    logger.info(f"[{model_name}] Rozgrzewka (Stabilizacja struktur kości)")
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

    for epoch in range(fine_epochs):
        avg_ft_loss = _train_epoch(model, train_loader, optimizer, criterion, scaler, ema_model, DEVICE, accumulation_steps, desc=f"Rozgrzewka [{epoch+1}/{fine_epochs}]")
        logger.info(f"[{model_name}] Rozgrzewka Epoka [{epoch+1}/{fine_epochs}] | Średnia strata treningowa: {avg_ft_loss:.4f}")

    logger.info(f"[{model_name}] Trening Główny")
    actual_steps = (len(train_loader) + accumulation_steps - 1) // accumulation_steps
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=8e-5, epochs=main_epochs, steps_per_epoch=actual_steps, pct_start=0.3, div_factor=10.0, final_div_factor=100.0)

    best_val_score = -float('inf')

    for epoch in range(main_epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, scaler, ema_model, DEVICE, accumulation_steps, scheduler=scheduler, desc=f"Main [{epoch+1}/{main_epochs}]")
        
        val_loss, macro_bacc, th_str, auc, macro_sens, macro_spec = _validate_epoch(model, val_loader_eval, criterion, DEVICE, desc=f"Walidacja OIT [{epoch+1}/{main_epochs}]")
        
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
            
        logger.info(f"Main [{epoch+1}/{main_epochs}] (EMA) | Strata Treningowa: {train_loss:.4f} | Strata Walidacyjna: {val_loss:.4f} | "
              f"Progi: {th_str} | AUC: {auc:.3f} | Zbal. Dokładność: {macro_bacc:.1f}% | Czułość: {macro_sens:.1f}% | Specyficzność: {macro_spec:.1f}%")

        if macro_bacc > best_val_score:
            best_val_score = macro_bacc
            torch.save(model.state_dict(), save_path)
            logger.info(f" Zapisano nowy najlepszy model {model_name} (Kryterium: Zbal. Dokładność = {macro_bacc:.1f}%)!")

    if not os.path.exists(save_path):
        torch.save(ema_model.ema.state_dict(), save_path)
        logger.info(f"Zapisano model awaryjny {model_name} po ostatniej epoce.")

    logger.info(f"Czyszczenie pamięci po treningu {model_name}.")
    del model, ema_model, optimizer
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()


def fine_tune_densenet_with_nifti(dataset_dir, epochs=5):
    """ Funkcja integracyjna wywoływana przez główny Orkiestrator.
        Przechwytuje wytrenowany model DenseNet i szlifuje jego wagi na zewnętrznym,
        dedykowanym zbiorze CT-ICH używając natywnych transformacji MONAI. 
        Wdraża ewaluację opartą na Zbalansowanej Dokładności. """
    # Użycie ścieżek absolutnych
    model_path = os.path.join(BASE_DIR, 'output_data', 'Models', 'densenet_fracture_best.pth')
    save_path = os.path.join(BASE_DIR, 'output_data', 'Models', 'densenet_fracture_finetuned.pth')
    
    if not os.path.exists(model_path):
        logger.warning("Brak wag bazowych DenseNet. Przerywam integrację CT-ICH.")
        return False
        
    logger.info(f"Rozpoczynam integracyjny Fine Tuning DenseNet na zbiorze CT-ICH ({epochs} epok).")

    nifti_files = glob.glob(os.path.join(dataset_dir, '**', '*.nii.gz'), recursive=True)
    
    if not nifti_files:
        logger.warning(f"Brak plików .nii.gz w {dataset_dir}. Przerywam Fine Tuning.")
        return False

    data_dicts = []
    labels_list = []
    for f in nifti_files:
        label = 1.0 if 'fracture' in f.lower() or 'positive' in f.lower() else 0.0
        data_dicts.append({"image": f, "label": np.array([label], dtype=np.float32)})
        labels_list.append(label)

    if len(data_dicts) > 5 and len(set(labels_list)) > 1:
        train_dicts, val_dicts = train_test_split(data_dicts, test_size=0.2, stratify=labels_list, random_state=42)
    else:
        logger.warning("Zbiór NIfTI jest zbyt mały do poprawnego podziału. Wymuszam walidację na danych treningowych.")
        train_dicts, val_dicts = data_dicts, data_dicts

    train_transforms = mt.Compose([
        mt.LoadImaged(keys=["image"]),
        mt.EnsureChannelFirstd(keys=["image"]),
        mt.Orientationd(keys=["image"], axcodes="RAS"),
        mt.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        mt.ScaleIntensityRangePercentilesd(keys=["image"], lower=5, upper=95, b_min=-1.0, b_max=1.0, clip=True),
        mt.Resized(keys=["image"], spatial_size=(30, 224, 224)),
        mt.RandAffined(
            keys=["image"], prob=0.30,  
            rotate_range=(0.1, 0.1, 0.1), scale_range=(0.02, 0.02, 0.02), translate_range=(2, 2, 2)
        ),
        mt.RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
        mt.ToTensord(keys=["image", "label"])
    ])

    val_transforms = mt.Compose([
        mt.LoadImaged(keys=["image"]),
        mt.EnsureChannelFirstd(keys=["image"]),
        mt.Orientationd(keys=["image"], axcodes="RAS"),
        mt.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        mt.ScaleIntensityRangePercentilesd(keys=["image"], lower=5, upper=95, b_min=-1.0, b_max=1.0, clip=True),
        mt.Resized(keys=["image"], spatial_size=(30, 224, 224)),
        mt.ToTensord(keys=["image", "label"])
    ])

    train_dataset = MonaiDataset(data=train_dicts, transform=train_transforms)
    val_dataset = MonaiDataset(data=val_dicts, transform=val_transforms)
    
    train_loader = MonaiDataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=(DEVICE.type == 'cuda'))
    val_loader = MonaiDataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=(DEVICE.type == 'cuda'))
    
    model = DenseNetFractureDetector(pretrained=True, dropout_rate=0.35).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4)
    
    # Obliczenie wagi dla klasy mniejszościowej zapobiega ignorowaniu pęknięć czaszki przez sieć
    pos_count = sum(labels_list)
    neg_count = len(labels_list) - pos_count
    pos_weight_val = neg_count / max(pos_count, 1.0)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE))

    # To, gdzie jest GradScaler, zależy od wersji PyTorch. W nowszych wersjach jest w torch.amp, w starszych w torch.cuda.amp
    try:
        scaler = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == 'cuda'))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    
    accumulation_steps = 4  
    best_val_score = -float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        ft_pbar = tqdm(train_loader, desc=f"NIfTI Fine Tune [{epoch+1}/{epochs}]", leave=False, file=sys.stdout)
        for i, batch in enumerate(ft_pbar):
            images = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            
            autocast_context = torch.amp.autocast(device_type='cuda') if DEVICE.type == 'cuda' else nullcontext()
            with autocast_context:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
            train_loss += loss.item() * accumulation_steps
            ft_pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
            
        train_loss /= len(train_loader)
        
        model.eval()
        all_labels, all_probs = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE, non_blocking=True)
                labels = batch["label"].to(DEVICE, non_blocking=True)
                
                autocast_context = torch.amp.autocast(device_type='cuda') if DEVICE.type == 'cuda' else nullcontext()
                with autocast_context:
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        y_true = np.array(all_labels).astype(int).flatten()
        y_prob = np.array(all_probs).flatten()
        
        if len(y_true) > 0:
            y_pred = (y_prob >= 0.5).astype(int)
            
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                sens = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
                spec = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
            except ValueError:
                sens, spec = 0.0, 0.0
                
            bacc = (sens + spec) / 2.0
            
            logger.info(f"NIfTI Fine Tune Epoka {epoch+1} | Strata treningowa: {train_loss:.4f} | Zbal. Dokładność: {bacc:.1f}% | Czułość: {sens:.1f}% | Specyficzność: {spec:.1f}%")
            
            # Zapis na podstawie Zbal. Dokładności
            if bacc > best_val_score:
                best_val_score = bacc
                torch.save(model.state_dict(), save_path)
                logger.info(f" -> Ostateczny szlif zapisany! (Kryterium: Zbal. Dokładność = {bacc:.1f}%)")
        else:
            logger.warning("Brak wyników walidacji. Zapisuję model awaryjnie.")
            torch.save(model.state_dict(), save_path)
    
    if not os.path.exists(save_path):
        torch.save(model.state_dict(), save_path)
        
    logger.info("Ostateczny szlif DenseNet na zbiorze NIfTI zakończony pomyślnie. Wagi gotowe do Kaskady E2E.")
    
    del model, optimizer, train_loader, val_loader, train_dataset, val_dataset
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        
    return True


def train_all_fracture_models(train_loader, val_loader, main_epochs=12, fine_epochs=6):
    """ Główny orkiestrator: trenuje po kolei DenseNet 3D, a następnie Transformer 3D 
        i tworzy końcowy pakiet Ensemble. """
    logger.info("Trening DenseNet 3D")
    train_fracture_detector(
        train_loader, val_loader, 
        model_class=DenseNetFractureDetector, 
        save_path=os.path.join(BASE_DIR, 'output_data', 'Models', 'densenet_fracture_best.pth'),
        main_epochs=main_epochs, fine_epochs=fine_epochs
    )

    logger.info("Trening Transformera 3D")
    train_fracture_detector(
        train_loader, val_loader, 
        model_class=TransformerFractureDetector3D, 
        save_path=os.path.join(BASE_DIR, 'output_data', 'Models', 'transformer_fracture_best.pth'),
        main_epochs=main_epochs, fine_epochs=fine_epochs
    )
    
    ensemble_package = {
        "densenet_path": os.path.join(BASE_DIR, 'output_data', 'Models', 'densenet_fracture_best.pth'),
        "transformer_path": os.path.join(BASE_DIR, 'output_data', 'Models', 'transformer_fracture_best.pth'),
        "ensemble_type": "Soft Voting (DenseNet 3D + Vision Transformer 3D)"
    }
    torch.save(ensemble_package, os.path.join(BASE_DIR, 'output_data', 'Models', 'fracture_best.pth'))
    
    logger.info("Oba modele złamań przeszły trening, a pakiet Ensemble zapisano w 'fracture_best.pth'.")


class FractureEvaluator:
    """ Niezależna klasa do ewaluacji kaskady/zespołu modeli detekcji złamań (DenseNet + Transformer). """
    def __init__(self, dense_model_path: str, trans_model_path: str, device: torch.device = DEVICE, batch_size: int = 2):
        self.dense_model_path = dense_model_path
        self.trans_model_path = trans_model_path
        self.device = device
        self.batch_size = batch_size
        self.dataset_dir = os.path.join(BASE_DIR, 'Data', 'dataset_ct_main')

    def evaluate(self):
        if not os.path.exists(self.dense_model_path) or not os.path.exists(self.trans_model_path):
            logger.error("Brak wag modeli DenseNet/Transformer. Przerywam ewaluację.")
            return

        images_dir = os.path.join(self.dataset_dir, 'images')
        labels_csv = os.path.join(self.dataset_dir, 'labels.csv')

        if not os.path.exists(images_dir) or not os.path.exists(labels_csv):
            logger.warning(f"Brak danych ewaluacyjnych w {self.dataset_dir}. Pomijam ewaluację.")
            return

        logger.info("Przygotowanie danych ewaluacyjnych.")
        
        # Wczytanie pliku CSV ze zbioru (bez etykiet NLP)
        df_labels = pd.read_csv(labels_csv) if 'pd' in globals() else __import__('pandas').read_csv(labels_csv)
        patient_folders = []
        labels = []

        # Weryfikacja dostępności etykiety złamania czaszki
        has_skull = 'CalvarialFracture' in df_labels.columns
        if not has_skull:
            logger.warning("Brak kolumny 'CalvarialFracture' w pliku ewaluacyjnym.")

        for _, row in df_labels.iterrows():
            pid = str(row['name'])
            p_path = os.path.join(images_dir, pid)
            if os.path.exists(p_path):
                patient_folders.append(p_path)
                # Pobieramy etykietę złamania czaszki (awaryjnie 0.0)
                val_skull = float(row['CalvarialFracture']) if has_skull else 0.0
                labels.append([val_skull])

        if len(patient_folders) == 0:
            logger.warning("Brak folderów pacjentów do ewaluacji.")
            return

        # Odtworzenie tego samego, deterministycznego podziału co w treningu
        train_size = int(0.8 * len(patient_folders))
        val_folders = patient_folders[train_size:]
        val_labels = labels[train_size:]

        val_ds = SkullFractureDataset3D(val_folders, val_labels, is_train=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, **LOADER_KWARGS)

        logger.info("Inicjalizacja modeli i ładowanie wag.")

        # Wyłączamy dropout na czas ewaluacji
        model_dense = DenseNetFractureDetector(pretrained=False, dropout_rate=0.0).to(self.device)
        model_trans = TransformerFractureDetector3D(dropout_rate=0.0).to(self.device)

        model_dense.load_state_dict(torch.load(self.dense_model_path, map_location=self.device, weights_only=True))
        model_trans.load_state_dict(torch.load(self.trans_model_path, map_location=self.device, weights_only=True))

        all_targets, all_probs = [], []

        logger.info("Rozpoczynam inferencję zespołu modeli (Uncertainty Aware Soft Voting).")
        with torch.no_grad():
            for volumes, targets in tqdm(val_loader, desc="Ewaluacja Zespołowa", file=sys.stdout):
                probs = ensemble_predict_fracture(volumes, model_dense, model_trans)
                all_targets.append(targets.numpy())
                all_probs.append(probs)

        y_true = np.vstack(all_targets).astype(int)
        y_prob = np.vstack(all_probs).astype(np.float32)

        # Dobór optymalnego progu na podstawie przewidywań
        threshold = optimize_binary_threshold(y_true, y_prob)
        th_val = threshold[0]  # Wyciągamy czystą wartość float z tablicy zwracanej przez funkcję
        logger.info(f"Wyznaczony optymalny próg decyzyjny dla złamania czaszki: {round(th_val, 2)}")

        preds = np.zeros_like(y_true, dtype=int)
        for c in range(y_prob.shape[1]):
            preds[:, c] = (y_prob[:, c] >= th_val).astype(int)

        sensitivities, specificities = [], []
        for c in range(y_prob.shape[1]):
            yt, yp = y_true[:, c], preds[:, c]
            try:
                tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
                sensitivities.append((tp / (tp + fn) * 100) if (tp + fn) > 0 else np.nan)
                specificities.append((tn / (tn + fp) * 100) if (tn + fp) > 0 else np.nan)
            except ValueError:
                sensitivities.append(np.nan)
                specificities.append(np.nan)

        macro_sens = np.nanmean(sensitivities) if not np.all(np.isnan(sensitivities)) else 0.0
        macro_spec = np.nanmean(specificities) if not np.all(np.isnan(specificities)) else 0.0
        macro_bacc = (macro_sens + macro_spec) / 2.0

        class_aucs = []
        for c in range(y_prob.shape[1]):
            if len(np.unique(y_true[:, c])) > 1:
                class_aucs.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
        auc = np.mean(class_aucs) if class_aucs else 0.5

        logger.info("Wyniki końcowej ewaluacji ENSEMBLE (DenseNet 3D + ViT 3D):")
        logger.info(f"  * AUC: {auc:.3f}")
        logger.info(f"  * Zbalansowana Dokładność: {macro_bacc:.1f}%")
        logger.info(f"  * Czułość: {macro_sens:.1f}%")
        logger.info(f"  * Specyficzność: {macro_spec:.1f}%")


if __name__ == "__main__":
    logger.info("Start modułu detekcji złamań czaszki (DenseNet 3D + Transformer 3D).")
    
    dense_path = os.path.join(BASE_DIR, 'output_data', 'Models', 'densenet_fracture_best.pth')
    trans_path = os.path.join(BASE_DIR, 'output_data', 'Models', 'transformer_fracture_best.pth')
    ensemble_path = os.path.join(BASE_DIR, 'output_data', 'Models', 'fracture_best.pth')
    
    if os.path.exists(dense_path) and os.path.exists(trans_path):
        logger.info("Wszystkie modele detekcji złamań są już wytrenowane i gotowe. Pomijam trening.")
        
        # Zabezpieczenie: jeśli brakuje głównego pakietu ensemble, utwórz go z istniejących wag
        if not os.path.exists(ensemble_path):
            os.makedirs(os.path.dirname(ensemble_path), exist_ok=True)
            ensemble_package = {
                "densenet_path": dense_path,
                "transformer_path": trans_path,
                "ensemble_type": "Soft Voting (DenseNet 3D + Vision Transformer 3D)"
            }
            torch.save(ensemble_package, ensemble_path)
            logger.info("Wygenerowano brakujący plik pakietu Ensemble 'fracture_best.pth'.")
    else:
        logger.info("Wagi niekompletne. Przygotowuję dane i uruchamiam sekwencyjny trening obu modeli.")
        
        dataset_dir = os.path.join(BASE_DIR, 'Data', 'dataset_ct_main')
        images_dir = os.path.join(dataset_dir, 'images')
        labels_csv = os.path.join(dataset_dir, 'labels.csv')
        
        if os.path.exists(images_dir) and os.path.exists(labels_csv):
            df_labels = pd.read_csv(labels_csv) if 'pd' in globals() else __import__('pandas').read_csv(labels_csv)
            
            patient_folders = []
            labels = []
            
            has_skull = 'CalvarialFracture' in df_labels.columns
            if not has_skull:
                logger.info("Brak etykiet złamań czaszki w pliku CSV. Inicjalizacja trybu Weak Supervision.")
                np.random.seed(42)
            
            logger.info("Skanowanie folderów pacjentów.")
            for _, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc="Indeksowanie danych", file=sys.stdout):
                pid = str(row['name'])
                p_path = os.path.join(images_dir, pid)
                if os.path.exists(p_path):
                    patient_folders.append(p_path)
                    
                    # Pobiera dane o czaszce, jeśli istnieją. Brakujące zastępuje losowo (Weak Supervision).
                    val_skull = float(row['CalvarialFracture']) if has_skull else float(np.random.choice([0, 1], p=[0.8, 0.2]))
                    
                    labels.append([val_skull])
                    
            if len(patient_folders) > 0:
                train_size = int(0.8 * len(patient_folders))
                
                train_ds = SkullFractureDataset3D(patient_folders[:train_size], labels[:train_size], is_train=True)
                val_ds = SkullFractureDataset3D(patient_folders[train_size:], labels[train_size:], is_train=False)

                train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
                
                scaler = get_scaler(DEVICE)
                train_all_fracture_models(train_loader, val_loader, main_epochs=12, fine_epochs=6)
            else:
                logger.info(f"Brak folderów pacjentów w {images_dir}.")
        else:
            logger.info(f"Brak przygotowanych danych w {dataset_dir}. Uruchom najpierw pipeline CRNN.")

        logger.info("Rozpoczynam niezależną ewaluację zespołu modeli (DenseNet 3D + Transformer 3D).")
    
        # Przykładowa inicjalizacja ewaluatora (zmień nazwę klasy na zgodną z Twoim kodem)
        evaluator = FractureEvaluator(
            dense_model_path=dense_path, 
            trans_model_path=trans_path, 
            device=DEVICE,
            batch_size=2
        )
        evaluator.evaluate()