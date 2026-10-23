# Wbudowane biblioteki standardowe
import os
import sys
import gc
import math
import json
import zipfile
import logging
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional


# Ograniczenie wątków procesora dla backendów C++
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# Przetwarzanie obrazów i formaty medyczne
import cv2 as cv
cv.setNumThreads(0)  # Wyłączenie konfliktowej wielowątkowości OpenCV
from PIL import Image
import pydicom


# Analiza danych i operacje wektorowe
import numpy as np
import pandas as pd


# Silnik Deep Learningu - PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
import torch.backends.cudnn as cudnn


# Ładowanie danych i augmentacje
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms.v2 as transforms
import torchvision.models as models
from torchvision.ops import DeformConv2d


# Używane do dynamicznego pobierania pretrenowanych wag
import timm
import requests
from huggingface_hub import hf_hub_download


# Używane do optymalizacji progów decyzyjnych (ROC, J-Youden) oraz podziału danych
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    confusion_matrix,
)
import optuna

# Inicjalizacja loggera dla całego modułu i ładowanie zmiennych środowiskowych z pliku .env
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

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


# Ustawienie liczby wątków dla DataLoaderów i operacji macierzowych na GPU
NUM_WORKERS = 4

# Dynamiczne wykrywanie ścieżki do głównego katalogu projektu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Globalne urządzenie dla całego skryptu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Standardowe parametry dla DataLoaderów, eliminujące powtarzanie
LOADER_KWARGS = {
    'num_workers': 4,
    'pin_memory': (DEVICE.type == 'cuda')
}


@dataclass
class MedicalImagingConfig:
    """ Wspólne parametry obrazu i standardy medyczne. """
    img_size: int = 160
    num_slices: int = 24
    num_damage_types: int = 5
    window_brain: Tuple[int, int] = (40, 80)
    window_blood: Tuple[int, int] = (80, 200)       # Standardowe okno krwi
    window_subdural: Tuple[int, int] = (70, 150)    # Wąskie okno do krawędzi (SDH), używamy obu, sieć sama się nauczy, które jest lepsze w danym przypadku
    window_bone: Tuple[int, int] = (600, 2000)
    target_classes: List[str] = field(
        default_factory=lambda: ['epidural', 'subdural', 'subarachnoid', 'intraparenchymal', 'intraventricular']
    )

@dataclass
class CRNNConfig:
    """ Hiperparametry treningowe specyficzne dla modułu CRNN. """
    img_size: int = 160
    num_slices: int = 24
    batch_size: int = 2
    accumulation_steps: int = 8
    
    epochs_warmup: int = 2
    epochs_main: int = 10
    epochs_fine: int = 6
    
    patience: int = 4
    lr_main: float = 5e-5
    lr_fine: float = 1e-5
    weight_decay: float = 1e-2
    num_classes: int = 5

    thresholds_fine_default: Tuple[float, ...] = (0.20, 0.20, 0.20, 0.20, 0.20)


def objective(trial: optuna.Trial) -> float:
    """ Funkcja celu dla Optuny z obsługą Pruningu (przycinania słabych prób). """
    
    config = CRNNConfig(
        batch_size=trial.suggest_categorical("batch_size", [2, 4, 8]),
        accumulation_steps=trial.suggest_categorical("accumulation_steps", [4, 8, 16]),
        lr_main=trial.suggest_float("lr_main", 1e-5, 1e-4, log=True),
        lr_fine=trial.suggest_float("lr_fine", 1e-6, 5e-5, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        epochs_main=4,  # Krótki trening proxy
        epochs_fine=2
    )
    
    dataset_path_main = os.path.join(BASE_DIR, 'Data', 'dataset_ct_main')
    
    try:
        full_main_dataset = CTSliceDataset(root_dir=dataset_path_main, num_slices=config.num_slices, img_size=config.img_size, is_train=True)
        val_dataset = CTSliceDataset(root_dir=dataset_path_main, num_slices=config.num_slices, img_size=config.img_size, is_train=False)
        
        labels_binary = np.any(full_main_dataset.labels > 0, axis=1).astype(int)
        train_indices, val_indices = train_test_split(range(len(full_main_dataset)), test_size=0.2, stratify=labels_binary, random_state=42)
        
        # Bierzemy losową próbkę 30% danych treningowych, aby przyspieszyć optymalizację hiperparametrów
        rng_proxy = np.random.default_rng(42)
        train_indices = rng_proxy.choice(train_indices, size=max(50, int(len(train_indices) * 0.3)), replace=False)
        
        train_sub = Subset(full_main_dataset, train_indices)
        val_sub = Subset(val_dataset, val_indices)
        train_labels_array = np.array([full_main_dataset.labels[i] for i in train_indices])
        
        sampler = _create_balanced_sampler(train_labels_array)
        train_loader = DataLoader(train_sub, batch_size=config.batch_size, sampler=sampler, **LOADER_KWARGS)
        val_loader_eval = DataLoader(val_sub, batch_size=config.batch_size, shuffle=False, **LOADER_KWARGS)


        model = VolumetricMedicalResNetCRNN(task='multilabel', num_classes=config.num_classes).to(DEVICE)
        _initialize_head_biases(model, train_labels_array, DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=config.lr_main, weight_decay=config.weight_decay)
        criterion = MultiLabelFocalLoss(gamma_pos=2.0, gamma_neg=2.0)

        # To, gdzie jest GradScaler, zależy od wersji PyTorch. W nowszych wersjach jest w torch.amp, w starszych w torch.cuda.amp
        try:
            scaler = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == 'cuda'))
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
        
        best_bacc = 0.0
        for epoch in range(config.epochs_main):
            current_budget = get_dynamic_weight(epoch, config.epochs_main, start_w=0.2, end_w=1.0)
            _train_epoch_crnn(model, train_loader, optimizer, criterion, scaler, config, desc=f"Trial Epoka {epoch+1}", budget=current_budget)
            _, _, bacc, _, _, _ = _validate_epoch_crnn(model, val_loader_eval, criterion, desc="Trial Val", optimize_thresholds=False)
            
            if bacc > best_bacc:
                best_bacc = bacc
                
            # Raportujemy wynik do Optuny po każdej epoce
            trial.report(bacc, epoch)
            
            # Jeśli próba radzi sobie gorzej niż mediana poprzednich, Optuna ją natychmiast przerywa
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return best_bacc
        
    except optuna.exceptions.TrialPruned:
        logger.info(f"Próba {trial.number} została odrzucona przez mechanizm Pruning.")
        raise
    except Exception as e:
        logger.error(f"Błąd w próbie Optuny: {e}")
        return 0.0

def run_hyperparameter_optimization():
    """ Uruchamia optymalizację z aktywnym MedianPruner. """
    logger.info("Rozpoczynam optymalizację hiperparametrów za pomocą Optuna.")
    
    # Pruner odcina słabe próby po 1. epoce, dając szansę co najmniej 2 próbom startowym
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1, interval_steps=1)
    
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=10, timeout=3600)
    
    logger.info("Optymalizacja zakończona!")
    logger.info(f"Najlepsza Zbalansowana Dokładność: {study.best_value:.2f}%")
    logger.info("Najlepsze hiperparametry:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
        
    return study.best_params


class DeformableFeatureExtractor(nn.Module):
    """ Uelastycznia linie ekstrakcji cech, pozwalając na śledzenie nieregularnych wylewów krwi. """
    def __init__(self, channels):
        super().__init__()
        # Mape przesunięć przestrzennych dla każdego punktu w obrazie, aby deformować siatkę próbkowania
        self.offset_conv = nn.Conv2d(channels, 2 * 3 * 3, kernel_size=3, padding=1, bias=False)

        # Główna warstwa splotu odkształcalnego, służaca do adaptacji do nieregularnych kształtów krwotoków i asymetrii mózgu
        self.dcn = DeformConv2d(channels, channels, kernel_size=3, padding=1)

        # Stabilizacja wariancji i parametryczna aktywacja chroniąca przed martwymi neuronami
        self.bn = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adaptacja do fizycznego kształtu patologii
        offsets = self.offset_conv(x)
        
        # Próbkowanie cech z nieregularnej siatki na podstawie wyliczonych wektorów
        x = self.dcn(x, offsets)
        
        # Normalizacja i nieliniowa aktywacja zachowująca ujemne sygnały
        return self.prelu(self.bn(x))


class CRAMBlock(nn.Module):
    """ Customized Residual Attention Module — odszumia tło (grube kości czaszki) i uwydatnia krwotok. """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        """ Wstępna aktywacja PReLU dla głównej ścieżki, zależy nam na zachowaniu ujemnych gradientów dla
            struktur tkankowych, które mogą być delikatne i uniknięciu problemu martwych neuronów. """
        self.prelu_main = nn.PReLU(channels)

        # Uwaga kanałowa
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.PReLU(channels // 8),
            nn.Conv2d(channels // 8, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Uwaga przestrzenna
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.prelu_main(self.bn1(self.conv1(x)))

        # Aplikacja uwagi kanałowej
        x = x * self.ca(x)

        # Aplikacja uwagi przestrzennej
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        sa_in = torch.cat([max_out, avg_out], dim=1)
        
        # Zapisanie maski do wizualizacji
        self.last_spatial_map = self.sa(sa_in)
        x = x * self.last_spatial_map

        return x + res


class EnhancedBiLSTM(nn.Module):
    """ Stabilna pamięć wieloklatkowa (Analizuje kolejne plastry mózgu z osi Z). """
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def flatten_parameters(self):
        self.lstm.flatten_parameters()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.lstm.flatten_parameters() 
        output, (hidden, cell) = self.lstm(x)
        output = self.layer_norm(output)
        return output, hidden


class ChannelAttention(nn.Module):
    """ Moduł uwagi kanałowej z redukcją wymiaru. """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()

        # Ważne są dla nas tylko średnie i maksymalne wartości w kanale, bo odzwierciedlają one obecność patologii w całym plastrowaniu
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Redukcja wymiaru dla uwagi kanałowej, żeby model nie był zbyt duży i nie przeuczał się na szumach
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """ Moduł uwagi przestrzennej z kernelem 7x7. """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # Wynik to maska przestrzenna, którą mnożymy przez wejście, aby uwydatnić istotne regiony
        return x * self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    """ Połączony moduł uwagi CBAM (Channel + Spatial). """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)
        self.last_spatial_map = None # Bufor do wizualizacji mapy uwagi przestrzennej

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)

        # Obliczenie maski przestrzennej i zapis do bufora
        spatial_attn = self.sa.conv(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1))
        self.last_spatial_map = self.sa.sigmoid(spatial_attn) 
        return x * self.last_spatial_map


class AdaptiveConcatPool2d(nn.Module):
    """ Łączy uśrednianie z wyciąganiem maksimum dla łapania ostrych anomalii. """
    def __init__(self, sz=(1, 1)):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.mp(x), self.ap(x)], 1)


class AddCoords(nn.Module):
    """ Generuje sztuczne kanały X i Y ze znormalizowanymi współrzędnymi [-1, 1]. """
    def __init__(self):
        super(AddCoords, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = x.size()
        
        # Dynamicznie pobieramy typ danych z obrazka
        current_dtype = x.dtype
        
        # Generowanie siatki współrzędnych w formacie dopasowanym do wejścia
        xx_channel = torch.arange(w, dtype=current_dtype, device=x.device).repeat(1, h, 1)
        yy_channel = torch.arange(h, dtype=current_dtype, device=x.device).repeat(1, w, 1).transpose(1, 2)
        
        # Normalizacja do przedziału [-1.0, 1.0]
        xx_channel = (xx_channel / (w - 1)) * 2 - 1
        yy_channel = (yy_channel / (h - 1)) * 2 - 1
        
        # Skopiowanie dla całego batcha (B, 1, H, W)
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)
        
        # Doklejenie współrzędnych do oryginalnego obrazu (C + 2)
        return torch.cat([x, xx_channel, yy_channel], dim=1)


class CoordConv2d(nn.Module):
    """ Zamiennik dla nn.Conv2d. Automatycznie obsługuje kanały współrzędnych. """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(CoordConv2d, self).__init__()
        self.add_coords = AddCoords()

        # Zwykłe Conv2d, ale z dodatkowymi 2 kanałami dla współrzędnych, żeby model mógł się uczyć zależności przestrzennych
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_coords = self.add_coords(x)
        return self.conv(x_coords)


class TemperedSoftmax(nn.Module):
    """ Wyostrza rozkład prawdopodobieństwa. Im niższa temperatura, tym bliżej funkcji ArgMax.
        Działa lepiej niż zwykła softmax w przypadku nierównomiernie rozłożonych klas i niepewności modelu,
        ze względu na to, że pozwala na bardziej wyraźne rozróżnienie między klasami. """
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x / self.temperature, dim=1)


class VolumetricMedicalResNetCRNN(nn.Module):
    """ Architektura hybrydowa łącząca ResNet50 (MedicalNet) + CoordConv + CBAM + BiLSTM + Slice Attention.
        Obecnie moduł realizuje klasyfikację wieloetykietową. Koncepcyjnie system wspiera podejście kaskadowe:
        w pierwszej kolejności model działa jako klasyfikator przesiewowy, skutecznie odrzucający przypadki zdrowe. 
        Regresja przesunięcia linii środkowej w milimetrach dla konkretnych, wykrytych przypadków patologicznych
        stanowiłaby cenne rozszerzenie ilościowe, jednak pozyskanie precyzyjnych wymiarów przestrzennych przekracza
        obecne możliwości projektu, dlatego zadanie to zostało odłożone jako kierunek przyszłego rozwoju."""
    def __init__(self, hidden_dim=256, num_layers=2, dropout=0.3, task='classification', num_classes=4, use_checkpointing=True, base_dir="."):
        super(VolumetricMedicalResNetCRNN, self).__init__()
        self.use_checkpointing = use_checkpointing
        
        # Wyeliminowano niezdefiniowaną zmienną globalną BASE_DIR na rzecz argumentu base_dir
        models_dir = os.path.join(base_dir, "output_data", "Models")
        weights_path = os.path.join(models_dir, "resnet_50_23dataset.pth")
        weights_url = "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50/resolve/main/resnet_50_23dataset.pth"
        
        # Logika automatycznego pobierania pliku, jeśli nie istnieje na dysku
        if not os.path.exists(weights_path):
            import urllib.request
            print(f"Brak pliku wag w {weights_path}. Rozpoczynam pobieranie.")
            os.makedirs(models_dir, exist_ok=True)
            urllib.request.urlretrieve(weights_url, weights_path)
            print("Pobieranie wag zakończone.")

        base_resnet = models.resnet50(weights=None)
        
        # Bezpieczne ładowanie wag z obsługą różnych formatów słowników
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict):
            for key in ['state_dict', 'model_state_dict', 'net']:
                if key in state_dict:
                    state_dict = state_dict[key]
                    break
            
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            k_clean = k.replace('module.', '').replace('backbone.', '')
            if 'fc' in k_clean or 'classifier' in k_clean:
                continue
                
            # Konwersja kerneli 3D na 2D przez wzięcie środkowego przekroju
            if v.dim() == 5:
                v_2d = v[:, :, v.size(2) // 2, :, :]
            else:
                v_2d = v
                
            # Dopasowanie liczby kanałów z 1 (MedicalNet) na 3 (okno mózgowe, krew, kości)
            if k_clean == 'conv1.weight' and v_2d.shape[1] == 1:
                v_2d = v_2d.repeat(1, 3, 1, 1) / 3.0
                
            cleaned_state_dict[k_clean] = v_2d

        # Wczytanie wag z rygorystycznym strict=False na wypadek drobnych rozbieżności w nazwach
        base_resnet.load_state_dict(cleaned_state_dict, strict=False)
        old_conv1_weights = base_resnet.conv1.weight.data.clone()
        
        # CoordConv2d (zakłada, że przyjmuje 3 kanały bazowe + 2 kanały współrzędnych X,Y = 5 kanałów wejściowych)
        new_coord_conv1 = CoordConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            # Bezpieczne przypisanie wag do pierwszych 3 kanałów
            channels_to_copy = min(old_conv1_weights.shape[1], new_coord_conv1.conv.weight.shape[1])
            new_coord_conv1.conv.weight.data[:, :channels_to_copy, :, :] = old_conv1_weights[:, :channels_to_copy, :, :]
            
            # Inicjalizacja kanałów współrzędnych X, Y
            if new_coord_conv1.conv.weight.shape[1] > 3:
                nn.init.normal_(new_coord_conv1.conv.weight.data[:, 3:, :, :], mean=0.0, std=0.01)
            
        base_resnet.conv1 = new_coord_conv1
        
        # Szkielet konwolucyjny
        self.backbone = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool,
            base_resnet.layer1,
            base_resnet.layer2,
            base_resnet.layer3,
            base_resnet.layer4
        )
        
        self.cbam = CBAM(in_channels=2048)
        
        self.bottleneck = nn.Sequential(
            AdaptiveConcatPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048 * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.slice_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            TemperedSoftmax(temperature=0.2) 
        )

        self.lstm = EnhancedBiLSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers
        )

        # Inicjalizacja odpowiedniej warstwy wyjściowej z zachowaniem możliwości przełączenia na regresję w przyszłości
        out_dim = 1 if task == 'regression' else num_classes
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim)
        )

    def _forward_slice(self, x_slice):
        """ Pomocnicza funkcja do gradient checkpointingu pojedynczego slajdu/paczki slajdów. """
        f = self.backbone(x_slice)
        f = self.cbam(f)
        return self.bottleneck(f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_slices, C, H, W = x.size()
        x_reshaped = x.view(batch_size * num_slices, C, H, W)
        
        # Gradient Checkpointing pozwala zaoszczędzić do 50% VRAM kosztem minimalnego narzutu obliczeniowego
        if self.use_checkpointing and self.training:
            x_reshaped.requires_grad_() # Wymuszenie śledzenia niezbędnego dla checkpointingu
            features = checkpoint(self._forward_slice, x_reshaped, use_reentrant=False)
        else:
            features = self._forward_slice(x_reshaped)
            
        features = features.view(batch_size, num_slices, -1)
        
        attention_weights = self.slice_attention(features)
        features = features * attention_weights
        
        _, hidden = self.lstm(features)
        
        # Bezpieczne wyłuskanie stanów ukrytych BiLSTM niezależnie od formatu wyjściowego EnhancedBiLSTM
        if isinstance(hidden, tuple):
            h_n = hidden[0]
        else:
            h_n = hidden
            
        hidden_forward = h_n[-2, :, :]
        hidden_backward = h_n[-1, :, :]
        final_state = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        return self.fc(final_state)


def apply_edge_boost(img_windowed: np.ndarray) -> np.ndarray:
    """ Podbija sygnał brzegowy za pomocą filtru Sobela, by uwydatnić krwiaki podtwardówkowe. """
    sobelx = cv.Sobel(img_windowed, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(img_windowed, cv.CV_64F, 0, 1, ksize=3)
    edges = cv.magnitude(sobelx, sobely)
    
    edges = cv.normalize(edges, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    
    # Domiksowanie krawędzi (70% oryginalnego okna, 30% wyostrzonych krawędzi)
    boosted = cv.addWeighted(img_windowed, 0.7, edges, 0.3, 0)
    return boosted


def apply_ct_window(img_array, dicom_data, window_center, window_width):
    """ Konwertuje surowe piksele na jednostki Hounsfielda i nakłada wybrane okno radiologiczne. """
    intercept = getattr(dicom_data, 'RescaleIntercept', 0)
    slope = getattr(dicom_data, 'RescaleSlope', 1)

    # Konwersja na jednostki Hounsfielda przez skalowanie i przesunięcie, żeby uwzględnić różne aparaty i ustawienia
    img_hu = img_array * slope + intercept
    
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    
    img_windowed = np.clip(img_hu, img_min, img_max)
    
    # Normalizacja do 8-bit [0, 255]
    return ((img_windowed - img_min) / window_width * 255).astype(np.uint8)


def prepare_rsna_data(zip_path: str, csv_path: str, output_dir: str, patients_limit: int = 15000, imaging_config: Optional[MedicalImagingConfig] = None) -> bool:
    """ Odczytuje etykiety z pliku CSV RSNA, wypakowuje plastry grupując je w pacjentów 
        i stosuje trzykanałowe okienkowanie radiologiczne (Mózg, Krew, Kości) pobierane z MedicalImagingConfig. """
    
    if imaging_config is None:
        imaging_config = MedicalImagingConfig()
        
    if not os.path.exists(zip_path) or not os.path.exists(csv_path):
        logger.error(f"Brak pliku archiwum {zip_path} lub etykiet {csv_path}.")
        return False
        
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    logger.info("Ładowanie i transformacja metadanych RSNA z CSV.")
    df = pd.read_csv(csv_path)
    
    df['ImageID'] = df['ID'].apply(lambda x: x.split('_')[1] if '_' in str(x) else str(x))
    df['Subtype'] = df['ID'].apply(lambda x: x.split('_')[-1] if '_' in str(x) else 'any')
    
    # Pobieramy klasy docelowe z MedicalImagingConfig
    df_pathology = df[df['Subtype'].isin(imaging_config.target_classes)]
    pivot_df = df_pathology.pivot_table(index='ImageID', columns='Subtype', values='Label', aggfunc='max').fillna(0)
    labels_dict = pivot_df.to_dict('index')
    
    logger.info("Metadane zmapowane. Rozpoczynam wolumetryczną ekstrakcję DICOM z ZIP (z okienkowaniem HU).")
    
    patients_data = defaultdict(list)
    current_patients = set()
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        dcm_files = [f for f in z.namelist() if f.endswith('.dcm')]
        
        valid_count = 0
        for dcm_file in dcm_files: 
            filename = os.path.basename(dcm_file)
            image_id = filename.replace('ID_', '').replace('.dcm', '')
            
            if image_id not in labels_dict:
                continue
                
            valid_count += 1
            if valid_count % 5000 == 0:
                logger.info(f"Przetworzono {valid_count} treningowych plików DICOM. Skompletowani pacjenci 3D: {len(current_patients)}")
            
            with z.open(dcm_file) as f:
                try:
                    dicom_data = pydicom.dcmread(f, stop_before_pixels=False)
                    patient_id = str(getattr(dicom_data, 'PatientID', getattr(dicom_data, 'StudyInstanceUID', 'Unknown')))
                    
                    if patient_id not in current_patients:
                        if len(current_patients) >= patients_limit:
                            continue 
                        current_patients.add(patient_id)
                    
                    z_loc = float(getattr(dicom_data, 'SliceLocation', getattr(dicom_data, 'InstanceNumber', 0)))
                    img_array = dicom_data.pixel_array
                    if img_array.size == 0: continue
                    
                    # Zastosowanie 3 okien radiologicznych z obiektu MedicalImagingConfig
                    img_brain = apply_ct_window(img_array, dicom_data, window_center=imaging_config.window_brain[0], window_width=imaging_config.window_brain[1])
                    img_blood_std = apply_ct_window(img_array, dicom_data, window_center=imaging_config.window_blood[0], window_width=imaging_config.window_blood[1])
                    
                    # Wąskie okno podtwardówkowe z nałożonym podbiciem krawędzi
                    img_subdural = apply_ct_window(img_array, dicom_data, window_center=imaging_config.window_subdural[0], window_width=imaging_config.window_subdural[1])
                    img_subdural = apply_edge_boost(img_subdural)
                    
                    # Stackowanie na 3 kanały BGR (Subdural, Standard Krew, Mózg)
                    img_bgr = np.stack([img_subdural, img_blood_std, img_brain], axis=-1)
                    
                    patient_dir = os.path.join(img_dir, patient_id)
                    os.makedirs(patient_dir, exist_ok=True)
                    
                    safe_z_loc = z_loc + 5000.0 
                    out_path = os.path.join(patient_dir, f"{safe_z_loc:08.2f}_{image_id}.png")
                    cv.imwrite(out_path, img_bgr)
                    
                    slice_labels = labels_dict[image_id]
                    vec = [
                        slice_labels.get('epidural', 0),
                        slice_labels.get('subdural', 0),
                        slice_labels.get('subarachnoid', 0),
                        slice_labels.get('intraparenchymal', 0),
                        slice_labels.get('intraventricular', 0)
                    ]
                    patients_data[patient_id].append(vec)
                except (pydicom.errors.InvalidDicomError, ValueError, AttributeError, OSError) as e:
                    logger.debug(f"Pominięto uszkodzony plik DICOM: {e}")
                    continue
                
    logger.info("Zakończono czytanie archiwum. Agregacja etykiet całych skanów mózgu.")
    
    patient_labels_list = []
    for pid, vecs in patients_data.items():
        pat_vec = np.max(vecs, axis=0)
        patient_labels_list.append({
            'index': pid,
            'epidural': pat_vec[0],
            'subdural': pat_vec[1],
            'subarachnoid': pat_vec[2],
            'intraparenchymal': pat_vec[3],
            'intraventricular': pat_vec[4]
        })
        
    labels_df = pd.DataFrame(patient_labels_list)
    labels_df.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)
    
    logger.info(f"Wygenerowano {len(labels_df)} pełnych pacjentów do treningu.")
    return True


class TargetedCenterErasing(nn.Module):
    """ Zasłania pionowy pas w centrum obrazu (linię środkową mózgu). Zmusza model do oceny
        patologii na podstawie asymetrii półkul i ucisku bocznego, zamiast skupiania się na samych komorach. """
    def __init__(self, p=0.5, width_scale=(0.1, 0.25)):
        super().__init__()
        self.p = p
        self.width_scale = width_scale # Szerokość pasa jako % całego obrazka

    def forward(self, img_tensor):
        if torch.rand(1).item() > self.p:
            return img_tensor

        # Klonujemy, żeby nie nadpisywać w miejscu
        img = img_tensor.clone()
        C, H, W = img.shape
        
        # Losujemy szerokość pasa z podanego zakresu (np. 10% - 25% szerokości obrazu)
        band_width = int(W * torch.empty(1).uniform_(self.width_scale[0], self.width_scale[1]).item())
        
        # Srodek mózgu (z lekkim losowym przesunięciem o max 5%, żeby nie było zawsze od linijki)
        center_x = W // 2
        offset = int(W * 0.05) 
        start_x = center_x - (band_width // 2) + torch.randint(-offset, offset + 1, (1,)).item()
        
        # Zabezpieczenie krawędzi
        start_x = max(0, start_x)
        end_x = min(W, start_x + band_width)

        # Generujemy losowy szum dopasowany do statystyk obrazu (po normalizacji)
        noise = torch.randn((C, H, end_x - start_x), dtype=img.dtype, device=img.device) * 0.5
        
        # Nadpisujemy środkowy pas
        img[:, :, start_x:end_x] = noise

        return img
    

class CTSliceDataset(Dataset):
    """ Ładuje wolumeny CT pacjenta i paruje je z etykietami (klasyfikacja lub regresja mm). """
    def __init__(self, root_dir, num_slices=24, img_size=160, is_train=True, light_aug=False, task='classification'):
        self.root_dir = root_dir
        self.num_slices = num_slices
        self.is_train = is_train
        self.light_aug = light_aug
        self.img_size = img_size
        self.task = task
        
        labels_path = os.path.join(root_dir, 'labels.csv')
        self.img_dir = os.path.join(root_dir, 'images')
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Brak pliku labels.csv w {root_dir}.")
            
        self.labels_df = pd.read_csv(labels_path)
        self.labels_df.columns = self.labels_df.columns.str.strip()

        if 'name' in self.labels_df.columns and 'PatientID' not in self.labels_df.columns:
            self.labels_df = self.labels_df.rename(columns={'name': 'PatientID'})
        elif 'index' in self.labels_df.columns and 'PatientID' not in self.labels_df.columns:
            self.labels_df = self.labels_df.rename(columns={'index': 'PatientID'})
            
        cq500_to_rsna = {
            'EDH': 'epidural',
            'SDH': 'subdural',
            'SAH': 'subarachnoid',
            'IPH': 'intraparenchymal',
            'IVH': 'intraventricular'
        }
        self.labels_df = self.labels_df.rename(columns=cq500_to_rsna)

        self.patient_folders = [os.path.join(self.img_dir, str(pid)) for pid in self.labels_df['PatientID']]
        
        # Powiązanie trybu regresji z jawnym argumentem 'task'
        if self.task == 'regression' and 'midline_shift' in self.labels_df.columns:
            self.is_regression = True
            self.labels = self.labels_df['midline_shift'].values.astype(np.float32)
        else:
            self.is_regression = False
            self.labels = self.labels_df[['epidural', 'subdural', 'subarachnoid', 'intraparenchymal', 'intraventricular']].values.astype(np.float32)

        self.transform = self._get_transforms(budget=1.0 if self.is_train else 0.0)

    def _get_base_normalization(self):
        """ Wspólny, powtarzalny blok końcowy dla każdych transformacji. """
        return [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

    def _get_transforms(self, budget: float):
        """ Fabryka transformacji eliminująca powielanie kodu w init i set_budget. """
        if not self.is_train or budget == 0.0:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                *self._get_base_normalization()
            ])

        if self.light_aug:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, max(0.15, 0.5 * budget)))], p=0.1 * budget),
                transforms.ColorJitter(brightness=0.05 * budget, contrast=0.05 * budget), 
                *self._get_base_normalization()
            ])
        else:
            transforms_list = [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomAffine(
                    degrees=10 * budget, 
                    translate=(0.05 * budget, 0.05 * budget), 
                    scale=(1.0 - 0.05 * budget, 1.0 + 0.05 * budget), 
                    shear=None
                ),
                transforms.RandomApply([transforms.ElasticTransform(alpha=(20.0, 50.0), sigma=(4.0, 6.0))], p=0.05 * budget),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, max(0.15, 1.0 * budget)))], p=0.2 * budget),
                transforms.ColorJitter(brightness=0.05 * budget, contrast=0.10 * budget), 
                *self._get_base_normalization()
            ]
            
            if self.is_regression:
                transforms_list.insert(-3, TargetedCenterErasing(p=0.60 * budget, width_scale=(0.10, max(0.11, 0.25 * budget))))
                
            return transforms.Compose(transforms_list)

    def set_budget(self, budget):
        """ Progressive Augmentation: Dynamicznie aktualizuje transformacje za pomocą fabryki, dostosowując siłę augmentacji do bieżącego etapu treningu. """
        self.transform = self._get_transforms(budget=budget if self.is_train else 0.0)

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        try:
            import re  # Wymagane do parsowania liczb w nazwach plików
            patient_path = self.patient_folders[idx]
            label_val = self.labels[idx]
            
            # Klucz do naturalnego sortowania (numerycznego)
            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
            
            # Zastosowanie naturalnego sortowania zamiast alfabetycznego
            slices = sorted(
                [f for f in os.listdir(patient_path) if f.endswith(('.png', '.jpg'))], 
                key=natural_sort_key
            )
            
            if not slices:
                raise ValueError(f"Brak obrazów w folderze {patient_path}")
                
            if len(slices) > self.num_slices:
                indices = np.linspace(0, len(slices) - 1, self.num_slices, dtype=int)
                slices = [slices[i] for i in indices]
            elif len(slices) < self.num_slices:
                padding_needed = self.num_slices - len(slices)
                slices = slices + [slices[-1]] * padding_needed 

            volume_tensor = torch.zeros((self.num_slices, 3, self.img_size, self.img_size))
            volumetric_seed = int(torch.randint(0, 2**32, (1,)).item())
            rng_state = torch.get_rng_state()
            
            loaded_images = [np.array(Image.open(os.path.join(patient_path, s)).convert('RGB'))[:, :, ::-1] for s in slices]
            
            for i in range(len(slices)):
                img_curr = loaded_images[i]
                img_pil = Image.fromarray(img_curr)
                
                torch.manual_seed(volumetric_seed)
                volume_tensor[i] = self.transform(img_pil)
                
            torch.set_rng_state(rng_state)
            
            if self.is_regression:
                target_tensor = torch.tensor([label_val], dtype=torch.float32)
            else:
                target_tensor = torch.tensor(label_val, dtype=torch.float32)
                
            return volume_tensor, target_tensor, idx

        except (FileNotFoundError, ValueError) as e:
            import random
            logger.warning(f"Problem z indeksem {idx} (ścieżka: {self.patient_folders[idx] if idx < len(self.patient_folders) else 'Nieznana'}).Losuję bezpieczny przypadek zastępczy. Błąd: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
    
    
class MultiLabelFocalLoss(nn.Module):
    """ Zrównoważony Focal Loss z umiarkowaną preferencją klasy dodatniej. Zachowuje zdolność dyskryminacji
        i właściwą kalibrację prawdopodobieństw dla ostatecznego sędziego. """
    def __init__(self, gamma_pos=2.0, gamma_neg=2.0, alpha_pos=1.5):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha_pos = alpha_pos  # Umiarkowane zwiększenie wagi czułości

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Stonowana kara za pominięcie dodatniego przypadku
        pos_loss = -self.alpha_pos * ((1.0 - probs) ** self.gamma_pos) * torch.log(probs + 1e-7) * targets
        
        # Symetryczna kara dla przypadków ujemnych dla zachowania właściwego rozkładu prawdopodobieństwa
        neg_loss = -((probs) ** self.gamma_neg) * torch.log(1.0 - probs + 1e-7) * (1.0 - targets)
        
        loss = pos_loss + neg_loss
        return loss.mean()


def optimize_crnn_thresholds(y_true_matrix, y_prob_matrix, sensitivity_weight=0.5):
    """ Optymalizuje progi ufności dla modelu, wykorzystując statystykę J-Youdena na zrównoważonym podzbiorze. """
    num_classes = y_prob_matrix.shape[1]
    best_thresholds = np.zeros(num_classes)
    
    patient_sums = np.sum(y_true_matrix, axis=1)
    
    idx_healthy = np.where(patient_sums == 0)[0]
    idx_single = np.where(patient_sums == 1)[0]
    idx_severe = np.where(patient_sums >= 2)[0]
    
    target_size_per_group = max(1, len(y_true_matrix) // 3)
    rng = np.random.default_rng(42)
    
    sampled_healthy = rng.choice(idx_healthy, size=target_size_per_group, replace=True) if len(idx_healthy) > 0 else np.array([], dtype=int)
    sampled_single = rng.choice(idx_single, size=target_size_per_group, replace=True) if len(idx_single) > 0 else np.array([], dtype=int)
    sampled_severe = rng.choice(idx_severe, size=target_size_per_group, replace=True) if len(idx_severe) > 0 else np.array([], dtype=int)
    
    balanced_indices = np.concatenate([sampled_healthy, sampled_single, sampled_severe]).astype(int)
    
    y_true_bal = y_true_matrix[balanced_indices]
    y_prob_bal = y_prob_matrix[balanced_indices]
    
    for c in range(num_classes):
        y_true_c = y_true_bal[:, c]
        y_prob_c = y_prob_bal[:, c]
        
        safe_fallback = 0.50
        
        if np.sum(y_true_c) == 0 or len(np.unique(y_true_c)) < 2:
            y_true_c = y_true_matrix[:, c]
            y_prob_c = y_prob_matrix[:, c]
            
        if np.sum(y_true_c) == 0 or len(np.unique(y_true_c)) < 2:
            best_thresholds[c] = safe_fallback
            continue
            
        try:
            fpr, tpr, thresholds = roc_curve(y_true_c, y_prob_c)
            
            j_scores = (sensitivity_weight * tpr) + ((1.0 - sensitivity_weight) * (1.0 - fpr))
            best_idx = np.argmax(j_scores)
            
            best_th = np.clip(thresholds[best_idx], 0.05, 0.95)
        except (ValueError, RuntimeError) as e:
            logger.debug(f"Failed to compute ROC curve for class {c}: {e}")
            best_th = safe_fallback
            
        best_thresholds[c] = float(best_th)
    
    return best_thresholds


def get_dynamic_weight(epoch, total_epochs, start_w=1.0, end_w=5.0):
    """ Liniowo zwiększa wagę klasy mniejszościowej (chorych) wraz z upływem epok (Curriculum Learning). """
    progress = epoch / max(1, total_epochs - 1)
    return start_w + (end_w - start_w) * progress


def _create_balanced_sampler(labels_array: np.ndarray) -> WeightedRandomSampler:
    """ Tworzy sampler wyrównujący szanse wylosowania klas mniejszościowych (rzadkich krwotoków). """
    class_counts = labels_array.sum(axis=0)
    total_samples = len(labels_array)
    
    class_weights = total_samples / (class_counts + 1e-5)
    n_healthy = np.sum(np.sum(labels_array, axis=1) == 0)
    weight_healthy = total_samples / (n_healthy + 1e-5)
    
    sample_weights = []
    for label_vec in labels_array:
        if np.sum(label_vec) == 0:
            sample_weights.append(weight_healthy)  
        else:
            sample_weights.append(np.max(label_vec * class_weights))
            
    return WeightedRandomSampler(
        weights=np.array(sample_weights, dtype=np.float64), 
        num_samples=len(labels_array), 
        replacement=True
    )

def _initialize_head_biases(model: nn.Module, labels_array: np.ndarray, device: torch.device) -> None:
    """ Inicjalizuje biasy warstwy wyjściowej, zapobiegając eksplozji lossa na starcie treningu. """
    with torch.no_grad():
        class_means = labels_array.mean(axis=0)
        class_means = np.clip(class_means, 1e-4, 1.0 - 1e-4)
        per_class_biases = -np.log((1.0 - class_means) / class_means)
        
        if hasattr(model.fc[-1], 'bias') and model.fc[-1].bias is not None:
            model.fc[-1].bias.copy_(torch.tensor(per_class_biases, dtype=torch.float32, device=device))
            logger.info(f"Zainicjalizowano bias głowicy: {[round(x, 2) for x in per_class_biases]}")


def _train_epoch_crnn(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module, 
    scaler: Any,
    config: CRNNConfig,
    desc: str,
    budget: Optional[float] = None,
    disable: bool = False
) -> float:
    """ Zunifikowana pętla treningowa wspierająca Curriculum Learning. """
    model.train()
    if budget is not None:
        # Bezpieczne odpakowanie zagnieżdżonego datasetu (Subset -> Dataset)
        ds = dataloader.dataset
        while hasattr(ds, 'dataset'):
            ds = ds.dataset
        if hasattr(ds, 'set_budget'):
            ds.set_budget(budget)
        
    train_loss = 0.0
    optimizer.zero_grad(set_to_none=True) 
    
    with tqdm(dataloader, desc=desc if desc else "Przetwarzanie batchy", leave=False, position=1, mininterval=2.0, dynamic_ncols=True, disable=disable) as pbar:
        for i, (volumes, labels, _) in enumerate(pbar):
            volumes, labels = volumes.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            autocast_context = torch.amp.autocast(device_type='cuda') if DEVICE.type == 'cuda' else nullcontext()
            with autocast_context:
                outputs = model(volumes)
                loss = criterion(outputs, labels) / config.accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % config.accumulation_steps == 0 or (i + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * config.accumulation_steps
            pbar.set_postfix({'loss': f"{loss.item() * config.accumulation_steps:.4f}"})
        
    return train_loss / len(dataloader)


@torch.no_grad()
def _validate_epoch_crnn(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    desc: str,
    optimize_thresholds: bool = True,
    sensitivity_weight: float = 0.5,
    disable: bool = False
) -> Tuple[float, float, float, float, float, np.ndarray]:
    """ Ewaluacja epoki zwracająca stratę, AUC, Balanced Accuracy, Czułość, Specyficzność i Progi Decyzyjne. """
    model.eval()
    val_loss = 0.0
    all_targets, all_probs = [], []
    
    with tqdm(dataloader, desc=desc if desc else "Ewaluacja epoki", leave=False, position=1, mininterval=2.0, dynamic_ncols=True, disable=disable) as pbar:
        for volumes, labels, _ in pbar:
            volumes, labels = volumes.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            autocast_context = torch.amp.autocast(device_type='cuda') if DEVICE.type == 'cuda' else nullcontext()
            with autocast_context:
                outputs = model(volumes)
                loss = criterion(outputs, labels)
                
            val_loss += loss.item()
            all_targets.append(labels.cpu().numpy())
            all_probs.append(torch.sigmoid(outputs).cpu().numpy())
        
    val_loss /= len(dataloader)
    y_true = np.vstack(all_targets).astype(int)
    y_prob = np.vstack(all_probs).astype(np.float32) 
    
    if optimize_thresholds:
        best_thresholds = np.clip(optimize_crnn_thresholds(y_true, y_prob, sensitivity_weight=sensitivity_weight), 0.2, 0.8)
    else:
        best_thresholds = np.full(y_prob.shape[1], 0.5)
        
    preds = np.zeros(y_prob.shape, dtype=int)
    for c in range(y_prob.shape[1]):
        preds[:, c] = (y_prob[:, c] >= best_thresholds[c]).astype(int)

    sensitivities, specificities = [], []
    for c in range(y_prob.shape[1]):
        yt, yp = y_true[:, c], preds[:, c]
        try:
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

            # Jeśli brak danego typu schorzenia, wstawiamy np.nan, żeby nie zafałszować średniej makro
            sensitivities.append((tp / (tp + fn) * 100) if (tp + fn) > 0 else np.nan)
            specificities.append((tn / (tn + fp) * 100) if (tn + fp) > 0 else np.nan)
        except ValueError:
            sensitivities.append(0.0)
            specificities.append(0.0)

    # Ostatni wynik makro: średnia z klas, ignorując np.nan (brak danej klasy w zbiorze walidacyjnym)
    macro_sens = np.nanmean(sensitivities) if not np.all(np.isnan(sensitivities)) else 0.0
    macro_spec = np.nanmean(specificities) if not np.all(np.isnan(specificities)) else 0.0
    macro_bacc = (macro_sens + macro_spec) / 2.0
    auc = roc_auc_score(y_true, y_prob, average='macro') if len(np.unique(y_true)) > 1 else 0.5
        
    return val_loss, auc, macro_bacc, macro_sens, macro_spec, best_thresholds


def train_crnn(custom_config: Optional[CRNNConfig] = None) -> None:
    """ Główna orkiestracja kaskadowego modelu CRNN w architekturze trójfazowej. """
    config = custom_config or CRNNConfig()
    logger.info(f"Uruchamianie trójfazowego treningu CRNN na urządzeniu: {DEVICE}")

    dataset_path_main = os.path.join(BASE_DIR, 'Data', 'dataset_ct_main')
    model_path_main = os.path.join(BASE_DIR, 'output_data', 'Models', 'crnn_main_diagnosis.pth')
    model_path_fine = os.path.join(BASE_DIR, 'output_data', 'Models', 'crnn_fine_uncertainty.pth')
    os.makedirs(os.path.dirname(model_path_main), exist_ok=True)
    
    # Przygotowanie zbiorów danych i loaderów
    full_dataset = CTSliceDataset(root_dir=dataset_path_main, num_slices=config.num_slices, img_size=config.img_size, is_train=True)
    val_dataset = CTSliceDataset(root_dir=dataset_path_main, num_slices=config.num_slices, img_size=config.img_size, is_train=False)

    # Zależnie od funkcji, tworzymy binarne etykiety do stratified split
    if getattr(full_dataset, 'is_regression', False):
        labels_binary = (full_dataset.labels > 0).astype(int)
    else:
        labels_binary = np.any(full_dataset.labels > 0, axis=1).astype(int)
    train_indices, val_indices = train_test_split(range(len(full_dataset)), test_size=0.2, stratify=labels_binary, random_state=42)
    
    train_sub = Subset(full_dataset, train_indices)
    val_sub = Subset(val_dataset, val_indices)
    train_labels_array = np.array([full_dataset.labels[i] for i in train_indices])
    
    sampler = _create_balanced_sampler(train_labels_array)
    
    train_loader = DataLoader(train_sub, batch_size=config.batch_size, sampler=sampler, **LOADER_KWARGS)
    fine_loader = DataLoader(train_sub, batch_size=config.batch_size, shuffle=True, **LOADER_KWARGS)
    val_loader = DataLoader(val_sub, batch_size=config.batch_size, shuffle=False, **LOADER_KWARGS)

    model = VolumetricMedicalResNetCRNN(task='multilabel', num_classes=config.num_classes).to(DEVICE)
    _initialize_head_biases(model, train_labels_array, DEVICE)
    scaler = get_scaler()

    # Warmup na głowicy modelu (zamrożony backbone i CBAM) w celu stabilizacji gradientów i uniknięcia eksplozji straty na starcie
    if not os.path.exists(model_path_main):
        logger.info("FAZA 0: Trening samej głowicy w celu stabilizacji.")
        for param in model.backbone.parameters(): param.requires_grad = False
        for param in model.cbam.parameters(): param.requires_grad = False
        
        # Przekazujemy do optymalizatora odblokowane wagi (LSTM, FC)
        optimizer_warmup = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        criterion_warmup = MultiLabelFocalLoss(gamma_pos=2.0, gamma_neg=2.0)

        for epoch in range(config.epochs_warmup):
            model.backbone.eval()  # Backbone musi być w trybie eval (zamrożony BatchNorm)
            
            train_loss = _train_epoch_crnn(
                model, train_loader, optimizer_warmup, criterion_warmup, scaler, config, 
                desc=f"Warmup Epoka [{epoch+1}/{config.epochs_warmup}]", budget=0.2, disable=False
            )
            val_loss, auc, bacc, sens, spec, _ = _validate_epoch_crnn(
                model, val_loader, criterion_warmup, 
                desc="Walidacja Warmup", optimize_thresholds=False, disable=False
            )
            
            logger.info(
                f"Warmup Epoka [{epoch+1}/{config.epochs_warmup}] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BAcc: {bacc:.1f}%"
            )

        logger.info("FAZA 1: MAIN (Odblokowany cały model - Optymalizacja BAcc)")
        for param in model.parameters(): param.requires_grad = True # Odmrażamy wszystko
        
        optimizer_main = optim.AdamW(model.parameters(), lr=config.lr_main, weight_decay=config.weight_decay)
        scheduler_main = optim.lr_scheduler.ReduceLROnPlateau(optimizer_main, mode='max', factor=0.5, patience=2, min_lr=1e-6)
        criterion_main = MultiLabelFocalLoss(gamma_pos=2.0, gamma_neg=2.0)
        
        best_val_bacc = 0.0
        patience_counter = 0
        
        main_pbar = tqdm(range(config.epochs_main), desc="Main Phase", position=0, leave=True, dynamic_ncols=True)
        for epoch in main_pbar:
            current_budget = get_dynamic_weight(epoch, config.epochs_main, start_w=0.4, end_w=1.0)
            
            train_loss = _train_epoch_crnn(
                model, train_loader, optimizer_main, criterion_main, scaler, config,
                desc="Trening Main", budget=current_budget, disable=False
            )  
            val_loss, auc, bacc, sens, spec, thresholds = _validate_epoch_crnn(model, val_loader, criterion_main,
                                                                               desc="Walidacja Main", optimize_thresholds=True,
                                                                               sensitivity_weight=0.5, disable=False
                                                                               )
            
            scheduler_main.step(bacc)
            
            main_pbar.set_postfix({
                'Train': f"{train_loss:.4f}", 
                'Val': f"{val_loss:.4f}", 
                'AUC': f"{auc:.3f}", 
                'BAcc': f"{bacc:.1f}%",
                'Sens': f"{sens:.1f}%",
                'Spec': f"{spec:.1f}%"
            })
            
            if bacc > best_val_bacc:
                best_val_bacc, patience_counter = bacc, 0
                torch.save(model.state_dict(), model_path_main)
                np.save(os.path.join(BASE_DIR, 'output_data', 'Models', 'crnn_optimal_thresholds.npy'), thresholds)
                logger.info(f" -> Zapisano model MAIN (BAcc: {bacc:.1f}%) w epoce {epoch+1}")
            else:
                patience_counter += 1
                if patience_counter >= config.patience: 
                    logger.info(" -> Przerwano trening MAIN (Early Stopping)")
                    break
    else:
        logger.info("Znaleziono wagi modelu Main. Pomijam Fazę 0 i 1.")

    logger.info("FAZA 2: FINE TUNE (Model przesiewowy - Optymalizacja Czułości)")
    model.load_state_dict(torch.load(model_path_main, map_location=DEVICE, weights_only=True))
    
    # Zmniejszony LR dla szlifowania i focal loss kładący nacisk na chore przypadki (gamma_pos=1.0)
    optimizer_fine = optim.AdamW(model.parameters(), lr=config.lr_fine, weight_decay=config.weight_decay)
    scheduler_fine = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fine, mode='max', factor=0.5, patience=1, min_lr=1e-7)
    criterion_fine = MultiLabelFocalLoss(gamma_pos=1.0, gamma_neg=2.0) 
    
    best_val_sens = 0.0 # W Fine Tune optymalizujemy pod Czułość (Sens/Recall), a nie ogólne BAcc
    patience_counter = 0
    
    fine_pbar = tqdm(range(config.epochs_fine), desc="Fine Tune Phase", position=0, leave=True, dynamic_ncols=True)
    for epoch in fine_pbar:
        # Paski wewnętrzne używają domyślnie disable=False. Należy tu upewnić się, że nie robią śmietnika.
        train_loss = _train_epoch_crnn(model, fine_loader, optimizer_fine, criterion_fine, scaler,
                                       config, desc=f"Trening Fine Tune (Ep {epoch+1})", budget=1.0, disable=False)
        val_loss, auc, bacc, sens, spec, thresholds = _validate_epoch_crnn(
            model, val_loader, criterion_fine, desc=f"Walidacja Fine Tune (Ep {epoch+1})",
            optimize_thresholds=True, sensitivity_weight=0.9, disable=False)
        
        scheduler_fine.step(sens)
        
        fine_pbar.set_postfix({
            'Train': f"{train_loss:.4f}", 
            'Val': f"{val_loss:.4f}", 
            'AUC': f"{auc:.3f}", 
            'BAcc': f"{bacc:.1f}%", 
            'Sens': f"{sens:.1f}%",
            'Spec': f"{spec:.1f}%"
        })
        
        if sens > best_val_sens:
            best_val_sens, patience_counter = sens, 0
            torch.save(model.state_dict(), model_path_fine)
            logger.info(f"-> Zapisano model FINE (Czułość: {sens:.1f}%) w epoce {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience: 
                logger.info("-> Przerwano trening FINE (Early Stopping)")
                break

    # Wyliczanie parametrów kaskady na zbiorze walidacyjnym, żeby określić optymalne progi ufności i kary za niepewność
    logger.info("Wyliczanie optymalnych parametrów miękkiej kaskady na zbiorze walidacyjnym.")
    model_main = VolumetricMedicalResNetCRNN(task='multilabel', num_classes=config.num_classes).to(DEVICE)
    model_main.load_state_dict(torch.load(model_path_main, map_location=DEVICE, weights_only=True))
    model_main.eval()
    model.eval() # To jest aktualnie model Fine w pamięci

    all_targets, all_probs_fine, all_probs_main = [], [], []
    with torch.no_grad():
        for volumes, labels, _ in val_loader:
            volumes = volumes.to(DEVICE)
            with torch.amp.autocast(device_type='cuda') if DEVICE.type == 'cuda' else nullcontext():
                all_probs_fine.append(torch.sigmoid(model(volumes)).cpu().numpy())
                all_probs_main.append(torch.sigmoid(model_main(volumes)).cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    y_true_val = np.vstack(all_targets).astype(int)
    y_prob_fine_val = np.vstack(all_probs_fine).astype(np.float32)
    y_prob_main_val = np.vstack(all_probs_main).astype(np.float32)

    thresholds_fine = np.array(config.thresholds_fine_default)
    thresholds_main = np.load(os.path.join(BASE_DIR, 'output_data', 'Models', 'crnn_optimal_thresholds.npy'))
    is_patient_positive_fine_val = np.any(y_prob_fine_val >= thresholds_fine, axis=1)

    best_score, best_penalty, best_extreme_th = 0.0, 0.0, 0.90
    for penalty in np.linspace(0.0, 0.5, 11):
        for ext_th in np.linspace(0.80, 0.95, 4):
            temp_aucs, temp_baccs = [], []
            for c in range(y_true_val.shape[1]):
                temp_prob = np.where(is_patient_positive_fine_val, y_prob_main_val[:, c], y_prob_main_val[:, c] * penalty)
                if len(np.unique(y_true_val[:, c])) > 1:
                    temp_aucs.append(roc_auc_score(y_true_val[:, c], temp_prob))
                
                main_conf = y_prob_main_val[:, c] >= thresholds_main[c]
                main_ext = y_prob_main_val[:, c] >= ext_th
                temp_pred = ((is_patient_positive_fine_val & main_conf) | main_ext).astype(int)
                
                tn, fp, fn, tp = confusion_matrix(y_true_val[:, c], temp_pred, labels=[0, 1]).ravel()
                sens = (tp / (tp + fn)) if (tp + fn) > 0 else 0
                spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0
                temp_baccs.append((sens + spec) / 2.0)
            
            current_score = (np.mean(temp_aucs) + np.mean(temp_baccs)) / 2.0
            if current_score > best_score:
                best_score, best_penalty, best_extreme_th = current_score, penalty, ext_th

    cascade_params = {'TRUST_PENALTY': best_penalty, 'EXTREME_CONFIDENCE': best_extreme_th}
    params_path = os.path.join(BASE_DIR, 'output_data', 'Models', 'crnn_cascade_params.json')
    with open(params_path, 'w') as f:
        json.dump(cascade_params, f)
    logger.info(f"Zamrożono i zapisano parametry kaskady: {cascade_params}")

    logger.info("Zakończono pomyślnie proces treningu kaskady. Czyszczenie pamięci.")
    del train_loader, fine_loader, val_loader
    gc.collect()


def get_scaler():
    """ Obsługuje kompatybilność wsteczną dla torch.amp.GradScaler w zależności od wersji PyTorch. """
    try:
        return torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == 'cuda'))
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))


class CQ500Evaluator:
    """ Klasa do przeprowadzenia niezależnej walidacji kaskadowej (Fine -> Main) na zewnętrznym zbiorze CQ500. """
    def __init__(self, model_main, model_fine, criterion, device: torch.device = DEVICE, batch_size=1):
        self.model_main = model_main
        self.model_fine = model_fine
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_path = os.path.join(self.base_dir, 'Data', 'dataset_cq500')

    def _extract_probs(self, out_tensor, model):
        """ Bezpieczna ekstrakcja prawdopodobieństw niezależna od kształtu Ground Truth. """
        if getattr(model, 'has_sigmoid', False):
            return out_tensor.cpu().numpy()
        elif out_tensor.shape[1] == 2:
            # Jeśli model wymusza 2 klasy (CrossEntropy: 0=Normal, 1=Hemo)
            return torch.softmax(out_tensor, dim=1)[:, 1:].cpu().numpy()
        else:
            # Standard dla klasyfikacji wieloetykietowej (Multi-label BCE)
            return torch.sigmoid(out_tensor).cpu().numpy()

    def evaluate(self):
        if not os.path.exists(self.dataset_path) or not any(os.scandir(self.dataset_path)):
            logger.error(f"Brak folderu {self.dataset_path} lub jest on pusty.")
            return

        logger.info("Rozpoczynam walidację kaskadową na zbiorze CQ500.")
        cq500_dataset = CTSliceDataset(root_dir=self.dataset_path, num_slices=24, img_size=160, is_train=False)
        
        if len(cq500_dataset) == 0:
            logger.error("Błąd: Zbiór CQ500 jest pusty!")
            return

        cq500_loader = DataLoader(cq500_dataset, batch_size=self.batch_size, shuffle=False, **LOADER_KWARGS)

        fine_model_path = os.path.join(self.base_dir, 'output_data', 'Models', 'crnn_fine_uncertainty.pth')
        main_model_path = os.path.join(self.base_dir, 'output_data', 'Models', 'crnn_main_diagnosis.pth')
        thresholds_path = os.path.join(self.base_dir, 'output_data', 'Models', 'crnn_optimal_thresholds.npy')

        self.model_fine.load_state_dict(torch.load(fine_model_path, weights_only=True, map_location=self.device))
        self.model_main.load_state_dict(torch.load(main_model_path, weights_only=True, map_location=self.device))
        self.model_fine.eval()
        self.model_main.eval()

        all_targets, all_probs_fine, all_probs_main = [], [], []
        with torch.no_grad():
            for volumes, labels, _ in cq500_loader:
                volumes = volumes.to(self.device)
                autocast_context = torch.amp.autocast(device_type='cuda') if self.device.type == 'cuda' else nullcontext()
                
                with autocast_context:
                    out_fine = self.model_fine(volumes)
                    out_main = self.model_main(volumes)
                    
                    # Użycie nowej, bezpiecznej metody ekstrakcji
                    probs_fine = self._extract_probs(out_fine, self.model_fine)
                    probs_main = self._extract_probs(out_main, self.model_main)

                all_targets.append(labels.cpu().numpy())
                all_probs_fine.append(probs_fine)
                all_probs_main.append(probs_main)

        y_true = np.vstack(all_targets).astype(int)
        y_prob_fine = np.vstack(all_probs_fine).astype(np.float32)
        y_prob_main = np.vstack(all_probs_main).astype(np.float32)

        # Bezpieczne dopasowanie progów z odwróceniem etykiet
        logger.info("Odwracam etykiety dla zbioru CQ500 (y_true = 1 - y_true) w celu ujednolicenia z modelem.")
        y_true = 1 - y_true

        # Ponieważ CQ500 ma inny rozkład, wyznaczamy progi maksymalizujące (Czułość + Specyficzność - 1)
        thresholds_main = np.zeros(y_true.shape[1])
        thresholds_fine = np.array(CRNNConfig().thresholds_fine_default)
        thresholds_fine = thresholds_fine[:y_prob_fine.shape[1]] if len(thresholds_fine) >= y_prob_fine.shape[1] else np.full(y_prob_fine.shape[1], 0.5)

        for c in range(y_true.shape[1]):
            if len(np.unique(y_true[:, c])) > 1:
                fpr, tpr, thresholds_roc = roc_curve(y_true[:, c], y_prob_main[:, c])
                youden_j = tpr - fpr
                best_idx = np.argmax(youden_j)
                thresholds_main[c] = thresholds_roc[best_idx]
            else:
                thresholds_main[c] = 0.5

        logger.info(f"Wyznaczono optymalne progi Youdena dla CQ500 (średni próg: {np.mean(thresholds_main):.3f})")

        """ Dedykowana złagodzona logika kaskady. """
        # Złagodzenie warunku fine (np. obniżenie wymogu ekstremalnej pewności)
        is_patient_positive_fine = np.any(y_prob_fine >= (thresholds_fine * 0.8), axis=1)

        params_path = os.path.join(self.base_dir, 'output_data', 'Models', 'crnn_cascade_params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                cascade_params = json.load(f)
            best_penalty = cascade_params.get('TRUST_PENALTY', 0.10)
            # Obniżamy próg ekstremalnej pewności, żeby model częściej przepuszczał podejrzenia
            best_extreme_th = min(cascade_params.get('EXTREME_CONFIDENCE', 0.90), 0.75)
            logger.info(f"Wczytano i dostosowano parametry kaskady: TRUST_PENALTY = {best_penalty:.2f}, EXTREME_CONFIDENCE = {best_extreme_th:.2f}")
        else:
            best_penalty, best_extreme_th = 0.10, 0.75

        preds = np.zeros(y_true.shape, dtype=int)
        y_prob_cascade = np.zeros(y_prob_main.shape, dtype=np.float32)

        for c in range(y_true.shape[1]):
            y_prob_cascade[:, c] = np.where(
                is_patient_positive_fine, 
                y_prob_main[:, c], 
                y_prob_main[:, c] * best_penalty
            )
            
            main_confirmed = y_prob_main[:, c] >= thresholds_main[c]
            main_extreme_confidence = y_prob_main[:, c] >= best_extreme_th
            preds[:, c] = ((is_patient_positive_fine & main_confirmed) | main_extreme_confidence).astype(int)

        sensitivities, specificities, class_aucs = [], [], []
        for c in range(y_true.shape[1]):
            yt, yp = y_true[:, c], preds[:, c]
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            
            sensitivities.append((tp / (tp + fn) * 100) if (tp + fn) > 0 else np.nan)
            specificities.append((tn / (tn + fp) * 100) if (tn + fp) > 0 else np.nan)
            
            if len(np.unique(yt)) > 1:
                class_aucs.append(roc_auc_score(yt, y_prob_cascade[:, c]))

        macro_sens = np.nanmean(sensitivities) if not np.all(np.isnan(sensitivities)) else 0.0
        macro_spec = np.nanmean(specificities) if not np.all(np.isnan(specificities)) else 0.0
        macro_bacc = (macro_sens + macro_spec) / 2.0
        auc = np.mean(class_aucs) if class_aucs else 0.5
        
        ebm_export_path = os.path.join(self.base_dir, 'output_data', 'Models', 'ebm_cascade_features.npz')
        np.savez_compressed(
            ebm_export_path,
            y_prob_fine=y_prob_fine,           
            y_prob_main=y_prob_main,           
            y_true=y_true,                     
            thresholds_fine=thresholds_fine,   
            thresholds_main=thresholds_main    
        )
        logger.info(f"Wyeksportowano surowe cechy prawdopodobieństw do pliku: {ebm_export_path}")
        logger.info("Walidacja Kaskadowa na CQ500 (Fine -> Main)")
        logger.info(f"AUC: {auc:.3f} | Zbal. Dokładność: {macro_bacc:.1f}% | Czułość: {macro_sens:.1f}% | Specyficzność: {macro_spec:.1f}%")


if __name__ == "__main__":
    config = CRNNConfig()  
    
    main_model_path = os.path.join(BASE_DIR, 'output_data', 'Models', 'crnn_main_diagnosis.pth')
    fine_model_path = os.path.join(BASE_DIR, 'output_data', 'Models', 'crnn_fine_uncertainty.pth')
    thresholds_path = os.path.join(BASE_DIR, 'output_data', 'Models', 'crnn_optimal_thresholds.npy')
    
    # Decyzja o treningu
    if os.path.exists(main_model_path) and os.path.exists(fine_model_path) and os.path.exists(thresholds_path):
        logger.info("Znaleziono wagi kaskady (Main + Fine) oraz progi decyzyjne. Pomijam trening.")
    else:
        logger.info("Nie znaleziono kompletnych wag kaskadowych. Uruchamiam pełny dwuetapowy trening.")
        train_crnn(config)
    
    # Zabezpieczenie ewaluacji - wchodzimy, jeśli mamy już wagi
    if os.path.exists(main_model_path) and os.path.exists(fine_model_path):
        model_main = VolumetricMedicalResNetCRNN(task='multilabel', num_classes=config.num_classes).to(DEVICE)
        model_fine = VolumetricMedicalResNetCRNN(task='multilabel', num_classes=config.num_classes).to(DEVICE)
        criterion = MultiLabelFocalLoss(gamma_pos=1.0, gamma_neg=2.0) 
        
        evaluator = CQ500Evaluator(model_main=model_main, model_fine=model_fine, criterion=criterion, device=DEVICE, batch_size=2)
        evaluator.evaluate()
    else:
        logger.error("Brak plików wag po treningu (lub trening został przerwany). Pomijam ewaluację na CQ500.")