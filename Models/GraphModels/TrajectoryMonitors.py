from datetime import datetime, timedelta
import json
import logging
import socket
from tokenize import group
from typing import Any, Dict, Optional, Tuple
import os
import pickle
import gc
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import redis
import torch
import torch.nn as nn
from tqdm import tqdm

# Bezpieczny import solvera Neural ODE
try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None

logger = logging.getLogger(__name__)

# Podstawowa konfiguracja projektu
CONFIG = {"PLOTS_DIR": "./plots"}

# Dynamiczne wykrywanie ścieżki do głównego katalogu projektu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SonodeFalseAlarmDetector:
    """ Pełnoprawny detektor fałszywych alarmów, artefaktów sprzętowych oraz ryzyka
        zatruć/intoksykacji farmakologicznej. Wykorzystuje rozszerzoną przestrzeń fazową
        i dynamikę drugiego rzędu do rozwikłania nieliniowych problemów typu XOR
        (np. rozróżnianie kontrolowanej, bezpiecznej śpiączki farmakologicznej od krytycznej
        intoksykacji lub wgłobienia pnia mózgu przy zbieżnych wartościach pojedynczych parametrów). """
    def __init__(self, contamination=0.1, device="cpu"):
        self.device = device
        self.is_fitted = False
        self.scaler = StandardScaler()

        # Model bazowy detekcji anomalii (izoluje punkty odstające i artefakty sygnałowe)
        self.anomaly_detector = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )

        """ To jest wektor stanu homeostazy układowo-neurologicznej. 
            Reprezentuje on minimalny zestaw 6 kluczowych cech klinicznych monitorowanych w czasie rzeczywistym:
            - heart_rate_max (maksymalne tętno)
            - sysbp_min (minimalne ciśnienie skurczowe)
            - map_min (minimalne ciśnienie średnie)
            - spo2_min (minimalna saturacja tlenu)
            - glucose_max (maksymalny poziom glukozy we krwi)
            - gcs_sum (suma punktów w skali Glasgow Coma Scale)
            Uzasadnienie doboru i ograniczenia do tych konkretnych cech:
            1. Pokrywają one 4 niezależne filary fizjologii krytycznej w OIT: układ krążenia, układ oddechowy, metabolizm oraz ośrodkowy układ nerwowy.
            2. Umożliwiają skuteczną detekcję artefaktów sprzętowych (fałszywych alarmów, np. odklejona elektroda EKG) oraz ryzyka zatruć i intoksykacji farmakologicznej.
            3. Pozwalają rozwikłać nieliniowe problemy decyzyjne typu XOR (np. rozróżnienie bezpiecznej, kontrolowanej sedacji farmakologicznej od krytycznego
                wgłobienia pnia mózgu przy identycznych wskazaniach pojedynczych parametrów).
            4. Ograniczenie liczby cech wyłącznie do tych ortogonalnych wskaźników chroni system analityczny przed lawiną fałszywych alarmów wywoływaną
                przez chwilowy szum dziesiątek innych, mniej istotnych sensorów. """
        self.monitor_features = [
            'heart_rate_max', 'sysbp_min', 'map_min',
            'spo2_min', 'glucose_max', 'gcs_sum'
        ]

    def fit(self, X_train: pd.DataFrame):
        """ Trenuje bazę odniesienia (profil normalnej fizjologii OIT) na danych historycznych. """
        features = [col for col in self.monitor_features if col in X_train.columns]
        if not features:
            raise ValueError(
                "Dane treningowe detektora SONODE muszą zawierać co najmniej jedną "
                "kliniczną cechę monitorowaną."
            )

        self.monitor_features = features
        X_clean = X_train[features].fillna(X_train[features].median()).values

        X_scaled = self.scaler.fit_transform(X_clean)
        self.anomaly_detector.fit(X_scaled)
        self.is_fitted = True
        return self

    def detect_false_alarms_and_poisoning(self, patient_state_df):
        """ Analizuje strumień danych pacjenta, wykrywając artefakty ruchowe (fałszywe alarmy),
            niestabilność dynamiki drugiego rzędu oraz ryzyko intoksykacji farmakologicznej. """
        if isinstance(patient_state_df, dict):
            patient_state_df = pd.DataFrame([patient_state_df])

        results = []

        # Upewniamy się, że kolumny istnieją, z bezpiecznymi wartościami domyślnymi OIT
        df = patient_state_df.copy()
        for col in self.monitor_features:
            if col not in df.columns:
                df[col] = 100.0 if 'glucose' in col else (80.0 if 'rate' in col else 120.0)

        X_eval = df[self.monitor_features].fillna(0.0).values

        if not self.is_fitted:
            raise RuntimeError(
                "SonodeFalseAlarmDetector musi być wcześniej wytrenowany metodą "
                "fit() na danych historycznych przed analizą pacjenta."
            )

        X_eval_scaled = self.scaler.transform(X_eval)

        # Ocena izolacji anomalii (-1 to anomalia/artefakt, 1 to norma)
        anomaly_preds = self.anomaly_detector.predict(X_eval_scaled)
        anomaly_scores = self.anomaly_detector.decision_function(X_eval_scaled)

        for position, (_, row) in enumerate(df.iterrows()):
            hr = row.get("heart_rate_max", 80.0)
            sysbp = row.get("sysbp_min", 120.0)
            glucose = row.get("glucose_max", 100.0)
            spo2 = row.get("spo2_min", 98.0)
            gcs = row.get("gcs_sum", 15.0)

            is_isolated_outlier = anomaly_preds[position] == -1
            raw_score = anomaly_scores[position]

            """ Heurystyka przestrzeni fazowej / dynamiki drugiego rzędu:
                Nagłe, fizjologicznie niemożliwe skoki (np. skrajne Tętno przy prawidłowym ciśnienie i SpO2)
                to klasyczny artefakt obluzowanej elektrody EKG (fałszywy alarm). """
            is_electrode_artifact = (hr > 150 and sysbp > 100 and spo2 > 95) or (spo2 < 60 and hr > 100 and sysbp > 110)

            # Prawdopodobieństwo fałszywego alarmu wyprowadzone z pewności modelu i wzorców artefaktów
            if is_electrode_artifact:
                false_alarm_prob = 0.92
                alarm_reliability = 0.10
            elif is_isolated_outlier and raw_score < -0.1:
                false_alarm_prob = 0.65
                alarm_reliability = 0.35
            else:
                false_alarm_prob = 0.15
                alarm_reliability = 0.88

            # Sprawdzamy kliniczne wskaźniki zapaści lub depresji OUN (np. głęboka bradykardia + hipotensja + spadek GCS)
            intoxication_indicators = 0
            if hr < 50: intoxication_indicators += 1
            if sysbp < 85: intoxication_indicators += 1
            if glucose > 300 or glucose < 50: intoxication_indicators += 1
            if gcs <= 6 and spo2 > 95: intoxication_indicators += 1  # Śpiączka niemetaboliczna / farmakologiczna

            intoxication_risk = min(0.95, float(intoxication_indicators * 0.25 + (0.2 if is_isolated_outlier else 0.0)))

            results.append({
                "false_alarm_prob": float(false_alarm_prob),
                "intoxication_risk": float(intoxication_risk),
                "alarm_reliability": float(alarm_reliability),
            })

        if len(results) == 1:
            return results[0]

        # Agregacja wyników dla całego wsadu pacjentów
        avg_fa = sum(r["false_alarm_prob"] for r in results) / len(results)
        avg_tox = sum(r["intoxication_risk"] for r in results) / len(results)
        avg_rel = sum(r["alarm_reliability"] for r in results) / len(results)

        return {
            "false_alarm_prob": avg_fa,
            "intoxication_risk": avg_tox,
            "alarm_reliability": avg_rel,
        }


def compute_ebm_integrated_values(patient_state, sonode_model, drug_models, baseline_ebm_scores):
    """ Integruje wyniki detekcji fałszywych alarmów/zatruć Sonode oraz modele
        lekowe jako ustrukturyzowane wartości w systemie EBM. """
    ebm_values = baseline_ebm_scores.copy()

    # Integracja Sonode z zabezpieczeniem przed brakiem modelu
    if sonode_model is not None:
        sonode_eval = sonode_model.detect_false_alarms_and_poisoning(patient_state)
        ebm_values["sonode_false_alarm_prob"] = float(
            sonode_eval.get("false_alarm_prob", 0.0)
        )
        ebm_values["sonode_intoxication_risk"] = float(
            sonode_eval.get("intoxication_risk", 0.0)
        )
        ebm_values["sonode_alarm_reliability"] = float(
            sonode_eval.get("alarm_reliability", 1.0)
        )
    else:
        ebm_values["sonode_false_alarm_prob"] = 0.0
        ebm_values["sonode_intoxication_risk"] = 0.0
        ebm_values["sonode_alarm_reliability"] = 1.0

    # Pętla integracji modeli lekowych
    drug_evaluations = {}
    if drug_models:
        for drug_name, model in drug_models.items():
            effect = model.evaluate(patient_state)
            drug_evaluations[f"drug_efficacy_{drug_name}"] = float(
                effect.get("efficacy_score", 0.0)
            )
            drug_evaluations[f"drug_toxicity_risk_{drug_name}"] = float(
                effect.get("toxicity_risk", 0.0)
            )
            drug_evaluations[f"drug_ebm_weight_{drug_name}"] = float(
                effect.get("ebm_evidence_weight", 1.0)
            )

    ebm_values.update(drug_evaluations)

    # Agregacja do kompozytowej wartości EBM (uwzględniająca tylko wartości numeryczne, ponieważ niektóre modele mogą zwracać słowniki z różnymi typami danych)
    numeric_vals = [v for v in ebm_values.values() if isinstance(v, (int, float))]
    if numeric_vals:
        ebm_values["ebm_composite_value"] = sum(numeric_vals) / len(numeric_vals)
    else:
        ebm_values["ebm_composite_value"] = 0.0

    return ebm_values


class PatientStateSONODE(nn.Module):
    """ Równanie różniczkowe drugiego rzędu dla fizjologii pacjenta.

        Uzasadnienie teoretyczne i kliniczne:
        Zwykłe Neural ODE (pierwszego rzędu) napotykają na fundamentalną barierę topologiczną
        wynikającą z twierdzenia Picarda-Lindelöfa: ich trajektorie w przestrzeni stanów nie mogą się
        przecinać. Powoduje to, że nieliniowo separowalne pułapki i stany ryzyka są dla nich matematycznie
        niemożliwe do rozwikłania:
        1. Problem XOR z GCS i sedacją: Izolowane objawy bywają mylące (np. wysoki GCS bez sedacji to stan
           stabilny, ale nagły spadek GCS przy braku leków sedacyjnych to alarm krytyczny). Zwykłe ODE nie potrafi
           przeciąć trajektorii tych pacjentów, co generuje fałszywe alarmy.
        2. Fizjologiczne Nested Spheres (koncentryczne sfery ryzyka): Fizjologia opiera się na optimum homeostazy
           (efekt U-Shape). Zarówno zbyt niskie, jak i zbyt wysokie wartości (np. hipotensja vs hypertensja)
           róvnocześnie zagrażają mózgowi. Standardowe ODE dusi się, próbując rozdzielić sferę zdrowia uwięzioną
           wewnątrz sfery stanu krytycznego bez możliwości przecięcia warstw.

        SONODE pokonuje te ograniczenia, rozszerzając przestrzeń fazową o wymiar dynamiki i przyspieszenia.
        Dzięki temu trajektorie mogą swobodnie przechodzić nad sobą w wyższym wymiarze, umożliwiając analizę
        drugiej pochodnej i natychmiastowe wychwycenie nieliniowego, przyspieszającego załamania homeostazy,
        w tym kluczowych stanów neurokrytycznych:
        - Wyczerpanie rezerwy kompensacyjnej czaszki (Wejście na strome ramię wykładniczej krzywej podatności objętościowej).
        - Opóźniona ekspansja krwiaka / Lucid Interval (Gdy pacjent z GCS 15 nagle zaczyna gwałtownie manewrować wektorami ciśnienia
          i tętna przed utratą przytomności z powodu pęknięcia tętnicy oponowej środkowej).
        - Dynamiczny Odruch Cushinga (Wykrywanie rozpędzania się rozjazdu ciśnienia skurczowego i bradykardii jako zwiastuna wgłobienia).
        - Dekompresja we wstrząsie krwotocznym (Moment, w którym mechanizmy wazokonstrykcji pękają, wywołując logarytmiczny spadek MAP). """
    def __init__(self, hidden_dim=16, treatment_dim=1):
        super().__init__()
        self.treatments = None
        self.treatment_times = None
        self.treatment_dim = treatment_dim

        # Sieć przyjmuje połączony stan i prędkość zmian [h, v]
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.treatment_net = nn.Sequential(
            nn.Linear(treatment_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t, state):
        """ Ponieważ torchdiffeq oczekuje wektora stanu, dla SONODE rozszerzamy wymiar o prędkość. """
        h, v = torch.chunk(state, 2, dim=-1)

        # Pochodna stanu to obecna prędkość: dh/dt = v
        dh_dt = v

        # Pochodna prędkości (przyspieszenie) to nieliniowa funkcja sił fizjologicznych: dv/dt = f(h, v)
        combined_input = torch.cat([h, v], dim=-1)

        dv_dt = self.net(combined_input)
        if self.treatments is not None:
            if self.treatment_times is None:
                raise RuntimeError(
                    "Harmonogram leczenia wymaga ustawienia treatment_times."
                )

            treatment_position = torch.searchsorted(
                self.treatment_times.to(t.device),
                t.detach().reshape(1),
                right=True,
            ).item()
            treatment_position = min(
                max(treatment_position - 1, 0),
                self.treatments.shape[1] - 1,
            )
            treatment = self.treatments[:, treatment_position, :].to(
                device=state.device,
                dtype=state.dtype,
            )
            if treatment.shape[0] == 1 and state.shape[0] != 1:
                treatment = treatment.expand(state.shape[0], -1)
            if treatment.shape[0] != state.shape[0]:
                raise ValueError(
                    "Liczba pacjentów w treatments musi odpowiadać batchowi stanu."
                )
            dv_dt = dv_dt + self.treatment_net(treatment)

        return torch.cat([dh_dt, dv_dt], dim=-1)

    def set_treatments(self, treatments, treatment_times):
        if treatments.ndim != 3:
            raise ValueError("treatments musi mieć kształt [batch, czas, cechy].")
        if treatment_times.ndim != 1:
            raise ValueError("treatment_times musi być wektorem czasu.")
        if treatments.shape[1] != treatment_times.numel():
            raise ValueError(
                "Liczba kroków treatments musi odpowiadać treatment_times."
            )
        if treatments.shape[2] != self.treatment_dim:
            raise ValueError(
                f"Oczekiwano {self.treatment_dim} cech leczenia, "
                f"otrzymano {treatments.shape[2]}."
            )
        self.treatments = treatments
        self.treatment_times = treatment_times


class LatentODE_VAE_Generator(nn.Module):
    """ Generatywny Model Neural ODE połączony z Variational Autoencoder.
        Zabezpiecza przed overfittingiem i generowaniem medycznych halucynacji. """
    def __init__(self, input_dim=22, hidden_dim=16, treatment_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # RNN czyta historię pacjenta
        self.encoder_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Przekształcenie w rozkład prawdopodobieństwa przy użyciu VAE
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)       # Średnia rozkładu
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)   # Logarytm wariancji

        # ODE dla dynamiki stanu pacjenta w czasie. To jest serce modelu, które przewiduje trajektorie fizjologiczne.
        self.ode_func = PatientStateSONODE(hidden_dim, treatment_dim)

        # Odtwarzanie surowych parametrów medycznych
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

        # Przewidywanie ryzyka zgonu w czasie na podstawie trajektorii ukrytej
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """ Reparameterization Trick: Pozwala na propagację wsteczną
            przez proces losowego próbkowania z przestrzeni ukrytej. """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Szum z rozkładu normalnego N(0,1)
        return mu + eps * std

    def forward(self, vitals_seq, t_seq):
        """ Przepływ treningowy z obliczaniem straty KLD """
        # Najpierw RNN przetwarza historię
        _, h_rnn = self.encoder_rnn(vitals_seq)
        h_rnn = h_rnn.squeeze(0)

        # Projekcja do przestrzeni VAE
        mu = self.fc_mu(h_rnn)
        logvar = self.fc_logvar(h_rnn)

        # Próbkowanie szablonu z=h(0)
        z_0 = self.reparameterize(mu, logvar)

        # Dodajemy wektor początkowej prędkości (wyzerowany) i łączymy ze stanem
        v_0 = torch.zeros_like(z_0)
        state_0 = torch.cat([z_0, v_0], dim=-1)

        # Odcałkowanie w czasie za pomocą ODE (wchodzi rozszerzony stan)
        state_t = odeint(self.ode_func, state_0, t_seq, method='dopri5')

        # Rozcinamy z powrotem, do dekodera idzie tylko stan
        h_t, _ = torch.chunk(state_t, 2, dim=-1)

        # Dekodowanie
        raw_reconstructed = self.decoder(h_t)

        # Zabezpieczamy sieć przed wygenerowaniem np. ujemnego ICP lub tętna.
        reconstructed_vitals = torch.clamp(raw_reconstructed, min=0.0)

        # Predykcja ryzyka
        risk_trajectory = self.risk_head(h_t).squeeze(-1)

        reconstructed_vitals = reconstructed_vitals.permute(1, 0, 2)
        risk_trajectory = risk_trajectory.permute(1, 0)

        # Obliczenie KLD. Zmusza przestrzeń ukrytą do przybrania kształtu idealnego rozkładu normalnego.
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return reconstructed_vitals, risk_trajectory, kld_loss

    def generate_synthetic_patients(self, base_vitals_seq, t_seq, num_samples=5):
        """ Generowanie z użyciem przestrzeni VAE. Szum jest nadzorowany przez wyuczoną
            wariancję danego pacjenta, a nie zgadywany w ciemno. """
        self.eval()
        with torch.no_grad():
            _, h_rnn = self.encoder_rnn(base_vitals_seq)
            h_rnn = h_rnn.squeeze(0)

            mu = self.fc_mu(h_rnn)
            logvar = self.fc_logvar(h_rnn)

            synthetic_patients = []
            for _ in range(num_samples):
                z_0 = self.reparameterize(mu, logvar)

                # Uwzględniamy prędkość początkową w SONODE, żeby model mógł uchwycić przyspieszające pogorszenie stanu pacjenta
                v_0 = torch.zeros_like(z_0)
                state_0 = torch.cat([z_0, v_0], dim=-1)

                state_t = odeint(self.ode_func, state_0, t_seq, method='dopri5')
                h_t, _ = torch.chunk(state_t, 2, dim=-1)

                raw_vitals = self.decoder(h_t)
                synthetic_vitals = torch.clamp(raw_vitals, min=0.0)
                synthetic_vitals = synthetic_vitals.permute(1, 0, 2).squeeze(0)

                synthetic_patients.append(synthetic_vitals)

        return torch.stack(synthetic_patients)

    def generate_counterfactual_augmentation(self, base_vitals_seq, t_seq, original_treatments, target_treatment_idx, time_of_intervention_idx):
        """ Tworzy kontrfaktyczną trajektorię pacjenta poprzez modyfikację interwencji.
            Zwraca zrekonstruowane parametry życiowe jako symulację 'co by było gdyby'. """
        self.eval()
        with torch.no_grad():
            # Kodowanie aktualnej historii pacjenta do punktu startowego
            _, h_rnn = self.encoder_rnn(base_vitals_seq)
            h_rnn = h_rnn.squeeze(0)

            mu = self.fc_mu(h_rnn)
            logvar = self.fc_logvar(h_rnn)
            z_0 = self.reparameterize(mu, logvar)

            # Stan początkowy (pozycja + początkowa prędkość dla SONODE)
            v_0 = torch.zeros_like(z_0)
            state_0 = torch.cat([z_0, v_0], dim=-1)

            # Kopiujemy oryginalne leczenie i zmieniamy w wybranym momencie
            cf_treatments = original_treatments.clone()
            current_action = cf_treatments[:, time_of_intervention_idx:, target_treatment_idx]
            cf_treatments[:, time_of_intervention_idx:, target_treatment_idx] = 1.0 - current_action

            # Dodajemy kontrfaktyczne leczenie do funkcji ODE, aby trajektoria uwzględniała zmienione interwencje
            self.ode_func.set_treatments(cf_treatments, t_seq)
            state_t = odeint(self.ode_func, state_0, t_seq, method='dopri5')
            self.ode_func.treatments = None
            self.ode_func.treatment_times = None
            h_t, _ = torch.chunk(state_t, 2, dim=-1)

            # Zdekodowanie ukrytych wektorów z powrotem na parametry medyczne
            raw_vitals = self.decoder(h_t)
            synthetic_vitals = torch.clamp(raw_vitals, min=0.0)
            synthetic_vitals = synthetic_vitals.permute(1, 0, 2).squeeze(0)

        return synthetic_vitals, cf_treatments


class LatentODE_PatientGenerator(nn.Module):
    """ Generatywny Model Neural ODE. Działa jako Enkoder-Dekoder oparty na równaniach różniczkowych. """
    def __init__(self, input_dim=22, hidden_dim=16, treatment_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.ode_func = PatientStateSONODE(hidden_dim, treatment_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, vitals_seq, t_seq):
        """ Przepływ treningowy (rekonstrukcja i predykcja) z obsługą dynamiki 2. rzędu SONODE. """
        _, h_0 = self.encoder(vitals_seq)
        h_0 = h_0.squeeze(0)

        # Dopasowanie wymiaru do 2. rzędu SONODE (stan + początkowa prędkość)
        v_0 = torch.zeros_like(h_0)
        state_0 = torch.cat([h_0, v_0], dim=-1)

        state_t = odeint(self.ode_func, state_0, t_seq, method='dopri5')
        h_t, _ = torch.chunk(state_t, 2, dim=-1)

        reconstructed_vitals = self.decoder(h_t)
        risk_trajectory = self.risk_head(h_t).squeeze(-1)

        reconstructed_vitals = reconstructed_vitals.permute(1, 0, 2)
        risk_trajectory = risk_trajectory.permute(1, 0)

        return reconstructed_vitals, risk_trajectory

    def generate_synthetic_patients_via_interpolation(self, tier1_vitals, tier2_vitals, t_seq, num_samples=5, mix_ratio=0.7):
        """ Kliniczna Interpolacja w Przestrzeni Ukrytej ODE ze wsparciem dynamiki SONODE. """
        self.eval()
        with torch.no_grad():
            _, h_tier2 = self.encoder(tier2_vitals)
            h_tier2 = h_tier2.squeeze(0)

            critical_center = torch.mean(h_tier2, dim=0, keepdim=True)

            _, h_tier1 = self.encoder(tier1_vitals)
            h_tier1 = h_tier1.squeeze(0)

            synthetic_patients = []

            for _ in range(num_samples):
                idx = torch.randint(0, h_tier1.size(0), (1,)).item()
                base_h = h_tier1[idx].unsqueeze(0)

                interpolated_h = (1.0 - mix_ratio) * base_h + mix_ratio * critical_center
                noisy_h = interpolated_h + torch.randn_like(interpolated_h) * 0.05

                # Przygotowanie stanu 2. rzędu dla SONODE
                v_0 = torch.zeros_like(noisy_h)
                state_0 = torch.cat([noisy_h, v_0], dim=-1)

                state_t = odeint(self.ode_func, state_0, t_seq, method='dopri5')
                h_t, _ = torch.chunk(state_t, 2, dim=-1)

                raw_vitals = self.decoder(h_t)
                synthetic_vitals = torch.clamp(raw_vitals, min=0.0)
                synthetic_vitals = synthetic_vitals.permute(1, 0, 2).squeeze(0)

                synthetic_patients.append(synthetic_vitals)

        return torch.stack(synthetic_patients)


def calculate_hierarchical_involution_modifier(
    unet_brain_mask: Optional[torch.Tensor] = None,
    unet_icv_mask: Optional[torch.Tensor] = None,
    brain_slice_2d: Optional[np.ndarray] = None,
    patient_age: Optional[float] = None,
    default_age: float = 50.0
) -> Tuple[float, Dict[str, Any]]:
    """ Hierarchiczny kalkulator modyfikatora inwolucyjnego (rezerwy kompensacyjnej czaszki).
        Strategia:
        Poziom 1: Objętość 3D z UNet (Brain Parenchyma Fraction - BPF).
        Poziom 2: Indeks Evansa na przekroju komorowym.
        Poziom 3: Płynna funkcja sigmoidalna wieku pacjenta. """
    metadata = {
        "tier_used": None,
        "method_description": "",
        "confidence_score": 0.0,
        "raw_metric_value": None
    }

    # Najpierw sprawdź objętość 3D z segmentacji UNet
    if unet_brain_mask is not None and unet_icv_mask is not None:
        v_brain = float(torch.sum(unet_brain_mask > 0.5).item())
        v_icv = float(torch.sum(unet_icv_mask > 0.5).item())

        if v_icv > 0 and v_brain > 0:
            bpf = v_brain / v_icv
            reserve_ratio = 1.0 - bpf
            mod_3d = float(np.clip(0.5 + (reserve_ratio * 3.0), 0.6, 1.8))

            metadata["tier_used"] = "Tier_1_Volumetric_3D"
            metadata["method_description"] = f"Pełna wolumetria 3D UNet (BPF={bpf:.3f})"
            metadata["confidence_score"] = 0.98
            metadata["raw_metric_value"] = bpf
            return mod_3d, metadata

    # Biometria 2D na pojedynczym przekroju (Indeks Evansa)
    if brain_slice_2d is not None and brain_slice_2d.size > 0:
        try:
            icv_pixels = np.where(brain_slice_2d > 20)
            ventricles_pixels = np.where((brain_slice_2d > 5) & (brain_slice_2d < 15))

            if len(icv_pixels[1]) > 0 and len(ventricles_pixels[1]) > 0:
                icv_width = np.max(icv_pixels[1]) - np.min(icv_pixels[1])
                ventricles_width = np.max(ventricles_pixels[1]) - np.min(ventricles_pixels[1])

                if icv_width > 0:
                    evans_index = ventricles_width / icv_width
                    mod_2d = float(np.clip(0.6 + (evans_index * 2.5), 0.6, 1.6))

                    metadata["tier_used"] = "Tier_2_Biometrics_2D"
                    metadata["method_description"] = f"Indeks Evansa z przekroju 2D (EI={evans_index:.3f})"
                    metadata["confidence_score"] = 0.82
                    metadata["raw_metric_value"] = evans_index
                    return mod_2d, metadata
        except Exception as e:
            logger.debug(f"Pominięto Poziom 2 z powodu błędu obróbki obrazu 2D: {e}")

    # Płynna funkcja sigmoidalna wieku
    target_age = patient_age if patient_age is not None else default_age

    min_mod, max_mod = 0.7, 1.5
    sigmoid_val = 1.0 / (1.0 + np.exp(-(target_age - 65.0) / 5.0))
    mod_age = float(min_mod + (max_mod - min_mod) * sigmoid_val)

    metadata["tier_used"] = "Tier_3_Age_Sigmoid_Fallback"
    metadata["method_description"] = f"Płynny modyfikator wiekowy (Wiek={target_age:.0f} lat)"
    metadata["confidence_score"] = 0.50 if patient_age is not None else 0.20
    metadata["raw_metric_value"] = target_age

    return mod_age, metadata


class BayesianTrajectoryMonitor:
    """ Moduł Longitudinalnego Monitorowania Pacjenta wzbogacony o zanikanie alarmów, kontekst sedacji farmakologicznej
        oraz nieliniowe kary za wgłobienie bazujące na hierarchicznej rezerwie inwolucyjnej. """
    def __init__(self, initial_prob: float, patient_age: int = 50):
        self.patient_age = patient_age
        self.current_prob = max(min(initial_prob, 0.99), 0.01)
        self.current_odds = self.current_prob / (1 - self.current_prob)
        self.history = [{
            "time": datetime.now(),
            "prob": self.current_prob,
            "event": "Admission (Cascade Baseline)"
        }]
        self.last_update_time = datetime.now()

    def update_belief(
        self,
        new_vitals: dict,
        current_time: Optional[datetime] = None,
        unet_brain_mask: Optional[torch.Tensor] = None,
        unet_icv_mask: Optional[torch.Tensor] = None,
        brain_slice_2d: Optional[np.ndarray] = None
    ) -> float:
        if current_time is None:
            current_time = datetime.now()

        likelihood_ratio = 1.0
        event_notes = []

        # Zanikanie czasowe
        days_passed = (current_time - self.last_update_time).total_seconds() / 86400.0
        
        # Dobowy współczynnik zanikania (np. 0.90 lub 0.95 na dobę stabilnego stanu)
        decay_factor = 0.95 ** days_passed
        self.current_odds *= decay_factor

        # Flaga śpiączki farmakologicznej
        is_sedated = new_vitals.get('sedatives_administered', False)

        # Płynna zmiana GCS
        if 'delta_gcs' in new_vitals:
            drop = new_vitals['delta_gcs']
            if drop < 0:
                base_penalty = 1.0 + (abs(drop) ** 1.5) * 0.5

                if is_sedated:
                    likelihood_ratio *= (1.0 + (base_penalty - 1.0) * 0.2)
                    event_notes.append(f"GCS {drop} (Sedacja, zmniejszona kara: LR: {(1.0 + (base_penalty - 1.0) * 0.2):.2f}x)")
                else:
                    likelihood_ratio *= base_penalty
                    event_notes.append(f"KRYTYCZNE GCS {drop} (LR: {base_penalty:.2f}x)")
            elif drop > 0:
                reward = 0.8 ** drop
                likelihood_ratio *= reward
                event_notes.append(f"GCS + {drop} (LR: {reward:.2f}x)")

        # Ciśnienie wewnątrzczaszkowe z hierarchiczną oceną rezerwy (Graceful Degradation)
        if 'icp_current' in new_vitals:
            icp = new_vitals['icp_current']
            if icp > 15.0:
                excess_icp = icp - 15.0

                # Wyznaczenie modyfikatora inwolucyjnego (Tier 1 -> Tier 2 -> Tier 3)
                involution_mod, meta = calculate_hierarchical_involution_modifier(
                    unet_brain_mask=unet_brain_mask,
                    unet_icv_mask=unet_icv_mask,
                    brain_slice_2d=brain_slice_2d,
                    patient_age=self.patient_age
                )

                # Nieliniowy mnożnik skoku ICP uwzględniający wyznaczoną rezerwę
                icp_multiplier = 1.0 + ((excess_icp * 0.2) ** 2) * involution_mod
                likelihood_ratio *= icp_multiplier
                event_notes.append(f"ICP {icp} mmHg (LR: {icp_multiplier:.2f}x, Metoda: {meta['tier_used']})")

        # Brak złych wieści i brak sedacji to dobra wiadomość
        if likelihood_ratio == 1.0 and not is_sedated:
            likelihood_ratio = 0.95
            event_notes.append("Stabilny stan (LR: 0.95x)")

        # Aktualizacja Szans i konwersja na Prawdopodobieństwo
        self.current_odds *= likelihood_ratio
        self.current_prob = self.current_odds / (1 + self.current_odds)
        self.last_update_time = current_time

        self.history.append({
            "time": current_time,
            "prob": self.current_prob,
            "event": " | ".join(event_notes)
        })

        return self.current_prob


class WeaningReadinessEvaluator(nn.Module):
    """ Moduł oceniający prawdopodobieństwo udanej ekstubacji (odłączenia od respiratora).
        Zaprojektowany do uczenia na rozszerzonych danych oddechowych ze zbioru eICU.
        Wymaga pełnego profilu oddechowo-hemodynamicznego:
        1-5. Baza: Wiek, GCS, Ciśnienie, Tętno, Wskaźnik Oksygenacji (SpO2/FiO2)
        6. RSBI (Rapid Shallow Breathing Index) - stosunek częstości oddechów do objętości oddechowej
        7-9. Gazometria: pH krwi, PaO2, PaCO2 (krytyczne dla zapobiegania obrzękowi mózgu)
        10-11. Parametry respiratora: PEEP (dodatnie ciśnienie końcowowydechowe), PIP (szczytowe ciśnienie)
        12. RASS (Poziom sedacji/pobudzenia) - pacjent musi współpracować. """
    def __init__(self, input_dim=12):
        super(WeaningReadinessEvaluator, self).__init__()
        self.evaluator = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3), # Zwiększony dropout ze względu na większą liczbę cech
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Zwraca prawdopodobieństwo (0.0 - 1.0) udanej próby oddechu własnego
        )

    def forward(self, x):
        return self.evaluator(x)


def prepare_real_icu_tensors(df: pd.DataFrame, max_seq_len: int = 24, default_population_values: Optional[dict] = None):
    """ Przetwarza surowe dane z MIMIC-IV:
        - Rozdziela parametry kliniczne na niezależne kanały (np. ICP, GCS).
        - Zachowuje wielowymiarowy tensor cech [Batch, SeqLen, InputDim].
        - Wykonuje niezależną interpolację i obsługę braków danych dla każdego kanału bez wycieku z przyszłości,
          wykorzystując średnie populacyjne. """
    df = df.copy()  # Bezpieczna kopia chroniąca przed modyfikacją oryginalnego DataFrame
    # Mapowanie MIMIC itemid na nazwy kliniczne
    item_mapping = {
        220045: 'icp',
        226537: 'icp',
        220739: 'gcs_total',
        223900: 'gcs_eye',
        223901: 'gcs_motor',
        223902: 'gcs_verbal'
    }

    df['clinical_feature'] = df['itemid'].map(item_mapping)
    
    # Ustalamy stałą listę cech wejściowych
    feature_columns = ['icp', 'gcs_total']
    
    # Jeśli nie podano słownika z zewnątrz, obliczamy średnie populacyjne dynamicznie z danych
    if default_population_values is None:
        default_population_values = {}
        for col in feature_columns:
            subset_vals = df[df['clinical_feature'] == col]['valuenum']
            if not subset_vals.empty and subset_vals.notnull().sum() > 0:
                default_population_values[col] = float(subset_vals.mean())
            else:
                # Awaryjne wartości domyślne (fallback medyczny), gdyby kolumna była pusta
                default_population_values[col] = 10.0 if col == 'icp' else 13.0

    df = df.dropna(subset=['clinical_feature'])

    batch_features = []
    batch_deltas = []

    grouped = df.groupby('stay_id')
    input_dim = len(feature_columns)

    for stay_id, group in grouped:
        group = group.sort_values('hours_from_admission')

        if len(group) < 2:
            continue

        """ Okno czasowe, żeby pogrupować dane po dobach, analiza trajektorii w skali godzinowej może być zbyt szczegółowa i podatna na szum. 
            Wydaje mi się, że w kontekście monitorowania pacjentów w ICU, agregacja po dobach może lepiej uchwycić ogólne trendy kliniczne. """
        group = group.copy()
        group['day_bin'] = (group['hours_from_admission'] // 24).astype(int)

        # Pivot table grupujący po dobach zamiast godzin:
        pivot_df = group.pivot_table(
            index='day_bin',
            columns='clinical_feature',
            values='valuenum',
            aggfunc='mean'  # Średnia lub max z całej doby
        )

        # Upewnienie się, że wszystkie wymagane kolumny istnieją w pivot_df
        for col in feature_columns:
            if col not in pivot_df.columns:
                pivot_df[col] = np.nan

        pivot_df = pivot_df[feature_columns]

        # Ograniczenie do max_seq_len godzin
        if len(pivot_df) > max_seq_len:
            pivot_df = pivot_df.head(max_seq_len)

        # Forward fill dla trajektorii, a następnie uzupełnienie braków początkowych średnimi populacyjnymi
        pivot_df = pivot_df.ffill()
        for col, mean_val in default_population_values.items():
            if col in pivot_df.columns:
                pivot_df[col].fillna(mean_val, inplace=True)
        pivot_df = pivot_df.fillna(0.0)

        features_matrix = pivot_df.values # Kształt: [Liczba rzeczywistych kroków, input_dim]
        hours = pivot_df.index.values

        # Obliczenie rzeczywistych odstępów czasu dt dla każdego kroku
        dt = np.diff(hours, prepend=hours[0])
        dt[dt <= 0] = 1.0 # Zabezpieczenie przed dt=0 przy wielu pomiarach w tej samej godzinie

        # Padding sekwencji do stałej długości
        current_len = len(features_matrix)
        if current_len < max_seq_len:
            pad_len = max_seq_len - current_len

            # Padding zerami dla cech, padding stałym krokiem czasowym dla dt
            features_matrix = np.pad(features_matrix, ((0, pad_len), (0, 0)), 'constant', constant_values=0.0)
            dt = np.pad(dt, (0, pad_len), 'constant', constant_values=1.0)
        elif current_len > max_seq_len:
            features_matrix = features_matrix[:max_seq_len]
            dt = dt[:max_seq_len]

        batch_features.append(features_matrix)
        batch_deltas.append(dt)

    if not batch_features:
        raise ValueError("Brak poprawnych trajektorii po przetworzeniu.")

    # Konwersja na tensory o wymiarach wymaganych przez model wielowymiarowy
    X_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32)
    Dt_tensor = torch.tensor(np.array(batch_deltas), dtype=torch.float32)

    return X_tensor, Dt_tensor


class ODEFunc(nn.Module):
    """ Definiuje pochodną stanu ukrytego w czasie: dh/dt = f(h(t), t) """
    def __init__(self, hidden_dim: int):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t, h):
        return self.net(h)


class NeuralODE_ICU_Monitor(nn.Module):
    """ Ciągły model monitorowania trajektorii stanu (ciągła ewolucja intencji lub wstrząsów
        w nieregularnych odstępach czasu) z użyciem mechanizmów Neural ODE. """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super(NeuralODE_ICU_Monitor, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence_features, time_deltas):
        batch_size, seq_len, _ = sequence_features.shape
        device = sequence_features.device
        h = torch.zeros(batch_size, self.ode_func.net[0].in_features, device=device)

        trajectory = []
        for i in range(seq_len):
            x_t = sequence_features[:, i, :]
            dt = time_deltas[:, i].unsqueeze(-1)

            # Aktualizacja stanu wejściowego w danym kroku
            h = h + self.input_projection(x_t)

            # Ciągła ewolucja trajektorii o zadany interwał czasowy dt
            deriv = self.ode_func(0.0, h)
            h = h + deriv * dt
            trajectory.append(torch.sigmoid(self.output_layer(h)))

        # Zwracamy pełną trajektorię w czasie [batch_size, seq_len, output_dim]
        return torch.stack(trajectory, dim=1)


class RealTimePatientBuffer:
    """ Zarządza stanem pacjenta w czasie rzeczywistym przy użyciu Redisa. Pozwala na asynchroniczne
        dopisywanie kolejnych danych i sklejanie ich w kompletny wektor cech, zanim model podejmie decyzję. """
    def __init__(self, host='localhost', port=6379, db=0):
        # Inicjalizacja połączenia z bazą Redis
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def update_patient_data(self, hadm_id: int, category: str, data: dict, ttl: int = 86400):
        """ Aktualizuje lub dodaje wybraną kategorię danych dla pacjenta. """
        key = f"patient:{hadm_id}:{category}"
        self.r.setex(key, ttl, json.dumps(data))

    def get_complete_patient_vector(self, hadm_id: int) -> dict:
        """ Skleja kompletny stan pacjenta ze wszystkich dostępnych podsystemów w locie.
            Jeśli brakuje konkretnego badania, funkcja nie przerywa działania, tylko zwraca
            aktualnie dostępny podzbiór cech do bezpiecznego uzupełnienia przez imputację. """
        categories = ['vitals', 'vision']
        combined_vector = {}

        for cat in categories:
            key = f"patient:{hadm_id}:{cat}"
            raw_data = self.r.get(key)
            if raw_data:
                try:
                    parsed = json.loads(raw_data)
                    combined_vector.update(parsed)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.error(f"Nieoczekiwany błąd podczas parsowania danych pacjenta {hadm_id} dla kategorii {cat}: {e}")

        return combined_vector

    def is_ready_for_inference(self, hadm_id: int, required_categories=None) -> bool:
        """ Sprawdza, czy pacjent zgromadził minimalny zestaw danych wymagany do uruchomienia pierwszej warstwy kaskady. """
        if required_categories is None:
            required_categories = ['vitals']

        for cat in required_categories:
            if not self.r.exists(f"patient:{hadm_id}:{cat}"):
                return False
        return True



def generate_icu_clinical_scenario(
    num_patients: Optional[int] = None,
    seq_length: Optional[int] = None,
    seed: int = 42,
    scenario_name: str = "sudden_hematoma",
    duration_days: int = 14
):
    """ Uniwersalna funkcja obsługująca zarówno generowanie wsadowe (batch),
        jak i pojedynczy scenariusz kliniczny dla strumienia OIT. """
    # Tryb wsadowy (batch) - gdy podano num_patients lub seq_length w sposób domyślny/jawny
    if num_patients is not None or (seq_length is not None and "scenario_name" not in locals()):
        num_patients = num_patients or 100
        seq_length = seq_length or 24
        np.random.seed(seed)

        ages = np.random.normal(45, 15, num_patients).clip(18, 90).astype(int)
        gcs_scores = np.random.randint(3, 16, num_patients)
        pupils = np.random.choice(['Both Reactive', 'One Reactive', 'Non-reactive'], num_patients, p=[0.7, 0.2, 0.1])

        df_tabular = pd.DataFrame({
            'PatientID': [f'ICU_{i:04d}' for i in range(num_patients)],
            'Age': ages,
            'Admission_GCS': gcs_scores,
            'Pupil_Reactivity': pupils,
            'Hypotension_Flag': np.random.choice([0, 1], num_patients, p=[0.8, 0.2])
        })

        time_series = np.zeros((num_patients, seq_length, 4))
        clinical_notes = []

        for i in range(num_patients):
            base_icp = np.random.normal(12, 3) if gcs_scores[i] > 8 else np.random.normal(22, 5)
            base_map = np.random.normal(85, 10)

            icp_trend = base_icp + np.cumsum(np.random.normal(0, 1.5, seq_length))
            map_trend = base_map + np.cumsum(np.random.normal(0, 2.0, seq_length))
            hr_trend = np.random.normal(75, 15) + np.cumsum(np.random.normal(0, 1.0, seq_length))
            spo2_trend = np.clip(np.random.normal(96, 2) + np.cumsum(np.random.normal(0, 0.5, seq_length)), 80, 100)

            time_series[i, :, 0] = np.clip(icp_trend, 0, 50)
            time_series[i, :, 1] = np.clip(map_trend, 40, 130)
            time_series[i, :, 2] = np.clip(hr_trend, 40, 180)
            time_series[i, :, 3] = spo2_trend

        return df_tabular, torch.tensor(time_series, dtype=torch.float32), clinical_notes

    # Tryb pojedynczego scenariusza strumieniowego (dla serve_patient_stream)
    else:
        np.random.seed(seed)
        events = []
        for day in range(1, duration_days + 1):
            if scenario_name == "sudden_hematoma":
                delta_gcs = -1 if day in [3, 6, 9] else 0
                icp_current = 12.0 + (day * 1.5 if day > 4 else day * 0.2)
            else:
                delta_gcs = 0
                icp_current = 12.0 + np.random.normal(0, 0.5)

            events.append({
                "delta_gcs": delta_gcs,
                "icp_current": float(icp_current),
                "sedatives_administered": False,
            })
        return events


def serve_patient_stream(calibrated_spec_pipe, X_test, ode_monitor=None, weaning_model=None, redis_host="localhost"):
    """ Symuluje strumieniowy napływ danych pacjenta z bazy Redis w czasie rzeczywistym.
        Porównuje trajektorię Bayesowską z ciągłą, odcałkowaną (Neural ODE). """
    logger = logging.getLogger(__name__)
    logger.info(
        "Pokaz działania na żywo: Inicjalna Kaskada (SOR) -> Monitorowanie (OIT)"
    )
    buffer = RealTimePatientBuffer(host=redis_host, port=6379)

    # Bezpieczna, granularna inicjalizacja klienta Redis dla alertów
    r_cache = None
    try:
        r_cache = redis.Redis(
            host=redis_host,
            port=6379,
            decode_responses=True,
            socket_timeout=2.0,
            socket_connect_timeout=2.0
        )
        r_cache.ping()
    except redis.AuthenticationError as e:
        logger.error(f"Błąd uwierzytelniania w bazie Redis (sprawdź poświadczenia): {e}")
        r_cache = None
    except (redis.ConnectionError, redis.TimeoutError) as e:
        logger.warning(f"Serwer Redis na hoście '{redis_host}' jest niedostępny lub przeciążony: {e}")
        r_cache = None
    except (socket.gaierror, OSError) as e:
        logger.warning(f"Problem z siecią lub rozwiązaniem adresu hosta Redis ({redis_host}): {e}")
        r_cache = None
    except redis.RedisError as e:
        logger.error(f"Nieznany błąd biblioteki Redis podczas inicjalizacji: {e}")
        r_cache = None
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas inicjalizacji klienta Redis: {e}")
        r_cache = None

    current_patient_id = 1
    buffer.update_patient_data(
        hadm_id=current_patient_id,
        category="vitals",
        data={"gcs_sum": 13, "sysbp_min": 120},
    )

    if buffer.is_ready_for_inference(current_patient_id):
        # Triage na SORze
        X_live_patient = pd.DataFrame([X_test.iloc[0].to_dict()])
        _, probs = calibrated_spec_pipe.predict_with_clinical_review(X_live_patient)

        initial_prob = float(probs[0])
        logger.info(
            f"Oceniono pacjenta {current_patient_id} na wejściu. Prawdopodobieństwo"
            f" krytyczne: {initial_prob*100:.1f}%"
        )

        logger.info(
            "Uruchamianie równoległego monitoringu: Bayesian Tracking vs Neural ODE"
            " Continuous Tracking."
        )

        # Wyciągamy wiek dla monitoringu
        p_age = (
            int(X_live_patient["age"].iloc[0])
            if "age" in X_live_patient.columns
            else 50
        )
        bayesian_monitor = BayesianTrajectoryMonitor(
            initial_prob, patient_age=p_age
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sprawdzenie obecności Neural ODE / torchdiffeq
        if odeint is None:
            raise RuntimeError("Pakiet 'torchdiffeq' (odeint) jest wymagany do uruchomienia Neural ODE.")
        if ode_monitor is None:
            raise RuntimeError("Model Neural ODE ('ode_monitor') nie został przekazany lub wytrenowany.")

        ode_monitor.eval()

        # Przykładowy scenariusz
        chosen_scenario = "sudden_hematoma"
        simulated_events = generate_icu_clinical_scenario(
            scenario_name=chosen_scenario, duration_hours=18
        )

        bayesian_probs_over_time = [initial_prob]

        # Pobieramy realne wartości początkowe z danych pacjenta na wejściu (zamiast sztywnych liczb)
        initial_gcs = (
            float(X_live_patient["gcs_sum"].iloc[0])
            if "gcs_sum" in X_live_patient.columns and pd.notnull(X_live_patient["gcs_sum"].iloc[0])
            else 13.0
        )
        initial_icp = (
            float(X_live_patient["icp_max"].iloc[0])
            if "icp_max" in X_live_patient.columns and pd.notnull(X_live_patient["icp_max"].iloc[0])
            else 12.0
        )

        # Zmienne do śledzenia ciągłej trajektorii ODE oparte na faktycznych danych pacjenta
        time_points = [0.0]
        cumulative_gcs = initial_gcs
        vitals_history = [[float(cumulative_gcs), initial_icp]]

        # Symulacja czasu bazująca na faktycznym kroku czasowym (timedelta)
        simulation_start = bayesian_monitor.last_update_time

        for hour, vitals in enumerate(simulated_events, 1):
            current_time = simulation_start + timedelta(hours=hour)

            # Ścieżka Bayesowska uwzględniająca upływający czas
            new_prob = bayesian_monitor.update_belief(vitals, current_time=current_time)
            bayesian_probs_over_time.append(new_prob)
            logger.info(
                f"Godzina {hour} OIT | Zmiany: {vitals} | Zaktualizowane Ryzyko:"
                f" {new_prob*100:.1f}%"
            )

            if new_prob > 0.85 and r_cache is not None:
                try:
                    r_cache.setex(
                        f"alert:patient:{current_patient_id}:urgent",
                        300,
                        "CODE RED: Wgłobienie/Krwotok w toku!",
                    )
                except redis.RedisError as e:
                    logger.warning(f"Nie udało się zapisać alertu w Redis: {e}")
                logger.warning("SYSTEM WYGENEROWAŁ NOWY, DYNAMICZNY ALERT RATUNKOWY!")

            # Gromadzenie danych dla ścieżki ODE
            cumulative_gcs += vitals.get("delta_gcs", 0)
            icp = vitals.get("icp_current", 12.0)
            time_points.append(float(hour))
            vitals_history.append([float(cumulative_gcs), float(icp)])

        # Ścieżka i kształt Neural ODE (Ciągła całka)
        t_seq = torch.tensor(time_points, dtype=torch.float32).to(device)

        # Wyznaczenie delty czasu dt i poprawny wymiar batcha [1, seq_len]
        dt_seq = torch.diff(t_seq, prepend=torch.tensor([0.0]).to(device)).unsqueeze(0)

        # Tensor wejściowy w formacie [batch_size=1, seq_len, input_dim=2]
        v_seq = torch.tensor(vitals_history, dtype=torch.float32).unsqueeze(0).to(device)

        ode_probs_over_time = []
        with torch.no_grad():
            ode_probs = ode_monitor(v_seq, dt_seq).cpu().numpy().squeeze()
            if ode_probs.ndim == 0:
                ode_probs = np.array([ode_probs])

            base_shift = initial_prob - ode_probs[0]
            ode_probs_over_time = np.clip(ode_probs + base_shift, 0, 1)

        # Wykres Trajektorii Pacjenta
        plt.figure(figsize=(10, 5))
        plt.plot(
            range(len(bayesian_probs_over_time)),
            [p * 100 for p in bayesian_probs_over_time],
            marker="o",
            linestyle="-",
            color="red" if chosen_scenario == "sudden_hematoma" else "green",
            label="Bayesian Discrete Tracking",
        )

        plt.plot(
            time_points,
            [p * 100 for p in ode_probs_over_time],
            linestyle="-.",
            color="purple",
            linewidth=2.5,
            label="Neural ODE Continuous Flow",
        )

        plt.axhline(y=85, color="darkred", linestyle=":", label="Próg Alarmu Krytycznego")
        plt.title(
            f"Trajektoria Pacjenta (Bayes vs Neural ODE) - Scenariusz:"
            f" {chosen_scenario}"
        )
        plt.xlabel("Godziny po przyjęciu na OIT")
        plt.ylabel("Prawdopodobieństwo Zgonu/Wgłobienia (%)")
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs(CONFIG["PLOTS_DIR"], exist_ok=True)
        plt.savefig(
            f'{CONFIG["PLOTS_DIR"]}/bayesian_vs_ode_{chosen_scenario}.png', dpi=300
        )
        plt.close()

        # Logistyka i Respirator (dla zaintubowanych)
        is_intubated = X_live_patient.get("is_intubated", pd.Series([0])).iloc[0]

        if is_intubated == 1:
            logger.info("Faza 9: Ocena gotowości do odłączenia od respiratora (Weaning Prediction).")

            sf_ratio = (
                X_live_patient.get("sf_ratio", pd.Series([250.0])).iloc[0]
            )
            gcs = X_live_patient.get("gcs_sum", pd.Series([8])).iloc[0]
            map_bp = X_live_patient.get("map_min", pd.Series([75])).iloc[0]
            hr = X_live_patient.get("heart_rate_max", pd.Series([90])).iloc[0]
            age = X_live_patient.get("age", pd.Series([50])).iloc[0]

            weaning_features = torch.tensor(
                [[sf_ratio / 500.0, gcs / 15.0, map_bp / 120.0, hr / 150.0, age / 100.0]],
                dtype=torch.float32,
            ).to(device)

            if weaning_model is None:
                weaning_model = WeaningReadinessEvaluator(input_dim=5).to(device)
            weaning_model.eval()

            with torch.no_grad():
                weaning_prob = weaning_model(weaning_features).item()

            logger.info(
                f"Oszacowane prawdopodobieństwo pomyślnej ekstubacji:"
                f" {weaning_prob*100:.1f}%"
            )

            if weaning_prob > 0.80:
                logger.info(
                    "SUGEROWANA INTERWENCJA (HITL): Parametry oddechowe ustabilizowane."
                    " Rozważ próbę oddechu własnego (SBT). Ostateczna decyzja należy do"
                    " Intensywisty."
                )
            else:
                logger.info(
                    "SUGEROWANA INTERWENCJA: Parametry zbyt niskie na odłączenie"
                    " maszyny. Kontynuuj wentylację mechaniczną."
                )


def extract_mimic_timeseries_3d(filepath, item_ids, admittimes_dict, max_days=14, default_population_values=None):
    """ Przetwarza surowe logi MIMIC-IV na trójwymiarowy tensor szeregów czasowych w cyklu dobowym
        [Liczba_Pacjentów, Max_Dni, Liczba_Cech]. Wykorzystuje Forward Fill i wartości populacyjne bez wycieku. """
    logger.info(f"Budowanie dobowych szeregów czasowych 3D z logów MIMIC-IV ({max_days} dni okno).")

    # Inicjalizacja słownika list dla każdego pacjenta
    patient_trajectories = {}
    item_id_list = list(item_ids.values())
    flat_item_ids = [item for sublist in item_id_list for item in sublist]

    chunk_iterator = pd.read_csv(filepath, compression='gzip', chunksize=2000000,
                                 usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'], parse_dates=['charttime'])

    for chunk in tqdm(chunk_iterator, desc="Ekstrakcja dobowa z 30 GB chartevents", leave=False):
        filtered = chunk[chunk['itemid'].isin(flat_item_ids)].copy()
        if filtered.empty:
            continue

        # Mapowanie itemid na ujednolicone nazwy cech
        reverse_map = {vid: k for k, vlist in item_ids.items() for vid in vlist}
        filtered['feature_name'] = filtered['itemid'].map(reverse_map)

        # Odrzucamy pacjentów, dla których nie mamy czasu przyjęcia
        filtered = filtered[filtered['hadm_id'].isin(admittimes_dict.keys())].copy()
        filtered['admittime'] = filtered['hadm_id'].map(admittimes_dict)

        """ Wyliczamy, w której dobie od przyjęcia nastąpił pomiar (86400 sekund = 24h). Podział na doby jest kluczowy,
            żeby jakoś walczyć z szumem dla pomiarów ciągłych. Nimi będzie musiał zająć się lekarz. """
        filtered['day'] = ((filtered['charttime'] - filtered['admittime']).dt.total_seconds() // 86400).astype(int)

        """ Najbardziej krytyczna faza wstrząsu, obrzęku mózgu, skoków ciśnienia wewnątrzczaszkowego oraz największego ryzyka zgonu
            lub nieodwracalnych powikłań mieści się zazwyczaj w pierwszych 7–14 dobach pobytu na OIT. To wtedy ważą się losy
            pacjenta i podejmowane są najważniejsze interwencje farmakologiczne. """
        valid_time = filtered[(filtered['day'] >= 0) & (filtered['day'] < max_days)]

        for _, row in valid_time.iterrows():
            hid = row['hadm_id']
            if hid not in patient_trajectories:
                # Inicjalizacja pustej macierzy dla pacjenta: [dni, cechy]
                patient_trajectories[hid] = np.full((max_days, len(item_ids)), np.nan)

            feat_idx = list(item_ids.keys()).index(row['feature_name'])

            # Wpisujemy wartość do odpowiedniej doby. Jeśli jest kilka pomiarów w tej samej dobie, nadpisujemy najnowszy
            patient_trajectories[hid][row['day'], feat_idx] = row['valuenum']

    logger.info("Imputacja sekwencyjna (Forward Fill -> Wartość populacyjna bez wycieku bfill)")
    final_3d_dict = {}

    # Bezpieczny fallback medyczny, jeśli słownik z zewnątrz nie został dostarczony
    if default_population_values is None:
        logger.warning(
            "Nie przekazano słownika default_population_values. "
            "Używam standardowych norm medycznych (10.0 dla ICP, 13.0 dla GCS)."
        )
        default_population_values = {
            0: 10.0,  # Domyślne ICP
            1: 13.0   # Domyślne GCS
        }

    for hid, matrix in patient_trajectories.items():
        df_seq = pd.DataFrame(matrix)

        # Forward Fill: przenosi ostatnią znaną wartość pacjenta na kolejne doby
        df_seq.ffill(inplace=True)

        # Używamy przekazanych danych populacyjnych jako bezpiecznych wartości domyślnych dla braków początkowych
        for col_idx, default_val in default_population_values.items():
            if col_idx in df_seq.columns:
                df_seq[col_idx].fillna(default_val, inplace=True)

        # Ostateczny bezpiecznik
        df_seq.fillna(0.0, inplace=True)

        final_3d_dict[hid] = df_seq.values

    return final_3d_dict


def get_cached_mimic_3d_timeseries(filepath, item_ids, admittimes_dict, max_days=14, default_population_values=None):
    """ Funkcja Wrapper: Błyskawicznie ładuje 30GB przetworzonych szeregów z dysku. """
    cache_path = os.path.join(BASE_DIR, 'Data', 'mimic_3d_timeseries_cache.pkl')

    if os.path.exists(cache_path):
        logger.info(f"Znaleziono zbuforowane szeregi 3D w '{cache_path}'. Omijam analizę plików CSV.")
        with open(cache_path, 'rb') as f:
            final_3d_dict = pickle.load(f)
        return final_3d_dict
    else:
        if not os.path.exists(filepath):
            logger.warning(f"Brak pliku z danymi MIMIC ({filepath}). Zwracam pusty słownik.")
            return {}

        logger.info("Brak bufora dyskowego. Uruchamiam pełną ekstrakcję z Chartevents.")
        
        # Przekazanie wyliczonych średnich w dół do głównej funkcji ekstrakcji
        final_3d_dict = extract_mimic_timeseries_3d(
            filepath, 
            item_ids, 
            admittimes_dict, 
            max_days=max_days, 
            default_population_values=default_population_values
        )

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(final_3d_dict, f)
        logger.info("Zapisano wyniki do pamięci podręcznej. Następne uruchomienie zajmie sekundy.")

        gc.collect()
        return final_3d_dict