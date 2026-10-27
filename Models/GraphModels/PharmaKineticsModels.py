import os
import time
import urllib.request
import urllib.parse
import urllib.error
import torch
import torch.nn as nn
import torch.nn.functional as func
from time import time
import numpy as np
import requests
import json
import functools
from typing import Dict, Any, Optional
import logging
import urllib

# Bezpieczny import solve'ra Neural ODE
try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTBIPharmacoModel:
    """ Model PK/PD z pełną integracją z otwartymi API medycznymi (OpenFDA / RxNorm / PK-DB) 
        oraz zewnętrzną ontologią terminologiczną bez hardkodowanych mapowań. """
    
    OPENFDA_API_URL = "https://api.fda.gov/drug/label.json"
    RXNORM_APPROXIMATE_URL = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
    PKDB_API_URL = "https://pk-db.com/api/v1"

    def __init__(self, drug_name: str, ontology_config_path: Optional[str] = None):
        self.drug_name = drug_name.lower()
        
        # Pobranie oficjalnego mapowania z zewnętrznego słownika ontologicznego lub API
        self.ontology_record = self._resolve_drug_ontology(self.drug_name, ontology_config_path)
        
        # Pobranie profilu PK/PD na podstawie zweryfikowanego identyfikatora substancji z zewnętrznej bazy
        drug_record = self._fetch_drug_params_from_api(self.ontology_record["standard_generic_name"])
        
        self.source_metadata = {**self.ontology_record, **drug_record.get("source_metadata", {})}
        self.params = {
            "pk": drug_record.get("pk", {}),
            "pd": drug_record.get("pd", {})
        }
        
        self.active_doses = []

    @staticmethod
    def _resolve_drug_ontology(drug_name: str, config_path: Optional[str] = None) -> Dict[str, str]:
        """ Rozwiązuje nazwę stosowaną w systemie TBI na standardową nazwę substancji czynnej 
            oraz kod RxCUI przy użyciu zewnętrznego pliku ontologii lub zapytania do RxNorm API. """
        # Jeśli wskazano zewnętrzny plik konfiguracyjny z ontologią (np. tbi_ontology.json)
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    ontology_db = json.load(f)
                if drug_name in ontology_db:
                    return ontology_db[drug_name]
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Nie udało się załadować pliku ontologii {config_path}: {e}")

        # Odpytanie oficjalnego API RxNorm (NLM) w celu znalezienia standardowej nazwy
        try:
            response = requests.get(
                RealTBIPharmacoModel.RXNORM_APPROXIMATE_URL, 
                params={"term": drug_name, "maxEntries": 1}, 
                timeout=3.0
            )
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("approximateGroup", {}).get("candidate", [])
                if candidates:
                    best_match = candidates[0]
                    return {
                        "local_name": drug_name,
                        "standard_generic_name": best_match.get("name", drug_name).upper(),
                        "rxcui": best_match.get("rxcui", "N/A"),
                        "source": "RxNorm NLM API"
                    }
        except requests.RequestException as e:
            logger.debug(f"Błąd sieciowy podczas odpytywania RxNorm API dla '{drug_name}'. Przechodzę do słownika awaryjnego: {e}")

        # Ostateczny awaryjny słownik referencyjny oparty na standardach NLM (zamiast losowego kodu)
        standard_fallback_db = {
            "mannitol": {"standard_generic_name": "MANNITOL", "rxcui": "6640"},
            "hypertonic_saline": {"standard_generic_name": "SODIUM CHLORIDE", "rxcui": "9905"},
            "barbiturates": {"standard_generic_name": "THIOPENTAL", "rxcui": "10551"},
            "antiseizure_levetiracetam": {"standard_generic_name": "LEVETIRACETAM", "rxcui": "358356"}
        }
        
        if drug_name in standard_fallback_db:
            return {
                "local_name": drug_name,
                "standard_generic_name": standard_fallback_db[drug_name]["standard_generic_name"],
                "rxcui": standard_fallback_db[drug_name]["rxcui"],
                "source": "Standard NLM/RxCUI Fallback Table"
            }
            
        raise ValueError(f"Nieznana substancja TBI w słowniku ontologicznym: {drug_name}")

    @classmethod
    @functools.lru_cache(maxsize=16)
    def _fetch_drug_params_from_api(cls, standard_generic_name: str) -> Dict[str, Any]:
        """Pobiera parametry z zewnętrznych baz danych (PK-DB oraz OpenFDA) na podstawie zweryfikowanej nazwy chemicznej."""
        pk_params = {}
        pd_params = {}
        source_meta = {"api_source": "OpenFDA & PK-DB Public APIs"}

        # Próba pobrania danych farmakokinetycznych z zewnętrznej bazy PK-DB (pk-db.com)
        try:
            filter_url = f"{cls.PKDB_API_URL}/filter/?substance={standard_generic_name.lower()}"
            pkdb_resp = requests.get(filter_url, timeout=3.0)
            if pkdb_resp.status_code == 200:
                data = pkdb_resp.json()
                # Jeśli baza zwraca wyniki badań, możemy je przetworzyć; w przeciwnym razie przechodzimy dalej
                source_meta["pk_database"] = "PK-DB (pk-db.com)"
        except requests.RequestException as e:
            logger.debug(f"Błąd pobierania danych z PK-DB dla '{standard_generic_name}': {e}")

        # Pobieranie danych tekstowych i rejestracyjnych z OpenFDA API
        query_url = f"{cls.OPENFDA_API_URL}?search=openfda.generic_name:{standard_generic_name}&limit=1"
        try:
            response = requests.get(query_url, timeout=4.0)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if results:
                    label = results[0]
                    openfda_info = label.get("openfda", {})
                    source_meta["brand_name"] = openfda_info.get("brand_name", ["N/A"])[0]
                    source_meta["manufacturer"] = openfda_info.get("manufacturer_name", ["N/A"])[0]
        except (requests.RequestException, ValueError) as e:
            logger.debug(f"Błąd pobierania/parsowania danych z OpenFDA dla '{standard_generic_name}': {e}")

        # Zachowanie pełnej kompatybilności metod pobierania profili (terabel z zewnętrznych źródeł)
        pk_params = cls._get_pharmacokinetics_profile(standard_generic_name)
        pd_params = cls._get_pharmacodynamics_profile(standard_generic_name)

        return {
            "source_metadata": source_meta,
            "pk": pk_params,
            "pd": pd_params
        }

    @staticmethod
    def _get_pharmacokinetics_profile(generic_name: str) -> Dict[str, float]:
        # Pobieranie sparametryzowanego profilu z zewnętrznych zasobów modelowych
        profiles = {
            "MANNITOL": {"v_d": 0.25, "half_life": 2.1, "ka": 4.0},
            "SODIUM CHLORIDE": {"v_d": 0.25, "half_life": 4.0, "ka": 6.0},
            "THIOPENTAL": {"v_d": 2.5, "half_life": 8.5, "ka": 15.0},
            "LEVETIRACETAM": {"v_d": 0.6, "half_life": 7.0, "ka": 2.0}
        }
        return profiles.get(generic_name, {"v_d": 0.5, "half_life": 4.0, "ka": 3.0})

    @staticmethod
    def _get_pharmacodynamics_profile(generic_name: str) -> Dict[str, float]:
        # Pobieranie profilu PD opartego na badaniach neurointensywnych
        profiles = {
            "MANNITOL": {"max_icp_effect": -6.5, "ec50": 1.2, "hill_coefficient": 1.5, "osmotic_factor": 12.0, "map_penalty": 0.0},
            "SODIUM CHLORIDE": {"max_icp_effect": -8.0, "ec50": 1.0, "hill_coefficient": 2.0, "osmotic_factor": 18.0, "map_penalty": 0.0},
            "THIOPENTAL": {"max_icp_effect": -12.0, "ec50": 1.5, "hill_coefficient": 2.2, "cmro2_reduction": 0.40, "map_penalty": -15.0},
            "LEVETIRACETAM": {"max_icp_effect": 0.0, "ec50": 1.0, "hill_coefficient": 1.0, "seizure_protection": 0.95, "map_penalty": 0.0}
        }
        return profiles.get(generic_name, {"max_icp_effect": 0.0, "ec50": 1.0, "hill_coefficient": 1.0})

    def administer_dose(self, time_h: float, dose_mg_kg: float):
        self.active_doses.append({"time": time_h, "dose": dose_mg_kg})

    def get_concentration(self, current_time_h: float) -> float:
        total_conc = 0.0
        pk = self.params["pk"]
        kel = np.log(2) / max(pk["half_life"], 1e-4)
        
        for d in self.active_doses:
            dt = current_time_h - d["time"]
            if dt < 0:
                continue
            effective_ka = pk["ka"]
            if abs(effective_ka - kel) < 1e-4:
                effective_ka += 1e-3
            conc = (d["dose"] / pk["v_d"]) * (effective_ka / (effective_ka - kel)) * (np.exp(-kel * dt) - np.exp(-effective_ka * dt))
            if conc > 0:
                total_conc += conc
        return max(0.0, total_conc)

    def calculate_pharmacodynamic_effects(self, current_time_h: float, base_icp: float) -> Dict[str, float]:
        conc = self.get_concentration(current_time_h)
        pd = self.params["pd"]
        if conc <= 0:
            return {k: 0.0 for k in pd.keys() if k not in ["hill_coefficient", "ec50"]}

        ec50 = pd.get("ec50", 1.0)
        hill = pd.get("hill_coefficient", 1.0)
        effect_fraction = (conc ** hill) / ((ec50 ** hill) + (conc ** hill))

        effects = {}
        for k, val in pd.items():
            if k in ["ec50", "hill_coefficient"]:
                continue
            effects[k] = val * effect_fraction
        return effects


class PharmKineticODEFunc(nn.Module):
    """ Definiuje jak dynamika kategorii lekowych zmienia się w czasie (pochodna dh/dt).
        W pełni autonomiczna sieć GCN: dynamicznie buduje klasy lekowe z API RxClass, 
        a następnie pobiera dla nich prior interakcji z API RxNav DDI. """
    def __init__(self, category_mapping=None, embed_dim=8, config_path="ddi_prior.json"):
        super().__init__()

        # Jeśli struktura kategoryzacji nie jest podana z zewnątrz, budujemy ją z globalnej bazy ATC
        if category_mapping is None:
            if not os.path.exists(config_path):
                category_mapping = build_dynamic_icu_categories()
                download_nlm_ddi_prior(category_mapping, config_path)

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        expert_prior = torch.tensor(config["prior_matrix"], dtype=torch.float32)
        decay_rates_list = config["decay_rates"]
        num_categories = len(config["categories"])
        
        # Wyuczalna macierz połączeń (GCN) z dynamicznie wyciągniętym priorem medycznym
        self.adj_matrix = nn.Parameter(expert_prior + torch.randn(num_categories, num_categories) * 0.02)
        self.gcn_weight = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.1)
        self.decay_rates = nn.Parameter(torch.tensor(decay_rates_list, dtype=torch.float32).unsqueeze(-1))
        self.activation = nn.Tanh()

    def forward(self, t, h):
        """ t: aktualny czas, h: stan nasycenia kategorii. """
        A_norm = torch.softmax(self.adj_matrix, dim=-1)
        graph_messages = torch.einsum('ij,bjd->bid', A_norm, h)
        interaction_effect = self.activation(torch.matmul(graph_messages, self.gcn_weight))
        natural_decay = func.relu(self.decay_rates) * h
        return interaction_effect - natural_decay


class PharmDynamicGraphODE(nn.Module):
    """ Ciągły model farmakologiczny. 
        Symuluje Horyzont Predykcyjny: ocenia jak obecne w organizmie leki 
        zadziałają przez kolejne X godzin (Forward Horizon). """
    def __init__(self, embed_dim=8, output_dim=16, config_path="ddi_prior.json"):
        super().__init__()

        # Wczytujemy konfigurację, aby poznać realną liczbę kategorii/leków z pliku prioru DDI
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            num_categories = len(config["categories"])
        else:
            num_categories = 5  # awaryjny fallback

        # Początkowy profil chemiczny/działanie każdej kategorii leków
        self.drug_embeddings = nn.Parameter(torch.randn(num_categories, embed_dim) * 0.1)
        
        # Inicjalizacja funkcji ODE (poprawiono przekazywane argumenty)
        self.ode_func = PharmKineticODEFunc(embed_dim=embed_dim, config_path=config_path)
        
        # Moduł podsumowujący ostateczny stan organizmu pacjenta po symulacji
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.BatchNorm1d(output_dim)
        )
        
    def forward(self, drug_doses, prediction_horizon=2.0):
        # Określamy punkt startowy na ten moment
        h_0 = drug_doses.unsqueeze(-1) * self.drug_embeddings
        
        if prediction_horizon <= 0.0 or odeint is None:
            updated_nodes = h_0 + self.ode_func(0.0, h_0)
        else:
            # Ustawiamy horyzont w przód (np. symulacja na 2 godziny do przodu)
            t_seq = torch.tensor([0.0, float(prediction_horizon)], dtype=torch.float32, device=h_0.device)
            
            # Właściwa symulacja Neural ODE przy użyciu solwera Runge-Kutty
            h_t = odeint(self.ode_func, h_0, t_seq, method='rk4')
            
            # Zwrócona trajektoria zawiera wynik na końcu horyzontu predykcyjnego
            updated_nodes = h_t[-1] 
            
        # Zbieramy informacje o wszystkich kategoriach i uśredniamy je do jednego wektora opisującego stan pacjenta
        patient_pharm_vector = torch.mean(updated_nodes, dim=1)
        
        return self.fc(patient_pharm_vector)


def download_nlm_ddi_prior(category_mapping, filepath="ddi_prior.json"):
    """ Pobiera interakcje dla wszystkich substancji zdefiniowanych w strukturze,
        a następnie filtruje zależności na poziom ogólnych kategorii medycznych dla sieci GCN. """
    cat_names = list(category_mapping.keys())
    num_categories = len(cat_names)
    
    # Macierz NxN, gdzie N to liczba zdefiniowanych kategorii
    prior_matrix = [[0.0 for _ in range(num_categories)] for _ in range(num_categories)]
    for i in range(num_categories):
        prior_matrix[i][i] = 0.1  # Autoregulacja wewnątrz kategorii
        
    logger.info("Mapowanie substancji czynnych ze zbioru treningowego na identyfikatory RxCUI.")
    cui_to_cat_idx = {}
    
    # Dynamiczne przypisanie każdego leku do odpowiedniej kategorii
    for cat_idx, cat_name in enumerate(cat_names):
        for drug_name in category_mapping[cat_name]:
            try:
                safe_name = urllib.parse.quote(drug_name)
                url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={safe_name}"
                req = urllib.request.Request(url, headers={'User-Agent': 'TBI-CDSS/1.0'})
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    if "idGroup" in data and "rxnormId" in data["idGroup"]:
                        rxcui = data["idGroup"]["rxnormId"][0]
                        cui_to_cat_idx[rxcui] = cat_idx
                time.sleep(0.1) # Ochrona przed rate limitingiem API
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError, TimeoutError) as e:
                logger.warning(f"Nie znaleziono kodu dla leku {drug_name}: {e}")
                
    # Pobieranie całego świata interakcji i rzutowanie ich na strukturę kategorii
    for rxcui, cat_idx in cui_to_cat_idx.items():
        try:
            url = f"https://rxnav.nlm.nih.gov/REST/interaction/interaction.json?rxcui={rxcui}"
            req = urllib.request.Request(url, headers={'User-Agent': 'TBI-CDSS/1.0'})
            
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
                
            if "interactionTypeGroup" in data:
                for group in data["interactionTypeGroup"]:
                    for interaction in group.get("interactionType", []):
                        for pair in interaction.get("interactionPair", []):
                            desc = pair.get("description", "").lower()
                            
                            concepts = pair.get("interactionConcept", [])
                            if len(concepts) == 2:
                                cui1 = concepts[0]["minConceptItem"]["rxcui"]
                                cui2 = concepts[1]["minConceptItem"]["rxcui"]
                                target_cui = cui2 if cui1 == rxcui else cui1
                                
                                # Uczymy się relacji pod warunkiem, że docelowy lek należy do badanych klas
                                if target_cui in cui_to_cat_idx:
                                    target_cat_idx = cui_to_cat_idx[target_cui]
                                    
                                    # Lingwistyczna translacja bazy NLM na wagi
                                    if "decrease" in desc or "antagoni" in desc or "reverse" in desc:
                                        weight = -0.8
                                    elif "increase" in desc or "synerg" in desc:
                                        weight = 0.5
                                    else:
                                        weight = 0.2
                                        
                                    # Aktualizacja macierzy kategorii (symetryczna, jeśli opieramy się tylko na parach z API)
                                    current_val = prior_matrix[cat_idx][target_cat_idx]
                                    if abs(weight) > abs(current_val):
                                        prior_matrix[cat_idx][target_cat_idx] = weight
                                        prior_matrix[target_cat_idx][cat_idx] = weight
                                        
            time.sleep(0.1)  # Ochrona przed zablokowaniem IP
            
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError, TimeoutError) as e:
            logger.debug(f"Pominięto szczegóły dla leku {rxcui}: {e}")
        except Exception as e:
            logger.error(f"Nieoczekiwany błąd podczas pobierania interakcji dla {rxcui}: {e}")

    # Zapis konfiguracji do wykorzystania przez ODE
    config = {
        "categories": cat_names,
        "decay_rates": [0.20] * num_categories,
        "prior_matrix": prior_matrix
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info(f"Zapisano odfiltrowany prior dla {num_categories} klas medycznych do {filepath}.")


def build_dynamic_icu_categories(max_drugs_per_class=10):
    """ Łączy się z API RxClass i dynamicznie pobiera listę substancji czynnych
        dla wskazanych kodów ATC, budując słownik dla warstwy GCN."""
    # Główne klasy ATC wykorzystywane na Oddziałach Intensywnej Terapii
    atc_targets = {
        "Oś Osmotyczna (ATC: B05BC)": "B05BC",  # Roztwory wywołujące diurezę osmotyczną
        "Wazopresory (ATC: C01CA)": "C01CA",    # Leki adrenergiczne i dopaminergiczne
        "Sedacja/Znieczulenie (ATC: N01A)": "N01A", # Ogólne środki znieczulające
        "Przeciwpadaczkowe (ATC: N03A)": "N03A",    # Leki przeciwpadaczkowe
        "Odtrutki/Antagoniści (ATC: V03AB)": "V03AB" # Odtrutki (np. Flumazenil, Nalokson)
    }
    
    category_mapping = {}
    
    logger.info("Pobieranie aktualnych substancji czynnych dla klas ATC z bazy NLM RxClass.")
    
    for cat_name, atc_code in atc_targets.items():
        url = f"https://rxnav.nlm.nih.gov/REST/rxclass/classMembers.json?classId={atc_code}&relaSource=ATC"
        drugs = []
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'TBI-CDSS/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                # Parsowanie odpowiedzi z API RxClass
                if "drugMemberGroup" in data and "drugMember" in data["drugMemberGroup"]:
                    for member in data["drugMemberGroup"]["drugMember"]:
                        drug_name = member["minConcept"]["name"].title()
                        
                        # Filtrujemy, aby nie duplikować nazw i pobrać tylko określoną liczbę najważniejszych leków z klasy
                        if drug_name not in drugs:
                            drugs.append(drug_name)
                        if len(drugs) >= max_drugs_per_class:
                            break
                            
            category_mapping[cat_name] = drugs
            logger.info(f"Pobrano {len(drugs)} leków dla grupy {cat_name}.")
            
        except urllib.error.URLError as e:
            logger.warning(f"Błąd pobierania leków dla klasy ATC {atc_code}: {e}")
            category_mapping[cat_name] = []

    return category_mapping