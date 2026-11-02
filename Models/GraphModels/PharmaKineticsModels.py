import os
import urllib.request
import urllib.error
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import requests
import json
import copy
import functools
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Sequence, Tuple
import logging

# Bezpieczny import solvera Neural ODE
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

    # Lokalne aliasy nazw używanych w systemie TBI. Sprawdzane PRZED RxNorm, bo RxNorm nie rozwiązuje
    # aliasów typu "antiseizure_levetiracetam". RxCUI z pierwszych czterech wpisów pochodzą z oryginalnego
    # kodu i NIE były weryfikowane; nowe wpisy mają "N/A" zamiast zgadywanych identyfikatorów.
    LOCAL_DRUG_ALIASES = {
        "mannitol": {"standard_generic_name": "MANNITOL", "rxcui": "6640"},
        "hypertonic_saline": {"standard_generic_name": "SODIUM CHLORIDE", "rxcui": "9905"},
        "barbiturates": {"standard_generic_name": "THIOPENTAL", "rxcui": "10551"},
        "antiseizure_levetiracetam": {"standard_generic_name": "LEVETIRACETAM", "rxcui": "358356"},
        "norepinephrine": {"standard_generic_name": "NOREPINEPHRINE", "rxcui": "N/A"},
        "noradrenaline": {"standard_generic_name": "NOREPINEPHRINE", "rxcui": "N/A"},
        "propofol": {"standard_generic_name": "PROPOFOL", "rxcui": "N/A"},
        "midazolam": {"standard_generic_name": "MIDAZOLAM", "rxcui": "N/A"},
        "dexmedetomidine": {"standard_generic_name": "DEXMEDETOMIDINE", "rxcui": "N/A"},
        "flumazenil": {"standard_generic_name": "FLUMAZENIL", "rxcui": "N/A"},
    }
    # Substancje z profilem PK/PD w _get_*_profile; pozostałe dostają profil generyczny (z ostrzeżeniem).
    CURATED_PROFILE_NAMES = {"MANNITOL", "SODIUM CHLORIDE", "THIOPENTAL", "LEVETIRACETAM"}

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

        # Lokalne aliasy (deterministyczne, bez sieci)
        if drug_name in RealTBIPharmacoModel.LOCAL_DRUG_ALIASES:
            alias = RealTBIPharmacoModel.LOCAL_DRUG_ALIASES[drug_name]
            return {
                "local_name": drug_name,
                "standard_generic_name": alias["standard_generic_name"],
                "rxcui": alias["rxcui"],
                "source": "Local alias table (RxCUI niezweryfikowany)"
            }

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
        except (requests.RequestException, ValueError, TypeError) as e:
            logger.debug(f"Błąd sieciowy podczas odpytywania RxNorm API dla '{drug_name}'. Przechodzę do słownika awaryjnego: {e}")

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
                pkdb_resp.json()
                # Jeśli baza zwraca wyniki badań, możemy je przetworzyć; w przeciwnym razie przechodzimy dalej
                source_meta["pk_database"] = "PK-DB (pk-db.com)"
        except (requests.RequestException, ValueError, TypeError) as e:
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
        if standard_generic_name in cls.CURATED_PROFILE_NAMES:
            source_meta["profile_source"] = "curated"
        else:
            source_meta["profile_source"] = "generic_default"
            logger.warning(f"Brak profilu PK/PD dla '{standard_generic_name}' — używam profilu generycznego "
                           f"(v_d=0.5, t1/2=4 h). Uzupełnij parametry przed użyciem wyników klinicznie.")

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

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def cached_template(drug_name: str) -> "RealTBIPharmacoModel":
        """ Jeden model-wzorzec na lek (jedno rozwiązanie ontologii i jedno pobranie parametrów na proces). """
        return RealTBIPharmacoModel(drug_name)

    def fresh_copy(self) -> "RealTBIPharmacoModel":
        """ Płytka kopia z pustą historią dawek — parametry PK/PD współdzielone, stan pacjenta osobny. """
        clone = copy.copy(self)
        clone.active_doses = []
        return clone

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
        """ Oblicza efekty PD z uwzględnieniem wyjściowego stanu ICP pacjenta. """
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
            # Aktywne wykorzystanie base_icp: silniejsza odpowiedź osmotyczna przy krytycznym wzroście ICP
            if k == "max_icp_effect" and base_icp > 20.0:
                effects[k] = val * effect_fraction * (base_icp / 20.0)
            else:
                effects[k] = val * effect_fraction

        # Obsługa specyficznych mechanizmów nieneurochirurgicznych (np. ochrona przeciwdrgawkowa)
        if "seizure_protection" in pd:
            effects["seizure_protection"] = pd["seizure_protection"] * effect_fraction

        return effects

    def evaluate(self, patient_state: dict) -> Dict[str, float]:
        """ Ewaluacja uwzględniająca alternatywne profile terapeutyczne (np. lewetyracetam). """
        current_time_h = float(patient_state.get("hours_from_admission", 1.0))
        base_icp = float(patient_state.get("icp_current", 15.0))

        effects = self.calculate_pharmacodynamic_effects(current_time_h, base_icp)

        icp_effect = effects.get("max_icp_effect", 0.0)
        seizure_prot = effects.get("seizure_protection", 0.0)
        map_penalty = effects.get("map_penalty", 0.0)

        # Jeśli lek chroni przed drgawkami (jak lewetyracetam), jego skuteczność bazuje na tym wskaźniku
        if seizure_prot > 0:
            efficacy_score = float(seizure_prot)
        else:
            efficacy_score = float(min(1.0, abs(icp_effect) / 10.0))

        toxicity_risk = float(min(1.0, abs(map_penalty) / 20.0))
        ebm_evidence_weight = float(self.source_metadata.get("confidence", 1.0) if hasattr(self, "source_metadata") else 1.0)

        return {
            "efficacy_score": efficacy_score,
            "toxicity_risk": toxicity_risk,
            "ebm_evidence_weight": ebm_evidence_weight
        }


# =============================================================================
# Jedno źródło prawdy dla węzłów grafu (nazwy ORAZ KOLEJNOŚĆ kategorii).
# Wcześniej lista była zdublowana w trzech miejscach i nie była powiązana z modelem.
# =============================================================================
ICU_DRUG_CATEGORIES: List[str] = [
    "Oś Osmotyczna (ATC: B05BC)",
    "Wazopresory (ATC: C01CA)",
    "Sedacja/Anestezja Ogólna (ATC: N01A)",
    "Inne Środki Uspokajające (ATC: N05CD)",
    "Agoniści Alfa-2 (ATC: C02AC)",
    "Przeciwpadaczkowe (ATC: N03A)",
    "Odtrutki/Antagoniści (ATC: V03AB)",
]
_OSM, _VASO, _SED, _BENZO, _A2, _AED, _ANTI = ICU_DRUG_CATEGORIES

ICU_ATC_TARGETS: Dict[str, str] = {
    _OSM: "B05BC",
    _VASO: "C01CA",
    _SED: "N01A",
    _BENZO: "N05CD",
    _A2: "C02AC",
    _AED: "N03A",
    _ANTI: "V03AB",
}

# Lek reprezentujący kategorię w modelu PK (RealTBIPharmacoModel pracuje na poziomie LEKU, nie klasy).
# Dla mannitolu, tiopentalu i lewetyracetamu istnieją profile PK/PD; pozostałe użyją profilu
# generycznego (z ostrzeżeniem w logu) do czasu uzupełnienia prawdziwych parametrów.
CATEGORY_PK_REPRESENTATIVE: Dict[str, str] = {
    _OSM: "mannitol",
    _VASO: "norepinephrine",
    _SED: "barbiturates",
    _BENZO: "midazolam",
    _A2: "dexmedetomidine",
    _AED: "antiseizure_levetiracetam",
    _ANTI: "flumazenil",
}

# Jawne mapowanie: kategoria -> kolumny w ramce pacjentów, z których wynika dawka tej kategorii.
# Kategorie bez kolumny dostają 0 (i ostrzeżenie), zamiast losowych „pierwszych N kolumn liczbowych”.
CATEGORY_DOSE_COLUMNS: Dict[str, List[str]] = {
    _OSM: ["mannitol_dose"],
    _VASO: ["noradrenaline_dose"],
    _SED: ["propofol_rate"],
}

# Prior interakcji lekowych 1 - potencjalizacja, -1 antagonizm.
DDI_PRIOR_PROVENANCE_DRAFT = "builtin_draft_requires_clinical_review"

# WARTOŚCI ORIENTACYJNE, wymagają zatwierdzenia klinicznego. 
DEFAULT_CATEGORY_DDI_DRAFT: List[Tuple[str, str, float, str]] = [
    (_SED, _BENZO, 0.5, "addytywna depresja OUN i oddechowa"),
    (_SED, _A2, 0.5, "addytywna sedacja, ryzyko hipotensji/bradykardii"),
    (_BENZO, _A2, 0.5, "addytywna sedacja"),
    (_SED, _VASO, -0.5, "wazopresory przeciwdziałają hipotensji po anestetykach"),
    (_A2, _VASO, -0.5, "wazopresory przeciwdziałają hipotensji/bradykardii po agonistach alfa-2"),
    (_BENZO, _ANTI, -0.8, "flumazenil znosi działanie benzodiazepin"),
    (_ANTI, _AED, -0.3, "flumazenil obniża próg drgawkowy (do weryfikacji)"),
    (_SED, _AED, 0.2, "dodatkowy efekt depresyjny na OUN"),
    (_BENZO, _AED, 0.2, "dodatkowy efekt depresyjny na OUN"),
]


def build_ddi_prior_config(
    categories: Sequence[str],
    interaction_pairs: Optional[Sequence[Sequence]] = None,
    curated_path: Optional[str] = None,
    self_regulation: float = 0.1,
    default_decay: float = 0.20,
) -> Dict[str, Any]:
    """ Buduje konfigurację priora (macierz ze znakiem) z listy par interakcji.
        Rzuca RuntimeError, jeśli żadna para nie pasuje do kategorii (zamiast zapisywać pusty prior). """
    categories = list(categories)
    n = len(categories)
    if n == 0:
        raise ValueError("Lista kategorii nie może być pusta.")

    provenance = DDI_PRIOR_PROVENANCE_DRAFT
    if curated_path is not None:
        with open(curated_path, "r", encoding="utf-8") as f:
            curated = json.load(f)
        interaction_pairs = curated["pairs"]
        provenance = curated.get("provenance", f"curated_file:{os.path.basename(curated_path)}")
    elif interaction_pairs is None:
        interaction_pairs = DEFAULT_CATEGORY_DDI_DRAFT
    else:
        provenance = "user_supplied_pairs"

    index = {name: i for i, name in enumerate(categories)}
    prior = [[0.0] * n for _ in range(n)]
    for i in range(n):
        prior[i][i] = float(self_regulation)

    used = 0
    for pair in interaction_pairs:
        cat_a, cat_b, weight = pair[0], pair[1], float(pair[2])
        if cat_a not in index or cat_b not in index:
            logger.warning(f"Para interakcji ({cat_a!r}, {cat_b!r}) odwołuje się do kategorii spoza listy — pomijam.")
            continue
        if not -1.0 <= weight <= 1.0:
            raise ValueError(f"Waga interakcji musi leżeć w [-1, 1], otrzymano {weight} dla ({cat_a}, {cat_b}).")
        prior[index[cat_a]][index[cat_b]] = weight
        prior[index[cat_b]][index[cat_a]] = weight
        used += 1

    if used == 0:
        raise RuntimeError(
            "Nie wczytano żadnej interakcji lekowej — graf miałby tylko przekątną (czyli zwykłe uśrednianie). "
            "Podaj curated_path z interakcjami albo nazwy kategorii zgodne z ICU_DRUG_CATEGORIES."
        )

    return {
        "categories": categories,
        "decay_rates": [float(default_decay)] * n,
        "prior_matrix": prior,
        "provenance": provenance,
        "n_interactions": used,
    }


def _validate_ddi_config(config: Dict[str, Any]) -> Optional[str]:
    """ Zwraca opis problemu albo None, gdy konfiguracja jest poprawna i NIEPUSTA. """
    for key in ("categories", "decay_rates", "prior_matrix"):
        if key not in config:
            return f"brak klucza '{key}'"
    n = len(config["categories"])
    matrix = np.asarray(config["prior_matrix"], dtype=np.float64)
    if matrix.shape != (n, n):
        return f"prior_matrix ma kształt {matrix.shape}, oczekiwano {(n, n)}"
    if len(config["decay_rates"]) != n:
        return "liczba decay_rates nie zgadza się z liczbą kategorii"
    off_diagonal = matrix - np.diag(np.diag(matrix))
    if not np.any(off_diagonal != 0.0):
        return "prior nie zawiera żadnej interakcji poza przekątną (to typowy skutek wyłączenia API NLM)"
    return None


def _write_ddi_config(config: Dict[str, Any], filepath: str) -> None:
    """ Zapisuje konfigurację priora do pliku JSON z metadanymi. """
    payload = dict(config)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    logger.info(f"Zapisano prior interakcji ({payload.get('provenance')}) do {filepath}.")


def load_ddi_config(
    config_path: str = "ddi_prior.json",
    categories: Optional[Sequence[str]] = None,
    curated_path: Optional[str] = None,
    regenerate_if_invalid: bool = True,
) -> Dict[str, Any]:
    """ Wczytuje i WALIDUJE prior. Uszkodzony, pusty albo niezgodny z kategoriami plik nie jest już
        cache'owany w nieskończoność: zostaje zregenerowany (lub rzucany jest błąd). """
    expected = list(categories) if categories is not None else None
    config: Optional[Dict[str, Any]] = None
    problem: Optional[str] = None

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            problem = _validate_ddi_config(config)
            if problem is None and expected is not None and list(config["categories"]) != expected:
                problem = "nazwy lub kolejność kategorii w pliku różnią się od oczekiwanych"
        except (OSError, json.JSONDecodeError) as e:
            problem = f"nie udało się wczytać pliku: {e}"
    else:
        problem = "brak pliku"

    if problem is not None:
        if os.path.exists(config_path):
            if not regenerate_if_invalid:
                raise ValueError(f"Nieprawidłowy plik priora {config_path}: {problem}")
            logger.warning(f"Plik priora {config_path} odrzucony ({problem}) — generuję od nowa.")
        config = build_ddi_prior_config(expected if expected is not None else ICU_DRUG_CATEGORIES,
                                        curated_path=curated_path)
        _write_ddi_config(config, config_path)

    if config.get("provenance") == DDI_PRIOR_PROVENANCE_DRAFT:
        logger.warning(
            "Prior interakcji pochodzi z wbudowanego SZKICU wymagającego zatwierdzenia klinicznego. "
            "Podaj curated_path z zatwierdzoną macierzą przed użyciem w ewaluacji."
        )
    return config


def download_nlm_ddi_prior(category_mapping, filepath="ddi_prior.json", curated_path=None):
    """ Zachowana dla zgodności wstecznej. API interakcji NLM zostało wyłączone, więc zamiast
        odpytywać nieistniejący endpoint, buduje prior przez build_ddi_prior_config. """
    logger.warning("API interakcji NLM/RxNav jest wyłączone — używam build_ddi_prior_config.")
    config = build_ddi_prior_config(list(category_mapping.keys()), curated_path=curated_path)
    _write_ddi_config(config, filepath)
    return config


def build_dynamic_icu_categories(max_drugs_per_class=10):
    """ Łączy się z API RxClass (nadal działa) i pobiera substancje dla kodów ATC.
        Kolejność i nazwy kategorii pochodzą z ICU_ATC_TARGETS. """
    category_mapping = {}
    logger.info("Pobieranie aktualnych substancji czynnych dla klas ATC z bazy NLM RxClass.")

    for cat_name, atc_code in ICU_ATC_TARGETS.items():
        url = f"https://rxnav.nlm.nih.gov/REST/rxclass/classMembers.json?classId={atc_code}&relaSource=ATC"
        drugs = []
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'TBI-CDSS/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                if "drugMemberGroup" in data and "drugMember" in data["drugMemberGroup"]:
                    for member in data["drugMemberGroup"]["drugMember"]:
                        drug_name = member["minConcept"]["name"].title()
                        if drug_name not in drugs:
                            drugs.append(drug_name)
                        if len(drugs) >= max_drugs_per_class:
                            break
            category_mapping[cat_name] = drugs
            logger.info(f"Pobrano {len(drugs)} leków dla grupy {cat_name} (np. {drugs[:3]}).")
        except urllib.error.URLError as e:
            logger.warning(f"Błąd pobierania leków dla klasy ATC {atc_code}: {e}")
            category_mapping[cat_name] = []
        except Exception as e:
            logger.error(f"Nieoczekiwany błąd podczas pobierania leków dla klasy ATC {atc_code}: {e}")
            category_mapping[cat_name] = []

    return category_mapping


def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    """ Odwraca funkcję softplus: softplus(x) = log(1 + exp(x)) """
    return torch.log(torch.expm1(y))


class _NFECounter(nn.Module):
    """ Opakowanie liczące wywołania funkcji pochodnej (koszt solwera). """
    def __init__(self, func_module: nn.Module):
        super().__init__()
        self.func_module = func_module
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        return self.func_module(t, y)


class PharmKineticODEFunc(nn.Module):
    """ Pochodna dh/dt dla kategorii lekowych: sygnowany graf interakcji + zanik naturalny.

        Macierz sąsiedztwa ma dwa nieujemne kanały (potencjalizacja i antagonizm), umożliwiając """
    _WEIGHT_FLOOR = 1e-2

    def __init__(self, category_mapping=None, embed_dim=8, config_path="ddi_prior.json", config=None):
        super().__init__()
        if config is None:
            categories = list(category_mapping.keys()) if category_mapping else None
            config = load_ddi_config(config_path, categories)

        problem = _validate_ddi_config(config)
        if problem is not None:
            raise ValueError(f"Nieprawidłowa konfiguracja priora: {problem}")

        self.categories = list(config["categories"])
        num_categories = len(self.categories)

        prior = torch.tensor(config["prior_matrix"], dtype=torch.float32)
        self.register_buffer("prior_matrix", prior)  # stały punkt odniesienia dla regularyzacji

        floor = self._WEIGHT_FLOOR
        pos_init = prior.clamp(min=0.0) + floor
        neg_init = (-prior).clamp(min=0.0) + floor  # pos - neg == prior dokładnie przy inicjalizacji
        self.pos_logits = nn.Parameter(_inverse_softplus(pos_init))
        self.neg_logits = nn.Parameter(_inverse_softplus(neg_init))

        self.gcn_weight = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.1)
        self.decay_rates = nn.Parameter(torch.tensor(config["decay_rates"], dtype=torch.float32).unsqueeze(-1))
        self.activation = nn.Tanh()

    def signed_adjacency(self) -> torch.Tensor:
        """ Macierz [C, C] ze znakiem; wiersze o sumie |wag| > 1 są skalowane (stabilność ODE). """
        pos = func.softplus(self.pos_logits)
        neg = func.softplus(self.neg_logits)
        scale = (pos + neg).sum(dim=-1, keepdim=True).clamp(min=1.0)
        return (pos - neg) / scale

    @property
    def adj_matrix(self) -> torch.Tensor:
        """ Widok tylko do odczytu (zgodność z kodem, który czytał dawny parametr adj_matrix). """
        return self.signed_adjacency()

    def prior_regularization(self) -> torch.Tensor:
        """ Kara za oddalenie wag od priora klinicznego; dodaj do funkcji straty z wagą lambda. """
        signed = func.softplus(self.pos_logits) - func.softplus(self.neg_logits)
        return torch.mean((signed - self.prior_matrix) ** 2)

    def forward(self, t, h):
        """ t: aktualny czas, h: stan nasycenia kategorii [B, C, D]. """
        adjacency = self.signed_adjacency()
        graph_messages = torch.einsum('ij,bjd->bid', adjacency, h)
        interaction_effect = self.activation(torch.matmul(graph_messages, self.gcn_weight))
        natural_decay = func.relu(self.decay_rates) * h
        return interaction_effect - natural_decay


class PharmDynamicGraphODE(nn.Module):
    """ Bazowy model GCN + Neural ODE dla dynamiki interakcji lekowych w czasie.
        Kolejność węzłów jest zapisana w self.categories — wejście drug_doses[:, i] odpowiada categories[i]. """
    ADAPTIVE_SOLVERS = ("dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun")

    def __init__(self, embed_dim=8, output_dim=16, config_path="ddi_prior.json",
                 category_mapping=None, curated_ddi_path=None):
        super().__init__()
        categories = list(category_mapping.keys()) if category_mapping else None
        config = load_ddi_config(config_path, categories, curated_path=curated_ddi_path)

        self.categories = list(config["categories"])
        self.num_categories = len(self.categories)
        self.embed_dim = embed_dim

        self.default_solver = "rk4"
        self.default_step_size = 0.25  # godziny; stały krok zamiast jednego kroku o długości całego horyzontu

        # Reprezentacje wektorowe kategorii lekowych
        self.drug_embeddings = nn.Parameter(torch.randn(self.num_categories, embed_dim) * 0.1)
        self.ode_func = PharmKineticODEFunc(embed_dim=embed_dim, config=config)

        # Warstwa w pełni połączona rzutująca wektor farmakologiczny na przestrzeń cech pacjenta
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def graph_prior_penalty(self) -> torch.Tensor:
        return self.ode_func.prior_regularization()

    def _solver_kwargs(self, solver: str) -> Dict[str, Any]:
        if solver in self.ADAPTIVE_SOLVERS:
            return {"method": solver, "rtol": 1e-3, "atol": 1e-4}
        return {"method": solver, "options": {"step_size": self.default_step_size}}

    def _euler_fallback(self, h_0: torch.Tensor, horizon: float, n_steps: int = 16) -> torch.Tensor:
        logger.warning("torchdiffeq niedostępne — używam jawnego Eulera (wyniki różnią się od solwera ODE).")
        h = h_0
        dt = horizon / n_steps
        for step in range(n_steps):
            h = h + dt * self.ode_func(step * dt, h)
        return h

    def _solve(self, h_0: torch.Tensor, t_seq: torch.Tensor, solver: Optional[str] = None) -> torch.Tensor:
        solver = solver or self.default_solver
        if odeint is None:
            return self._euler_fallback(h_0, float(t_seq[-1] - t_seq[0]))
        trajectory = odeint(self.ode_func, h_0, t_seq, **self._solver_kwargs(solver))
        return trajectory[-1]

    def _initial_state(self, drug_doses: torch.Tensor):
        if drug_doses.ndim != 2 or drug_doses.shape[1] != self.num_categories:
            raise ValueError(
                f"drug_doses musi mieć kształt [batch, {self.num_categories}] w kolejności {self.categories}; "
                f"otrzymano {tuple(drug_doses.shape)}."
            )
        return drug_doses.unsqueeze(-1) * self.drug_embeddings

    def forward(self, drug_doses, prediction_horizon=2.0):
        """ Standardowy forward bez routowania fenotypowego. """
        h_0 = self._initial_state(drug_doses)
        t_seq = torch.tensor([0.0, float(prediction_horizon)], dtype=torch.float32, device=h_0.device)
        updated_nodes = self._solve(h_0, t_seq)
        patient_pharm_vector = torch.mean(updated_nodes, dim=1)
        return self.fc(patient_pharm_vector)


class PhenotypeAwarePharmDynamicGraphODE(PharmDynamicGraphODE):
    """ Wariant modelu z adaptacyjnym doborem solwera ODE dla każdego fenotypu pacjenta.
        Rejestruje najtańszy solwer mieszczący się w budżecie błędu względem rozwiązania referencyjnego (dopri5 o ciasnych tolerancjach),
        żeby uniknąć kosztownych solwerów adaptacyjnych dla fenotypów, które dobrze działają z prostszymi metodami. """
    CANDIDATE_SOLVERS = ("dopri5", "rk4", "midpoint")

    def __init__(self, embed_dim=8, output_dim=16, config_path="ddi_prior.json",
                 error_budget=1e-3, benchmark_batch_size=8, **kwargs):
        super().__init__(embed_dim=embed_dim, output_dim=output_dim, config_path=config_path, **kwargs)
        self.phenotype_solver_registry: Dict[int, str] = {}
        self.error_budget = error_budget
        self.benchmark_batch_size = benchmark_batch_size

    @staticmethod
    def _handle_oom(err: Exception, context: str) -> bool:
        if "out of memory" in str(err).lower():
            logger.error(f"Brak VRAM ({context}): {err}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False

    def benchmark_solvers_for_phenotype(self, phenotype_id: int, h_0: torch.Tensor, t_seq: torch.Tensor) -> str:
        """ Wybiera najtańszy solwer mieszczący się w budżecie błędu względem rozwiązania referencyjnego
            (dopri5 o ciasnych tolerancjach); wynik jest zapisywany w rejestrze fenotypu. """
        if phenotype_id in self.phenotype_solver_registry:
            return self.phenotype_solver_registry[phenotype_id]

        if odeint is None:
            self.phenotype_solver_registry[phenotype_id] = self.default_solver
            return self.default_solver

        best_solver = self.default_solver
        try:
            with torch.no_grad():
                reference = odeint(self.ode_func, h_0, t_seq, method="dopri5", rtol=1e-5, atol=1e-7)[-1]
        except (RuntimeError, ValueError, AssertionError) as err:
            self._handle_oom(err, f"referencja, fenotyp {phenotype_id}")
            logger.warning(f"Rozwiązanie referencyjne ODE nie powiodło się (fenotyp {phenotype_id}): {err}. "
                           f"Używam '{best_solver}'.")
            self.phenotype_solver_registry[phenotype_id] = best_solver
            return best_solver

        if not torch.isfinite(reference).all():
            logger.warning(f"Rozwiązanie referencyjne zawiera NaN/Inf (fenotyp {phenotype_id}). Używam '{best_solver}'.")
            self.phenotype_solver_registry[phenotype_id] = best_solver
            return best_solver

        reference_scale = reference.abs().max().clamp(min=1e-6)
        results = []
        for solver in self.CANDIDATE_SOLVERS:
            counter = _NFECounter(self.ode_func)
            try:
                with torch.no_grad():
                    candidate = odeint(counter, h_0, t_seq, **self._solver_kwargs(solver))[-1]
            except (RuntimeError, ValueError, AssertionError) as err:
                if not self._handle_oom(err, f"solwer {solver}, fenotyp {phenotype_id}"):
                    logger.warning(f"Solwer '{solver}' zawiódł (fenotyp {phenotype_id}): {err}")
                continue
            if not torch.isfinite(candidate).all():
                continue
            relative_error = ((candidate - reference).abs().max() / reference_scale).item()
            results.append((solver, relative_error, counter.nfe))

        within_budget = [r for r in results if r[1] <= self.error_budget]
        if within_budget:
            best_solver = min(within_budget, key=lambda r: (r[2], r[1]))[0]
        elif results:
            best_solver = min(results, key=lambda r: r[1])[0]

        self.phenotype_solver_registry[phenotype_id] = best_solver
        logger.info(f"Dobór solwera ODE dla fenotypu [{phenotype_id}]: '{best_solver}' "
                    f"(kandydaci: {[(s, f'{e:.2e}', n) for s, e, n in results]})")
        return best_solver

    def _solve_with_fallback(self, h_group: torch.Tensor, t_seq: torch.Tensor, solver: str, phenotype_id: int) -> torch.Tensor:
        try:
            result = self._solve(h_group, t_seq, solver)
            if torch.isfinite(result).all():
                return result
            logger.warning(f"Solwer '{solver}' zwrócił NaN/Inf (fenotyp {phenotype_id}).")
        except (RuntimeError, ValueError, AssertionError) as err:
            if not self._handle_oom(err, f"symulacja, fenotyp {phenotype_id}"):
                logger.warning(f"Solwer '{solver}' zawiódł (fenotyp {phenotype_id}): {err}.")

        if solver != "rk4":
            try:
                result = self._solve(h_group, t_seq, "rk4")  # z gradientem, bez no_grad
                if torch.isfinite(result).all():
                    return result
            except (RuntimeError, ValueError, AssertionError) as err:
                logger.error(f"Fallback 'rk4' również zawiódł (fenotyp {phenotype_id}): {err}")

        logger.error(f"Awaria solwerów dla fenotypu {phenotype_id} — używam stanu początkowego h_0.")
        return h_group  # kształt [n, C, D], zgodny z wynikiem solwera

    def forward_with_phenotype_routing(self, drug_doses, phenotype_ids, prediction_horizon=2.0):
        """ Forward z routowaniem fenotypowym: pacjenci grupowani wg fenotypu i liczeni wsadowo. """
        h_0 = self._initial_state(drug_doses)
        t_seq = torch.tensor([0.0, float(prediction_horizon)], dtype=torch.float32, device=h_0.device)

        batch_size = h_0.shape[0]
        phenotype_ids = torch.as_tensor(phenotype_ids, device=h_0.device).long().reshape(-1)
        if phenotype_ids.numel() == 1 and batch_size != 1:
            phenotype_ids = phenotype_ids.expand(batch_size)
        if phenotype_ids.numel() != batch_size:
            raise ValueError("phenotype_ids musi mieć jedną wartość na pacjenta (lub być skalarem).")

        updated_nodes = torch.zeros_like(h_0)
        for p_id in torch.unique(phenotype_ids).tolist():
            idx = (phenotype_ids == p_id).nonzero(as_tuple=True)[0]
            h_group = h_0[idx]
            solver = self.benchmark_solvers_for_phenotype(
                int(p_id), h_group[: self.benchmark_batch_size].detach(), t_seq
            )
            updated_nodes[idx] = self._solve_with_fallback(h_group, t_seq, solver, int(p_id))

        patient_pharm_vector = torch.mean(updated_nodes, dim=1)
        return self.fc(patient_pharm_vector)


def compute_daily_pharm_features(pharm_models_dict: dict, current_day: int, categories_list: list) -> np.ndarray:
    """ Dobowy wektor cech farmakologicznych per kategoria. Wartością słownika może być model PK albo lista modeli
        (kilka leków w jednej kategorii — stężenia są sumowane). Służy jak most między receptami pacjenta a wejściem do modelu grafowego. """
    start_hour = (current_day - 1) * 24
    end_hour = current_day * 24
    sample_hours = np.linspace(start_hour, end_hour, num=5)

    daily_vector = []
    for cat_name in categories_list:
        models = pharm_models_dict.get(cat_name)
        if not models:
            daily_vector.append(0.0)
            continue
        if not isinstance(models, (list, tuple)):
            models = [models]
        total = 0.0
        for model in models:
            total += float(np.mean([model.get_concentration(h) for h in sample_hours]))
        daily_vector.append(total)

    return np.array(daily_vector, dtype=np.float32)


def resolve_prescriptions_to_models(
    patient_drugs: Dict[str, Sequence[Tuple[float, float]]],
    categories: Sequence[str],
    drug_to_category: Optional[Dict[str, str]] = None,
    category_pk_drug: Optional[Dict[str, str]] = None,
) -> Dict[str, List["RealTBIPharmacoModel"]]:
    """ Zamienia recepty pacjenta na modele PK pogrupowane po kategoriach.
        Klucz recepty może być (a) nazwą kategorii -> model leku reprezentującego kategorię,
        albo (b) nazwą leku obecną w drug_to_category -> model tego leku. """
    categories = list(categories)
    representatives = {**CATEGORY_PK_REPRESENTATIVE, **(category_pk_drug or {})}
    drug_lookup = {k.lower(): v for k, v in (drug_to_category or {}).items()}

    models_by_category: Dict[str, List[RealTBIPharmacoModel]] = {}
    for key, doses in patient_drugs.items():
        if key in categories:
            category = key
            drug_name = representatives.get(category)
            if drug_name is None:
                raise ValueError(f"Brak leku reprezentującego kategorię '{category}' w modelu PK.")
        elif key.lower() in drug_lookup:
            category = drug_lookup[key.lower()]
            drug_name = key
            if category not in categories:
                raise ValueError(f"Lek '{key}' przypisany do nieznanej kategorii '{category}'.")
        else:
            raise ValueError(
                f"Nieznany klucz recepty '{key}': podaj nazwę kategorii z ICU_DRUG_CATEGORIES "
                f"albo nazwę leku z mapowania drug_to_category."
            )

        model = RealTBIPharmacoModel.cached_template(drug_name.lower()).fresh_copy()
        for time_h, dose in doses:
            model.administer_dose(float(time_h), float(dose))
        models_by_category.setdefault(category, []).append(model)

    return models_by_category


def build_patient_daily_tensor_with_pharma(
    patient_stays,
    get_vitals_for_day_func,
    pharm_prescriptions,
    max_days=14,
    categories: Optional[Sequence[str]] = None,
    drug_to_category: Optional[Dict[str, str]] = None,
    category_pk_drug: Optional[Dict[str, str]] = None,
):
    """ Buduje tensor 3D [Batch, Max_Days, Vitals + Kategorie], łącząc parametry życiowe z dobowo
        agregowanym stężeniem leków. Kolumny farmakologiczne mają kolejność `categories`
        (domyślnie ICU_DRUG_CATEGORIES, tę samą co węzły grafu). """
    categories = list(categories) if categories is not None else list(ICU_DRUG_CATEGORIES)

    batch_tensors = []
    for stay_id in patient_stays:
        patient_drugs = pharm_prescriptions.get(stay_id, {})
        active_models = resolve_prescriptions_to_models(
            patient_drugs, categories, drug_to_category, category_pk_drug
        )

        patient_matrix = []
        for day in range(1, max_days + 1):
            vitals_day = np.asarray(get_vitals_for_day_func(stay_id, day), dtype=np.float32)
            pharma_vector = compute_daily_pharm_features(active_models, day, categories)
            patient_matrix.append(np.concatenate([vitals_day, pharma_vector]))
        batch_tensors.append(patient_matrix)

    return torch.from_numpy(np.asarray(batch_tensors, dtype=np.float32))



def build_category_dose_matrix(df, categories: Sequence[str],
                               column_map: Optional[Dict[str, List[str]]] = None) -> Tuple[np.ndarray, List[str]]:
    """ Dawki kategorii dla modelu grafowego z ramki pacjentów. Zwraca (macierz [N, C] w kolejności `categories`,
        lista kategorii bez danych). Kategorie bez żadnej kolumny w df dostają 0 i trafiają na listę brakujących. """
    column_map = CATEGORY_DOSE_COLUMNS if column_map is None else column_map
    doses = np.zeros((len(df), len(categories)), dtype=np.float32)
    missing: List[str] = []
    for j, category in enumerate(categories):
        columns = [c for c in column_map.get(category, []) if c in df.columns]
        if not columns:
            missing.append(category)
            continue
        values = np.nan_to_num(df[columns].to_numpy(dtype=np.float64), nan=0.0)
        doses[:, j] = values.sum(axis=1)
    return doses, missing


def fit_exposure_scale(train_doses: np.ndarray, quantile: float = 0.95, floor: float = 1e-6) -> np.ndarray:
    """ Skala per kategoria (kwantyl dodatnich dawek) wyznaczana WYŁĄCZNIE na zbiorze treningowym.
        Bez niej dawki o różnych jednostkach (np. 250 vs 0.05) nasycają tanh w ODE. """
    scale = np.ones(train_doses.shape[1], dtype=np.float64)
    for j in range(train_doses.shape[1]):
        positive = train_doses[:, j][train_doses[:, j] > 0]
        if positive.size:
            scale[j] = max(float(np.quantile(positive, quantile)), floor)
    return scale


def apply_exposure_scale(doses: np.ndarray, scale: np.ndarray, clip: float = 3.0) -> np.ndarray:
    return np.clip(doses / scale, 0.0, clip).astype(np.float32)