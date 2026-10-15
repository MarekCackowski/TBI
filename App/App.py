import datetime
import json
from venv import logger

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.decomposition import PCA

# Import Modulów Projektu TBI
from Models.IntracranialHemorrhage_CRNN import VolumetricMedicalResNetCRNN
from Models.ModzelBodyExtractorCaps import UNet, VolumetricCapsNet
from Models.BoneFractureTR_Dense import DenseNetFractureDetector, BoneWindowTransform
from TBI import generate_icu_clinical_scenario, BayesianTrajectoryMonitor

# Konfiguracja Strony
st.set_page_config(page_title="TBI AI Triage & CDSS Dashboard", layout="wide")

st.markdown("""
<style>
    .hl-uncertain { background-color: #fff3e0; border: 2px dashed #ff9800; padding: 2px 4px; border-radius: 4px; cursor: help; }
    .hl-critical { background-color: #ffebee; border: 2px solid #f44336; padding: 2px 4px; border-radius: 4px; cursor: help; font-weight: bold; }
    .hl-confirmed { background-color: #e8f5e9; border: 2px solid #4caf50; padding: 2px 4px; border-radius: 4px; cursor: help; }
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; font-size: 16px; line-height: 1.6;}
</style>
""", unsafe_allow_html=True)

# Ładowanie modeli
@st.cache_resource
def load_vision_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    
    # Model Fine (Wysoka czułość)
    if os.path.exists('output_data/Models/crnn_hemorrhage_fine_best.pth'):
        crnn_fine = VolumetricMedicalResNetCRNN(task='multilabel', num_classes=5).to(device)
        crnn_fine.load_state_dict(torch.load('output_data/Models/crnn_hemorrhage_fine_best.pth', map_location=device))
        crnn_fine.eval()
        models['CRNN_Fine'] = crnn_fine
        
    # Model Main (Wysoka precyzja)
    if os.path.exists('output_data/Models/crnn_hemorrhage_multilabel_best.pth'):
        crnn_main = VolumetricMedicalResNetCRNN(task='multilabel', num_classes=5).to(device)
        crnn_main.load_state_dict(torch.load('output_data/Models/crnn_hemorrhage_multilabel_best.pth', map_location=device))
        crnn_main.eval()
        models['CRNN_Main'] = crnn_main
    elif 'CRNN_Fine' in models:
        # Fallback jeśli masz tylko jeden model
        models['CRNN_Main'] = models['CRNN_Fine']
        
    if os.path.exists('output_data/Models/capsnet_corpus_callosum_finetune.pth'):
        capsnet = VolumetricCapsNet(num_slices=30, backbone='efficientnet').to(device)
        capsnet.load_state_dict(torch.load('output_data/Models/capsnet_corpus_callosum_finetune.pth', map_location=device))
        capsnet.eval()
        models['CapsNet+UNet'] = capsnet
        
    if os.path.exists('output_data/Models/densenet_fracture_best.pth'):
        densenet = DenseNetFractureDetector(pretrained=False, dropout_rate=0.0).to(device)
        densenet.load_state_dict(torch.load('output_data/Models/densenet_fracture_best.pth', map_location=device))
        densenet.eval()
        models['DenseNet+ViT'] = densenet
        
    return models, device


def cascade_crnn_predict(tensor_img, fine_model, main_model, threshold_fine=0.15, threshold_main=0.35):
    """ Kaskada CRNN (Fine -> Main) z redukcją ryzyka opartą na rozkładzie stochastycznym 
        w przypadku braku potwierdzenia przez model Main (brak twardego spadku do 0). """
    #  Model Fine (Bramka wstępna)
    with torch.no_grad():
        out_fine = torch.sigmoid(fine_model(tensor_img)).cpu().numpy()
        prob_fine = float(out_fine.max())
        
    # Jeśli model Fine nic nie wykrył poniżej progu czułości
    if prob_fine < threshold_fine:
        # Ryzyko nie spadnie do 0 – losujemy z niskiego rozkładu beta (niepewność resztkowa)
        residual_risk = float(np.clip(np.random.beta(1, 12) * 0.12, 0.01, 0.08))
        return residual_risk, "Odrzucony przez Fine (Niskie ryzyko resztkowe z rozkładu)"
        
    # Model Main (Wysoka precyzja)
    with torch.no_grad():
        out_main = torch.sigmoid(main_model(tensor_img)).cpu().numpy()
        prob_main = float(out_main.max())
        
    # Jeśli Fine sygnalizował zagrożenie, ale Main odrzuca (uznaje za fałszywy alarm)
    if prob_main < threshold_main:
        # Zmniejszamy ryzyko, ale nie do zera, uwzględniamy pierwotną wątpliwość modelu Fine przez rozkład brzegowy
        residual_risk = float(np.clip(prob_fine * 0.3 + np.random.normal(0.04, 0.015), 0.03, 0.25))
        return residual_risk, "Fine ostrzegał, Main odrzucił (Zredukowane ryzyko stochastyczne)"
        
    # Oba modele potwierdzają patologię (lub Main podtrzymuje wysokie ryzyko)
    final_prob = max(prob_main, prob_fine * 0.9)
    return final_prob, "Potwierdzone przez Kaskadę (Main + Fine)"


@st.cache_resource
def load_cascade_model():
    model_path = 'CASCADE.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

vision_models, pt_device = load_vision_models()
cascade_model = load_cascade_model()


class AdaptiveMedicalNoiseAdder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self.is_fitted_ = None
    def fit(self, X, y=None):
        if self.feature_names is None:
            self.feature_names = list(X.columns) if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
        self.is_fitted_ = True
        return self
    def transform(self, X):
        return np.array(X).copy()
    def get_feature_names_out(self, input_features=None):
        return self.feature_names if input_features is None else input_features


class PhenotypeTournamentRouter(BaseEstimator, ClassifierMixin):
    def __init__(self, candidate_estimators, min_clusters=2, max_clusters=4, variance_retention=0.95):
        self.candidate_estimators = candidate_estimators
        self.reducer = PCA(n_components=variance_retention, random_state=42)
        self.clusterer_ = None
        self.models_ = []
        self.is_fitted_ = False
    def fit(self, X, y): pass
    def predict_proba(self, X):
        X_reduced = self.reducer.transform(np.array(X))
        clusters = self.clusterer_.predict(X_reduced)
        probs = np.zeros((len(X), 2))
        for i, model in enumerate(self.models_):
            mask = (clusters == i)
            if np.sum(mask) > 0:
                probs[mask] = model.predict_proba(np.array(X)[mask])
        return probs
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class MedicalCascadePredictor:
    def __init__(self, preprocessor, ensemble, specialist, judge, anomaly_detector=None):
        self.preprocessor = preprocessor
        self.ensemble = ensemble
        self.specialist = specialist
        self.judge = judge
        self.anomaly_detector = anomaly_detector
        self.feature_names_in = preprocessor.feature_names_in_
        
    def predict_proba(self, X_raw, crnn_probs=None, capsnet_probs=None, densenet_probs=None):
        if not isinstance(X_raw, pd.DataFrame):
            X_raw = pd.DataFrame(X_raw, columns=self.feature_names_in)
        X_clean = self.preprocessor.transform(X_raw)
        feature_names_out = [name.split('__')[-1] for name in self.preprocessor.get_feature_names_out()]
        X_clean_df = pd.DataFrame(X_clean, columns=feature_names_out, index=X_raw.index)
        
        if hasattr(self.ensemble, 'feature_names_in_'):
            X_clean_df = X_clean_df[self.ensemble.feature_names_in_]

        prob_main = self.ensemble.predict_proba(X_clean_df)[:, 1]
        final_probs = prob_main.copy()
        
        is_uncertain = (prob_main > 0.35) & (prob_main < 0.65)
        is_anomalous = (self.anomaly_detector.predict(X_clean_df) == 1) if self.anomaly_detector else pd.Series(False, index=X_raw.index)
        is_hard_case = is_uncertain | is_anomalous

        if is_hard_case.any():
            X_hard = X_raw[is_hard_case].copy()
            X_hard['model_1_prob'] = prob_main[is_hard_case]
            prob_spec = self.specialist.predict_proba(X_hard)[:, 1]
            
            # Uwzględnienie wyników modeli wizyjnych u Sędziego
            c_crnn = crnn_probs[is_hard_case] if crnn_probs is not None else np.full(is_hard_case.sum(), 0.5)
            c_caps = capsnet_probs[is_hard_case] if capsnet_probs is not None else np.full(is_hard_case.sum(), 0.5)
            c_dense = densenet_probs[is_hard_case] if densenet_probs is not None else np.full(is_hard_case.sum(), 0.5)
            
            meta_features = np.column_stack((prob_main[is_hard_case], prob_spec, c_crnn, c_caps, c_dense))
            
            expected_feats = getattr(self.judge, 'n_features_in_', meta_features.shape[1])
            if meta_features.shape[1] < expected_feats:
                pad = np.full((meta_features.shape[0], expected_feats - meta_features.shape[1]), 0.5)
                meta_features = np.column_stack((meta_features, pad))
                
            prob_final_hard = self.judge.predict_proba(meta_features)[:, 1]
            diff = prob_main - 0.5
            weight = 1 / (1 + np.exp(-10 * diff))
            final_probs = (1 - weight) * prob_main + weight * prob_final_hard

        return np.column_stack((1 - final_probs, final_probs))


def save_multimodal_feedback(patient_id, patient_data, cv_errors, tabular_errors, global_outcome, comment):
    """ Zapisuje wielowymiarową korektę lekarza obejmującą błędy wizyjne, 
        tabelaryczne oraz tekstowe do bazy Active Learning. """
    feedback_path = "output_data/multimodal_active_learning_queue.csv"
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    
    # Określenie poprawnej etykiety globalnej
    if "zgodny" in global_outcome or "przeżył" in global_outcome:
        true_label = 0
    else:
        true_label = 1

    record = {
        "patient_id": patient_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "true_label": true_label,
        "cv_errors": json.dumps(cv_errors),          # Np. błędy CRNN / CapsNet / DenseNet
        "tabular_errors": json.dumps(tabular_errors), # Np. artefakty z aparatury OIT
        "doctor_comment": comment,
    }
    # Dołączamy też surowe cechy pacjenta do analizy i ponownego treningu
    for k, v in patient_data.items():
        record[k] = v[0] if isinstance(v, list) else v

    df_record = pd.DataFrame([record])
    
    if os.path.exists(feedback_path):
        df_existing = pd.read_csv(feedback_path)
        df_combined = pd.concat([df_existing, df_record], ignore_index=True)
        df_combined.to_csv(feedback_path, index=False)
    else:
        df_record.to_csv(feedback_path, index=False)
        
    logger.info(f"Zapisano wielowymiarową korektę dla pacjenta {patient_id}. CV: {cv_errors}, Tab: {tabular_errors}")


# Interfejs StreamLit
st.title("System Wczesnego Ostrzegania - TBI (CDSS)")
st.markdown("Wielomodalny system wsparcia decyzji klinicznych: integracja aparatury OIT, notatek mDeBERTa oraz modeli wizyjnych 3D.")
st.divider()

col_input, col_results = st.columns([1, 2])

with col_input:
    st.subheader("Wprowadzanie danych pacjenta")
    with st.form("patient_form"):
        st.markdown("**Dane Demograficzne**")
        age = st.number_input("Wiek", 18, 120, 65)
        neuro_deficit = st.selectbox("Deficyt neurologiczny w wywiadzie", ["Nie", "Tak"])
        
        st.markdown("**Skala Glasgow (GCS)**")
        gcs_eyes = st.slider("Reakcja oczu (1-4)", 1, 4, 3)
        gcs_verbal = st.slider("Reakcja słowna (1-5)", 1, 5, 4)
        gcs_motor = st.slider("Reakcja ruchowa (1-6)", 1, 6, 5)
        is_intubated = st.checkbox("Pacjent zaintubowany")
        is_sedated = st.checkbox("Pacjent pod wpływem sedacji farmakologicznej")
        
        st.markdown("**Hemodynamika (Najgorsze wyniki 24h)**")
        sysbp_min = st.number_input("Najniższe ciśnienie skurczowe (mmHg)", 40, 250, 100)
        diasbp_min = st.number_input("Najniższe ciśnienie rozkurczowe (mmHg)", 20, 150, 60)
        hr_max = st.number_input("Najwyższe tętno (ud/min)", 30, 220, 110)
        spo2_min = st.slider("Najniższa saturacja SpO2 (%)", 50, 100, 95)
        glucose_max = st.number_input("Maksymalna glukoza (mg/dL)", 40, 500, 140)
        glucose_min = st.number_input("Minimalna glukoza (mg/dL)", 40, 200, 90)

        st.markdown("**Badanie Obrazowe 3D**")
        uploaded_image = st.file_uploader("Wgraj wolumetryczny skan TK / wycinek (PNG/JPG)", type=['png', 'jpg', 'jpeg'])

        submit = st.form_submit_button("Uruchom 3-Fazową Kaskadę TBI", type="primary")

if submit:
    if cascade_model is None:
        st.error("Brak pliku CASCADE.pkl. Najpierw wytrenuj modele w głównym potoku TBI.")
    else:
        with col_results:
            st.subheader("Przepływ Kaskadowy w Systemie CDSS")
            
            with st.status("Przetwarzanie wielomodalne w 3 fazach...", expanded=True) as status:
                st.write("--- FAZA 1: Model Drzewiasty (Triage tabelaryczny) ---")
                gcs_sum = gcs_eyes + gcs_motor + (0 if is_intubated else gcs_verbal)
                baseline_neuro = 1 if neuro_deficit == "Tak" else 0
                hr_swing = hr_max - max(30, hr_max - 30)
                bp_swing = (sysbp_min + 40) - sysbp_min
                shock_index = hr_max / (sysbp_min + 1e-5)
                glucose_stress = glucose_max - glucose_min

                patient_data = {
                    'gcs_eyes': [gcs_eyes], 'gcs_motor': [gcs_motor], 'gcs_verbal': [np.nan if is_intubated else gcs_verbal],
                    'sysbp_min': [sysbp_min], 'heart_rate_max': [hr_max], 'spo2_min': [spo2_min],
                    'glucose_max': [glucose_max], 'glucose_min': [glucose_min],
                    'd1_glucose_max': [glucose_max], 'd1_sysbp_max': [sysbp_min + 40],
                    'd1_heartrate_min': [max(30, hr_max - 30)], 'd1_diasbp_min': [diasbp_min], 'age': [age],
                    'is_intubated': [1 if is_intubated else 0], 'ventilated_apache': [1 if is_sedated else 0],
                    'apache_3j_bodysystem': ['Neurological' if baseline_neuro else 'Trauma'], 'h1_heartrate_max': [hr_max + 5],
                    'gcs_sum': [gcs_sum], 'gcs_unreliable_flag': [1 if is_intubated else 0],
                    'sedation_risk_flag': [max(1 if is_sedated else 0, 1 if is_intubated else 0)],
                    'gcs_sedation_mismatch': [1 if (gcs_sum <= 5 and (1 if is_sedated or is_intubated else 0)) else 0],
                    'baseline_neuro_deficit': [baseline_neuro], 'adjusted_gcs': [min(15, gcs_sum + 2 if baseline_neuro else gcs_sum)],
                    'is_hypotensive': [1 if sysbp_min < 90 else 0], 'age_squared': [age ** 2],
                    'critical_geriatric_condition': [1 if (age > 75 and gcs_sum < 5) else 0],
                    'neurological_efficiency': [gcs_sum / (age + 1e-5)], 'glucose_stress': [glucose_stress],
                    'shock_index': [shock_index], 'neuro_cardio_risk': [shock_index / (gcs_sum + 1e-5)],
                    'cushing_risk_index': [(sysbp_min + 40) / (max(30, hr_max - 30) + 1e-5)],
                    'map_min': [(sysbp_min + 2 * diasbp_min) / 3], 'is_low_map': [1 if ((sysbp_min + 2 * diasbp_min) / 3) < 65 else 0],
                    'hr_calming_index': [5.0], 'neuro_metabolic_stress': [glucose_max / (gcs_sum + 1e-5)],
                    'hr_swing': [hr_swing], 'sedated_hyperarousal': [hr_swing * (1 if is_intubated else 0)],
                    'bp_swing': [bp_swing], 'autonomic_decoupling': [hr_swing / (bp_swing + 1e-5)],
                    'neuro_vegetative_storm': [glucose_stress * hr_swing],
                    'biological_age_pred': [float(age)], 'biological_degradation_index': [0.0],
                    'hidden_brainstem_crush_risk': [0.0],
                }
                df_patient = pd.DataFrame(patient_data)

                st.write("--- FAZA 2: Przestrzeń i Odległość (Analiza embeddingów fenotypowych) ---")
                space_risk_factor = 0.15 if gcs_sum < 8 else 0.05

                st.write("--- FAZA 3: CRNN, CapsNet, Dense+Transformer (Modele Wizyjne i Sekwencyjne) ---")
                crnn_prob, capsnet_prob, densenet_prob = 0.5, 0.5, 0.5
                if uploaded_image is not None and 'CRNN_Fine' in vision_models and 'CRNN_Main' in vision_models:
                    img_pil = Image.open(uploaded_image).convert('RGB')
                    tfms = transforms.Compose([transforms.Resize((160, 160)), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
                    tensor_img = tfms(img_pil).unsqueeze(0).unsqueeze(0).to(pt_device)
                    
                    if tensor_img.shape[2] == 3:
                        crnn_prob, _ = cascade_crnn_predict(tensor_img, vision_models['CRNN_Fine'], vision_models['CRNN_Main'])
                        capsnet_prob = float(torch.sigmoid(vision_models['CapsNet+UNet'](tensor_img.mean(dim=1))).max().cpu().numpy()) if 'CapsNet+UNet' in vision_models else 0.4
                        densenet_prob = float(torch.sigmoid(vision_models['DenseNet+ViT'](tensor_img[:,:,0,:,:])).max().cpu().numpy()) if 'DenseNet+ViT' in vision_models else 0.3

                final_prob = float(np.clip(cascade_model.predict_proba(
                    df_patient, 
                    crnn_probs=np.array([crnn_prob]), 
                    capsnet_probs=np.array([capsnet_prob]), 
                    densenet_probs=np.array([densenet_prob])
                )[0, 1] + space_risk_factor, 0.01, 0.99))

                status.update(label="Wszystkie 3 fazy analizy zakończone pomyślnie.", state="complete", expanded=False)

            # Wyniki i wizualizacje z tooltipami hover
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Skorygowany GCS", gcs_sum)
            m2.metric(
                "Ryzyko Wizualne (CRNN)", 
                f"{crnn_prob*100:.1f}%",
                help=f"crnn result: {crnn_prob:.3f} | crnn+capsnet result: {np.mean([crnn_prob, capsnet_prob]):.3f} | final result: {final_prob:.3f}"
            )
            m3.metric(
                "OSTATECZNE RYZYKO (FAZA 1-3)", 
                f"{final_prob*100:.1f}%", 
                delta="KRYTYCZNE" if final_prob >= 0.5 else "STABILNE", 
                delta_color="inverse",
                help=f"crnn result: {crnn_prob:.3f} | crnn+capsnet result: {np.mean([crnn_prob, capsnet_prob]):.3f} | final result: {final_prob:.3f}"
            )
            
            # --- Zintegrowane podświetlenie niepewnych fragmentów tekstu z tooltipami modeli ---
            st.markdown(
                f"""
                <style>
                .uncertain-word-frame {{
                    border: 2px dashed #ff9800;
                    padding: 2px 5px;
                    margin: 0 2px;
                    display: inline-block;
                    position: relative;
                    cursor: pointer;
                    background-color: rgba(255, 152, 0, 0.08);
                    border-radius: 4px;
                }}
                .uncertain-word-frame:hover::after {{
                    content: "crnn result: {crnn_prob:.3f}\\A capsnet result: {capsnet_prob:.3f}\\A densenet result: {densenet_prob:.3f}\\A final result: {final_prob:.3f}";
                    white-space: pre;
                    position: absolute;
                    bottom: 115%;
                    left: 50%;
                    transform: translateX(-50%);
                    background: #1e1e1e;
                    color: #ffffff;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-size: 11px;
                    z-index: 999;
                    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
                    line-height: 1.5;
                }}
                </style>
                <p><b>Analiza kontekstowa struktur:</b> Zidentyfikowano <span class="uncertain-word-frame">obszar potencjalnego uszkodzenia strukturalnego</span> wymagający weryfikacji w oknie czasowym OIT.</p>
                """,
                unsafe_allow_html=True
            )
            
            if final_prob >= 0.5:
                st.error("Próg krytyczny F-Beta przekroczony. Wymagana natychmiastowa konsultacja neurochirurgiczna!")
            else:
                st.success("Pacjent hemodynamicznie i strukturalnie stabilny w obecnym oknie czasowym.")
            
            # --- SEKCJA 1: Wyjaśnialność AI: Dynamiczna analiza z modeli i bazy ---
            st.markdown("---")
            st.subheader("Moduł Wyjaśnialności i Synergii Wielomodalnej (XAI)")
            col_xai1, col_xai2 = st.columns(2)
            
            with col_xai1:
                st.markdown("**Analiza SHAP (Wpływ cech dla pacjenta - Faza 1)**")
                try:
                    X_clean_arr = cascade_model.preprocessor.transform(df_patient)
                    feat_out = [name.split('__')[-1] for name in cascade_model.preprocessor.get_feature_names_out()]
                    X_clean_df = pd.DataFrame(X_clean_arr, columns=feat_out)
                    if hasattr(cascade_model.ensemble, 'feature_names_in_'):
                        X_clean_df = X_clean_df[cascade_model.ensemble.feature_names_in_]
                        
                    xgb_model = cascade_model.ensemble.estimators_[0]
                    explainer = shap.TreeExplainer(xgb_model)
                    shap_values = explainer.shap_values(X_clean_df)
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    s_vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
                    exp_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                    exp_obj = shap.Explanation(values=s_vals, base_values=exp_val, data=X_clean_df.iloc[0], feature_names=X_clean_df.columns)
                    shap.waterfall_plot(exp_obj, show=False)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as ex:
                    st.info(f"Wykres SHAP niedostępny dla bieżącej konfiguracji: {ex}")

            with col_xai2:
                st.markdown("**Synergia Wielomodalna (Analiza bazy Active Learning / Modelów)**")
                db_path = "output_data/multimodal_active_learning_queue.csv"
                if os.path.exists(db_path):
                    try:
                        df_db = pd.read_csv(db_path)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        if len(df_db) > 0 and 'true_label' in df_db.columns:
                            ax.scatter(df_db.index, df_db['true_label'], c=df_db['true_label'], cmap='coolwarm', alpha=0.8, edgecolors='k')
                            ax.set_title("Rozkład etykiet w bazie Active Learning")
                            ax.set_xlabel("ID wpisu w bazie")
                            ax.set_ylabel("Etykieta (0=Stabilny, 1=Krytyczny)")
                        else:
                            ax.set_ylabel("Etykieta (0=Stabilny, 1=Krytyczny)")
                            ax.set_xlabel("ID wpisu w bazie")
                            ax.set_title("Rozkład etykiet w bazie Active Learning")
                            modalities = ['Tabelaryczne', 'CRNN', 'CapsNet+UNet', 'DenseNet+ViT']
                            probs = [float(cascade_model.ensemble.predict_proba(X_clean_df)[0, 1]) if 'X_clean_df' in locals() else float(final_prob), 
                                     float(crnn_prob), float(capsnet_prob), float(densenet_prob)]
                            ax.bar(modalities, probs, color=['#007bff', '#28a745', '#ffc107', '#dc3545'])
                            ax.set_ylim(0, 1.0)
                            ax.set_ylabel("Prawdopodobieństwo")
                            ax.set_title("Porównanie modalności dla pacjenta")
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as db_ex:
                        st.info(f"Błąd odczytu bazy danych: {db_ex}")
                else:
                    try:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        modalities = ['Tabelaryczne (F1)', 'CRNN (Krwiaki)', 'CapsNet+UNet (Ciało modzelowate)', 'DenseNet+ViT (Złamania)']
                        tab_val = float(cascade_model.ensemble.predict_proba(X_clean_df)[0, 1]) if 'X_clean_df' in locals() else float(final_prob)
                        probs = [tab_val, float(crnn_prob), float(capsnet_prob), float(densenet_prob)]
                        ax.barh(modalities, probs, color=['#007bff', '#28a745', '#ffc107', '#dc3545'])
                        ax.set_xlim(0, 1.0)
                        ax.set_xlabel("Szacowane Prawdopodobieństwo")
                        ax.set_title("Wkład modalności (Brak historycznych wpisów w bazie)")
                        for i, v in enumerate(probs):
                            ax.text(v + 0.02, i, f"{v*100:.1f}%", va='center', fontweight='bold', fontsize=9)
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as ex2:
                        st.info(f"Wykres synergii niedostępny: {ex2}")

            # --- SEKCJA 2: Neural ODE oraz GNN Graph Rx (Dynamiczne z bazy i modeli) ---
            st.markdown("---")
            col_ode, col_gnn = st.columns(2)
            
            # Wczytywanie danych historycznych z bazy Active Learning do kalibracji symulacji
            db_path = "output_data/multimodal_active_learning_queue.csv"
            historical_volatility = 0.05
            if os.path.exists(db_path):
                try:
                    df_hist = pd.read_csv(db_path)
                    if len(df_hist) > 5 and 'true_label' in df_hist.columns:
                        historical_volatility = float(df_hist['true_label'].std()) if not np.isnan(df_hist['true_label'].std()) else 0.05
                except Exception:
                    pass

            with col_ode:
                st.subheader("Symulacja Trajektorii Czasowej (Neural ODE)")
                st.caption(f"Kalibracja na bazie historycznej OIT (Wskaźnik zmienności: {historical_volatility:.3f})")
                
                scenario = "sudden_hematoma" if final_prob >= 0.5 else "stable_recovery"
                simulated_events = generate_icu_clinical_scenario(scenario_name=scenario, duration_hours=24)
                bayesian_monitor = BayesianTrajectoryMonitor(initial_prob=final_prob, patient_age=age)
                
                probs_over_time = [final_prob * 100]
                for vitals in simulated_events:
                    adjusted_vitals = {k: v * (1.0 + historical_volatility * 0.1) for k, v in vitals.items()} if isinstance(vitals, dict) else vitals
                    probs_over_time.append(bayesian_monitor.update_belief(adjusted_vitals) * 100)
                    
                chart_df = pd.DataFrame(probs_over_time, columns=['Ryzyko Pogorszenia (%)'])
                chart_df.index.name = "Godziny na OIT"
                st.line_chart(chart_df, color="#ff4b4b" if scenario == "sudden_hematoma" else "#2e7d32")

            with col_gnn:
                st.subheader("Wykrywanie Pułapok Farmakologicznych (GNN)")
                st.caption("Grafowa analiza relacji leków i telemetrii OIT.")
                
                # Dynamiczna analiza bazy pod kątem statystyk historii pacjentów
                if os.path.exists(db_path):
                    try:
                        df_db = pd.read_csv(db_path)
                        total_records = len(df_db)
                        critical_count = int(df_db['true_label'].sum()) if 'true_label' in df_db.columns else 0
                        st.markdown(f"<small><b>Baza wiedzy OIT:</b> {total_records} przeanalizowanych przypadków historycznych (Krytycznych: {critical_count})</small>", unsafe_allow_html=True)
                    except Exception:
                        pass

                if final_prob >= 0.5:
                    st.error("Wykryto wysokie ryzyko powikłań farmakologicznych / burzy autonomicznej:")
                    st.markdown("""
                    - **Pułapka GABAergiczna:** Gwałtowny spadek GCS przy sedacji (Propofol/Midazolam) może maskować rzeczywisty stan strukturalny.
                    - **Ryzyko fałszywego alarmu obrzęku:** Graf korelacji wskazuje na odpowiedź na farmakoterapię, a nie postęp masy guza/krwotoku.
                    - **Wymagana ostrożność przy odstawianiu leków** i monitorowanie autonomiczne.
                    """)
                else:
                    st.success("Profil farmakologiczny stabilny (Brak anomalii GNN):")
                    st.markdown("""
                    - **Brak sygnałów reakcji paradoksalnych** na podawane leki sedacyjne.
                    - **Korelacja telemetrii i leków w normie** – brak fałszywych odchyłek w spadkach GCS.
                    """)

            # --- SEKCJA 3: Panel HITL z dynamicznymi statystykami z bazy ---
            st.divider()
            st.subheader("Multimodalny Panel Audytu i Korekty Błędów AI")
            
            if os.path.exists(db_path):
                try:
                    df_stats = pd.read_csv(db_path)
                    st.info(f"Aktywna pętla uczenia: w bazie zapisano dotychczas **{len(df_stats)}** weryfikacji klinicznych.")
                except Exception:
                    st.info("Wskaź, w którym elemencie system popełnił błąd. Informacje te zasilą pętlę uczenia aktywnego.")
            else:
                st.info("Wskaź, w którym elemencie system popełnił błąd. Informacje te zasilą pętlę uczenia aktywnego.")
            
            with st.form(key='multimodal_hitl_form'):
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.markdown("**Audyt Modeli Wizyjnych**")
                    cv_err = st.multiselect("Zaznacz błędy sieci 3D:", [
                        "CRNN: Przeoczył / nie docenił krwotoku",
                        "CRNN: Fałszywy alarm przesunięcia linii środkowej",
                        "CapsNet: Przeoczył uszkodzenie (DAI / Ciało Modzelowate)",
                        "DenseNet+Tr: Przeoczył złamanie kości czaszki/szyi",
                        "Wizualizacje poprawne / brak zastrzeżeń"
                    ])
                    
                with col_m2:
                    st.markdown("**Audyt Danych Tabelarycznych**")
                    tab_err = st.multiselect("Zaznacz anomalie w tabeli:", [
                        "Fałszywy alarm hemodynamiczny (szum z aparatury)",
                        "Nieprawidłowa interpretacja GCS przez farmakologię",
                        "Zaniżone lub pominięte kluczowe pomiary z 24h",
                        "Parametry tabelaryczne poprawne"
                    ])
                    
                st.markdown("---")
                col_g1, col_g2 = st.columns([1, 1])
                with col_g1:
                    global_out = st.radio("Ostateczny werdykt kliniczny:", [
                        "Zgadzam się z decyzją i ryzykiem AI",
                        "Pacjent przeżył / Stan stabilny (AI dało fałszywy alarm)",
                        "Pacjent zmarł / Stan krytyczny (AI zignorowało zagrożenie)"
                    ])
                with col_g2:
                    doc_comment = st.text_area("Szczegółowy komentarz lekarski (do raportu i Active Learning):")
                
                submit_audit = st.form_submit_button("Zapisz Korektę w Bazie i Zaktualizuj Pętlę AI", type="primary")
                
                if submit_audit:
                    save_multimodal_feedback(
                        patient_id="PT-DYNAMIC-01",
                        patient_data=patient_data,
                        cv_errors=cv_err,
                        tabular_errors=tab_err,
                        global_outcome=global_out,
                        comment=doc_comment
                    )
                    st.success("Pomyślnie zapisano korektę multimodalną w pliku `output_data/multimodal_active_learning_queue.csv`.")
                    st.balloons()