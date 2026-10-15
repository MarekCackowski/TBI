# TBI Clinical Decision Support System
**A multimodal artificial intelligence system for predicting mortality, herniation risk, and functional recovery in Traumatic Brain Injury patients in the Intensive Care Unit.**

> "Professor Jan Talar proved that there are no hopeless patients. Only those who were robbed of time and hope too early. Awakening from a coma is a slow process, not a single event. Assessing prognosis in this early phase is a trap, because we are examining an individual whose brain is in deep shock, operating in emergency mode, and has not yet fully recovered its function. Denying a person a chance based on a five point questionnaire is the ultimate capitulation to the complexity of the brain."

When Bryan Jennett and Graham Teasdale created the **Glasgow Coma Scale** in 1974, their goal was to standardize rapid communication between nurses and doctors. The creators themselves were horrified when, decades later, hospitals began using this 5 item, superficial questionnaire as the ultimate oracle for deciding whether to withdraw life support or deny treatment. 

GCS is blind to what happens inside the skull. This project is physical proof of how to fix that flaw. The system does not evaluate a patient based on a few external reflexes; it reads raw tomographic scans and tabular data, tracks continuous ICU telemetry, and untangles the deceptive webs of modern intensive care pharmacotherapy.

---

## System Architecture

The system was designed for a high risk environment where minimizing missed critical patients is an absolute priority. The architecture relies on **late multimodal fusion**, integrating data from 4 independent pipelines:

### Tabular Cascade & Phenotyping

Processing structured data from life support equipment and medical history through a multi stage defense line designed to overcome the rigidity of standard algorithms:

* **Data Quality Guardrails:** Automated ETL validation suite enforcing strict schema types, uniqueness checks, and physiological range assertions before data enters the persistence layer.
* **Tabular Cascade & Auto Noising (`RLAutoNoiseAgent`):** A reinforcement learning agent that dynamically injects structured noise during training. This forces the model to ignore subjective human scoring errors (such as inconsistent nursing entries for GCS or APACHE) and learn true underlying physiological fragility.

#### Phase 1: Asymmetric Tree Ensemble (Broad Screening)
Rejecting bagging in favor of gradient boosting algorithms that aggressively minimize residuals on critical cases. This phase combines distinct algorithms tailored for their complementary strengths:
* **XGBoost:** Provides precise residual regularization to prevent overfitting on noisy medical telemetry.
* **LightGBM:** Utilizes leaf wise growth optimized to capture non linear interactions across high dimensional features.
* **CatBoost:** Employs symmetrically built trees to completely prevent feature overfitting on severe minority mortality classes.
* **TabNet:** Applies sequential attention mechanisms over tabular features to mimic human feature selection and reasoning.

#### Phase 2: GMM Phenotype Router & Isolation Forest (Borderline Stratification)
Standard decision trees split feature space along strict right angles, which catastrophically fails for complex, borderline ICU patients whose multi system failures cannot be boxed into axis aligned rules. To solve this:
* **Isolation Forest:** First isolates out of distribution anomalies.
* **Gaussian Mixture Model:** Performs soft clustering to map patients into distinct physiological phenotypes based on multidimensional distance.

### Specialist Model Routing (Targeted Decision Making)
Rather than forcing a single architecture to handle everything, ambiguous and high-risk cases identified by the router are dynamically delegated to specialized mathematical paradigms:
- **HistGradientExpert:** Utilizes a `HistGradientBoostingClassifier` tuned by `GridSearchCV` to handle residual patterns and optimize decision boundaries on difficult feature subsets.
- **Non Linear SVM:** Projects complex borderline cases into higher dimensional spaces using a kernel based `SVC` (with balanced class weights and probability calibration) to find smooth geometric separation where standard trees break down.
- **Clinical Profile Expert (LogReg + KNN):** Combines calibrated linear risk weights (`LogisticRegression`) with the physiological distance to historical records (`KNeighborsClassifier`) through a soft voting ensemble.
- **Non Linear MLP:** Models dense, continuous interactions of metabolic collapse for patients in profound neurogenic shock using multi layer perceptrons optimized with `HalvingGridSearchCV`.

### Spatial Vision
Instead of relying solely on tabular data, the system directly processes raw CT scans, identifying structural pathologies that dictate long term functional recovery and pose the steepest rehabilitation challenges:

- **CRNN (Hemorrhage & Volumetric Mass Effect Cascades):** Sequential analysis of brain slices enhanced by a *Slice Attention* mechanism for multilabel classification of hemorrhages and tracking the 3D spatial dynamics of mass lesions and tumors.
  *Clinical Impact:* Lesions and blood collections act as expanding masses that rapidly escalate intracranial pressure and induce secondary ischemia. By decomposing volumetric analysis into 2D feature extraction coupled with recurrent Zaxis tracking, the CRNN bypasses 3D CNN bottlenecks while accurately mapping the three dimensional morphology and spatial expansion patterns of complex pathologies.

- **DenseNet 3D + ViT (Calvarial & Skull Base Fractures):** A network operating on the bone window combined with a Vision Transformer to instantly detect advanced calvarial and skull base fractures by capturing global spatial context across volumetric slices.
*Clinical Impact:* Beyond compromising structural skull integrity, basal fractures frequently tear the dura mater or impinge on cranial nerves and major cerebral vessels, risking catastrophic cerebrospinal fluid leaks, intracranial infections, and permanent sensory motor losses.

- **UNet + EfficientNet (Corpus Callosum & Diffuse Axonal Injury):** Corpus callosum extraction masking background noise, fed into an optimized EfficientNet backbone that extracts robust spatial features to detect microscopic diffuse axonal injuries invisible to standard CNNs.<br>
  *Clinical Impact:* As the massive white matter highway connecting both cerebral hemispheres, the corpus callosum is acutely vulnerable to mechanical shearing forces. Damage here permanently decouples interhemispheric communication, resulting in severe executive dysfunction and long term vegetative states representing some of the most stubborn and heartbreaking barriers in neurological rehabilitation.

### Holistic Judge & Dynamic Trajectory
Combining logits from all modalities into a final life saving decision:
-   **EBM & Logistic Regression:** Ensure decision transparency and rule extraction.
-   **Cost Sensitive Hard Mining:** Implementation of OHEM with a heavy penalty for unpredicted fatalities and unexpected survivors.
-   **Dynamic NPV Optimization:** Automated control of the classification threshold, driving **NPV up to ~95%**.
-   **Longitudinal Tracking:** Continuous monitoring using a **Bayesian Trajectory Monitor** (accounting for risk decay and sedation context) and **Neural ODE** (for continuous time physiological modeling).

---

### Domain Adaptation & Hospital Deployment
Deploying a model from a research database to a new hospital environment inherently risks *Domain Shift* caused by calibration differences in monitoring hardware, varying acquisition protocols, or local documentation habits. The system natively addresses this through **Meta Level Transfer Learning**:
-   **Core Backbone Freezing:** Heavy convolutional extractors and specialized vision weights remain locked, preserving robust anatomical feature representations.
-   **Local Judge Recalibration:** Utilizing a small local cohort from the target ICU, the system rapidly adapts its final Meta Judge layer to local measurement baselines, eliminating hardware biases in real time and restoring peak predictive calibration without requiring full model retraining.

## Clinical Pitfalls & Advanced Pharmacology Handling

### The GABAergic & Flumazenil Trap
Traditional machine learning models make catastrophic errors when ICU patients undergo complex pharmacotherapy. The system natively resolves the **paradoxical GABA reaction**:
1. In severe TBI, administering GABAergics (Propofol, Midazolam) can trigger extreme paradoxical agitation and a sharp drop in GCS.
2. Clinicians often administer Flumazenil (a GABA antagonist) to reverse the sedation; however, due to the extreme risk of precipitating status epilepticus in injured brains, it is strictly controlled.
3. A naive AI sees an agitated patient with a falling GCS and sounds a false alarm for rapidly progressing brain edema. By crossreferencing `inputevents` (drug infusion logs), the system filters out drug induced noise from true structural failure.

### Cross Modal Pharmacovigilance & Osmotherapy Dynamics
Beyond sedatives, the system models pharmacokinetic-pharmacodynamic interactions during aggressive neuro resuscitation:
- **Osmotherapy Tracking:** Dynamically evaluates patient response curves to Mannitol and Hypertonic Saline infusions, cross validating osmotic shifts against real time intracranial pressure telemetry.
- **Isolating Iatrogenic Artifacts:** Weighs drug half lives and infusion rates against spatial vision outputs to prevent false positive deterioration alerts during routine therapeutic paralysis or sedation weaning.

### The Brain Death Controversy & Autonomic Storms
The system bridges a critical gap in neurointensive care: distinguishing between **permanent brain death** and **profound neurogenic shock / autonomic storming (Paroxysmal Sympathetic Hyperactivity)**. 
- Early absence of brainstem reflexes or transient autonomic surges can lead to premature declarations of hopelessness. 
- By fusing autonomous storming indicators, respiratory drive records, and structural CT metrics, the system rejects snap judgments, upholding Professor Talar’s core doctrine: *recovering brains in deep shock require time, protection from premature life support withdrawal, and algorithmic patience.*

---

## Results & Evaluation

The system was trained and tested on a simulated, highly asymmetric ICU cohort (75% critical / 25% stable). In direct comparative stress tests against the traditional clinical heuristic ($GCS < 5$), the multimodal cascade demonstrated a significantly lower **False Omission Rate**, successfully intercepting patients with compensated mass effect who were overlooked by static scoring thresholds.

**Crucial Ethical Disclaimer:** It must be strongly emphasized that this project does not aim to prove the definitive superiority of this AI architecture over medical professionals or established clinical tools. Given the relatively small size of the dataset, outperforming traditional baselines here could simply be an artifact of statistical chance. Rather, the true purpose of this work is a moral provocation: to draw attention to what we are doing to living human beings. It is a technological plea to stop condemning patients with severe brain injuries based on superficial, rigid questionnaires, and to start treating the recovering brain with the patience it deserves.

Furthermore, this architecture is built strictly as a decision support tool. In the chaotic, high stress environment of the ICU, cognitive overload makes it perilously easy for exhausted clinicians to overlook subtle signs of stability, or worse, to fall into a "well of despair" where severe cases are reflexively deemed hopeless. To combat this, the system acts as a safety net. **Ultimately, this project advocates for a fundamental paradigm shift in medical AI: algorithms must never be used to calculate a human's "value," predict their "uselessness," or justify the withdrawal of care. Instead, their immense computational power must be harnessed to uncover hidden pathways to recovery, filtering out clinical noise so that doctors can focus on their true calling: healing and fighting for every human life.**

<!-- METRICS_START -->
<!-- METRICS_END -->

---

## Future Horizons & Advanced Extensions (With Richer Datasets)
With access to a comprehensive, multi center international clinical database, the system’s architecture is natively primed for several extensions:
- **High Frequency ICU Waveform Integration:** Incorporating raw, second by second arterial blood pressure and intracranial pressure waveforms to model **cerebral autoregulation (PRx index)**, enabling real time detection of micro vascular dysfunction before macroscopic tissue shifts occur.
- **Serial Longitudinal Imaging (Spatiotemporal Evolution):** Transitioning from baseline single scan predictions to tracking the dynamic expansion of contusions, progression of edema, or resorption of hemorrhages across serial follow up CT scans using spatiotemporal 3D transformers.
- **Multi Center Domain Generalization:** Implementing adversarial domain adaptation layers to ensure robust inference across heterogeneous hospital environments, differing scanner manufacturers, and varying acquisition protocols without performance degradation.
- **Serum Biomarker & Multi-Omics Fusion:** Integrating blood based brain injury biomarkers (e.g., GFAP, UCH-L1, neurofilaments) alongside genomic predisposition markers to quantify cellular level neurodegeneration and tailor individualized rehabilitation pathways.

---

## Tech Stack
-   **Deep Learning & Vision:** `PyTorch`, `TorchVision`, `torchdiffeq`
-   **Machine Learning:** `Scikit-Learn`, `XGBoost`, `LightGBM`, `CatBoost`, `TabNet`, `InterpretML`, `SkopeRules`
-   **Data Quality & Testing:** Custom Data Quality Assurance Suite (`Great Expectations` inspired)
-   **Optimization & MLOps:** `Optuna`, `SHAP`, `Imbalanced-Learn`
-   **Databases & Cache:** `PostgreSQL`, `MongoDB`, `Redis`

---

## Initialization and Data Download

Downloading the main RSNA archive (~190 GB) via the Kaggle CLI inside Docker's virtual file system causes memory leaks and out of memory crashes.
The solution is to perform a one time native download directly on the host machine after clearing the database volumes. The orchestrator will automatically detect the downloaded file, rebuild the database from scratch, and process the remaining datasets.

**The following command executes these steps in a single run:**
1. Performs a Docker hard reset and removes old database data (`down -v`).
2. Prepares the required directory structure.
3. Downloads the RSNA archive natively, bypassing container memory overhead.
4. Starts the orchestrator (`run_all.py`).

**Prerequisites:**
1. Ensure your Kaggle API credentials (`kaggle.json`) are placed in `~/.kaggle/kaggle.json`.
2. Ensure your project's `.env` file contains your configuration and required API keys/credentials.

**Copy and run the following in the main project directory:**

```bash
cd ~/TBI && \
sudo apt update && sudo apt install aria2 unzip -y && \
docker-compose down && \
rm -rf output_data/Models/* 2>/dev/null || true && \
mkdir -p ~/TBI/Data/rsna_raw && \
sudo chown -R $USER:$USER ~/TBI/Data/rsna_raw && \
if [ ! -f ~/TBI/Data/rsna_raw/stage_2_train.zip ] || [ ! -f ~/TBI/Data/rsna_raw/stage_2_train.csv ]; then \
    echo "Brak plików RSNA. Rozpoczynam pobieranie."; \
    URL=$(python3 -c '
import json, os, requests
with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
    c = json.load(f)
r = requests.get("[https://www.kaggle.com/api/v1/competitions/data/download-all/rsna-intracranial-hemorrhage-detection](https://www.kaggle.com/api/v1/competitions/data/download-all/rsna-intracranial-hemorrhage-detection)", auth=(c["username"], c["key"]), allow_redirects=False)
print(r.headers["Location"])
    ') && \
    aria2c -x 16 -s 16 -c -d ~/TBI/Data/rsna_raw -o stage_2_train.zip "$URL" && \
    unzip -o -j ~/TBI/Data/rsna_raw/stage_2_train.zip "*stage_2_train.csv" -d ~/TBI/Data/rsna_raw/; \
else \
    echo "Pliki RSNA już istnieją. Pomijam pobieranie."; \
fi && \
docker-compose build --no-cache tbi_app && \
docker-compose up -d tbi_mongo && \
sleep 3 && \
docker-compose run --rm tbi_app python -u App/RunAll.py
