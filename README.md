# Multimodal TBI (Traumatic Brain Injury) Analysis System

An advanced multimodal machine learning architecture designed for comprehensive analysis, prognosis, and deterioration prediction in Traumatic Brain Injury patients. The system utilizes a hybrid approach, seamlessly integrating computer vision for medical imaging, structured electronic health records, and graph based neural networks for pharmacological interactions and temporal patient state tracking.
The project is optimized for local execution and utilizes MLflow for robust experiment tracking and logging.

## System Architecture

The system processes multimodal patient data through three specialized branches, which are then fused to output a holistic prediction regarding patient deterioration and outcomes.

### Vision Module - Brain Scan Analysis
Responsible for the baseline visual recognition and feature extraction from medical imaging (CT/MRI scans). This deep learning backbone identifies spatial anomalies, lesions, and critical visual indicators of brain trauma, converting raw pixel/voxel data into high-dimensional semantic embeddings.

### Tabular Module - Hospital EHR Data
Processes structured clinical data sourced from the hospital's Electronic Health Records. This includes patient demographics, vital signs, laboratory results, and historical medical data. The module identifies subtle statistical patterns in raw numerical and categorical hospital data.

### Graph Module - Pharmacology & Deterioration
Models the complex relationships between administered medications and patient outcomes over time. 
* **Drug Graphs:** Analyzes pharmacological interactions and contraindications using node/edge representations.
* **Deterioration Tracking:** Captures the temporal dynamics of the patient's state, predicting the trajectory of clinical deterioration based on sequential graph structures.

### Multimodal Fusion & Results Analysis
The final aggregation module concatenates the embeddings from the Vision, Tabular, and Graph branches. A unified prediction head processes these combined vectors to generate the final patient outcome and deterioration risk score.

## Key Features

* **Holistic Multimodal Fusion:** The architecture intelligently weights information from scans, clinical history, and drug interactions, mimicking a multidisciplinary medical consultation.
* **Temporal Deterioration Graphs:** Leverages graph structures to visualize and compute the risk of abrupt clinical deterioration over a timeline.
* **Explainable AI:** Ensures that the predictions are interpretable for clinical professionals, highlighting the most critical brain scan regions, key laboratory deviations, and risky drug interactions that contributed to the final output.
* **Automated Pipeline Orchestration:** A built in orchestrator script ensures that data processing, individual module training, and late stage fusion are executed sequentially and safely.

## Environment and Execution

The project is configured to run directly on a local machine in an environment with a CUDA enabled GPU for efficient processing of medical images and graphs.

### Training the Modules

Training the entire architecture can be executed using the main orchestrator script. This script sequentially triggers the vision, tabular, graph, and fusion models, ensuring proper feature extraction at every step.

```bash
python run_all.py
