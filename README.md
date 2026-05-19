# Traumatic Brain Injury Risk-Stratification System

## Project Overview
This repository contains a production-ready **Multimodal Decision Cascade & Stacking System** designed for real-time triage of patients with severe Traumatic Brain Injury (TBI). 

Instead of relying on a single monolithic model, the system introduces a **two-level clinical hierarchy**:
1. **Level 1 (The Consilium):** A Domain-Specific `StackingClassifier` that processes features divided into independent clinical areas: Hemodynamic, Neurological, and Geriatric.
2. **Level 2 (The Cascade Judge):** An automated routing system that identifies highly ambiguous or critical cases ($GCS \le 8$) and invokes a calibrated Support Vector Machine (SVM) specialized in medical noise and edge cases.

The system optimizes for the **$F_{1.5}$-Score** to prioritize clinical Sensitivity (Recall) and minimize the risk of missing critical patients.

---

## Architecture & Data Pipeline

The pipeline extracts and processes data from a PostgreSQL database through the following stages:

```text
   RAW PATIENT DATA (PostgreSQL)
                    │
                    ▼
   AUTOMATED PIPELINE (MICE)
   Handles missing values via Iterative MICE 
   Adds adaptive Gaussian noise to clinical scales
                    │
                    ▼
   STACKING
   ├── Internist (LightGBM)   ─► Hemodynamic features
   ├── Neurolog (RandomForest)─► Neurological features
   └── Geriatrist (XGBoost)   ─► Geriatric features
                    │
                    ▼
   META-LEARNER (Logistic Regression)
   Trained OOF data via SWA
                    │
                    ▼
   CASCADE ROUTING
   Is the case ambiguous? (GCS <= 8 OR Prob ≈ 0.5)
          ├── NO ──► Use Consilium Prediction
          │
          └── YES ─► ACTIVATE SPECIALIST (SVM + SMOTE)
                        │
                        ▼
               FINAL BLENDING JUDGE
           (Calibrated Logistic Regression)
