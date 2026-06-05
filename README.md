<div align="center">

<h1>🩺 NEO SEPSIS — Early Neonatal Sepsis Detection</h1>

<p>End-to-end explainable ML pipeline predicting neonatal sepsis <strong>6 hours before clinical onset</strong> — XGBoost + LSTM ensemble on 40,000+ NICU time-series records from MIMIC-III.</p>

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AD3?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

![AUROC](https://img.shields.io/badge/AUROC-0.921-2ea44f?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-40K%2B%20NICU%20Records-e11d48?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20AI-6366f1?style=flat-square)

</div>

---

## Overview

Neonatal sepsis is a leading cause of mortality in ICU settings. Early detection is hindered by severe class imbalance, noisy time-series data, and the need for clinically interpretable predictions. This project builds an explainable ensemble framework that predicts sepsis onset 6 hours ahead of clinical diagnosis.

| Property | Details |
|----------|---------|
| ![](https://img.shields.io/badge/Dataset-e11d48?style=flat-square) | MIMIC-III — 40,000+ NICU patient time-series records |
| ![](https://img.shields.io/badge/Task-0ea5e9?style=flat-square) | Binary classification: sepsis onset within 6h window |
| ![](https://img.shields.io/badge/Output-16a34a?style=flat-square) | Risk probability + SHAP/LIME per-patient explanation |
| ![](https://img.shields.io/badge/AUROC-f59e0b?style=flat-square) | **0.921** — comparable to specialist-physician performance |

---

## Model Pipeline

```
MIMIC-III NICU Time-Series (40K+ records)
      │
      ▼
┌──────────────────────────┐
│   Preprocessing          │  ← layered imputation, outlier handling,
│                          │    temporal feature extraction
└────────────┬─────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌───────────────┐
│ XGBoost  │  │  LSTM (PyTorch)│
│ Tabular  │  │  Time-series  │
│ Lab Feats│  │  Temporal Decay│
└────┬─────┘  └──────┬────────┘
     └────────┬───────┘
              ▼
     ┌─────────────────┐
     │  Ensemble Layer │  ← probability fusion
     └────────┬────────┘
              ▼
     ┌─────────────────┐
     │  SHAP / LIME    │  ← per-patient clinical explanations
     └────────┬────────┘
              ▼
       Sepsis Risk Score
       + Feature Attributions
```

### Key Design Decisions

- **LSTM** captures temporal decay patterns in vitals and lab trends over time
- **XGBoost** handles tabular lab features (WBC, CRP, lactate, etc.) with non-linear interactions
- **SMOTE** applied to address severe class imbalance in NICU sepsis labels
- **SHAP + LIME** provide feature-level explanations for each patient prediction — critical for clinical trust

---

## Evaluation

> **Primary metric:** AUROC — chosen over accuracy due to severe class imbalance in clinical data

![AUROC](https://img.shields.io/badge/AUROC-0.921-2ea44f?style=flat-square)
![Precision](https://img.shields.io/badge/Precision-3b82f6?style=flat-square)
![Recall](https://img.shields.io/badge/Recall-f97316?style=flat-square)
![F1 Score](https://img.shields.io/badge/F1%20Score-8b5cf6?style=flat-square)

AUROC **0.921** is comparable to specialist-physician performance on the held-out test set.

---

## Repository Structure

```
sepsis-prediction/
├── sepsis_model.ipynb    # Full pipeline: EDA, preprocessing, training, evaluation
├── training/             # Model training scripts
├── models/               # Saved model artifacts
├── evaluation/           # Metrics, curves, SHAP plots
├── utils/                # Feature engineering, SMOTE, imputation helpers
├── scripts/              # Data processing and inference scripts
├── config/               # Hyperparameter configs
├── data/                 # Data loading utilities
├── docs/                 # Architecture and methodology notes
├── requirements.txt
└── README.md
```

---

## Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AD3?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

</div>

---

## Author

<div align="center">

**Kriti Raj** — B.Tech CSE (AI/ML), KIIT University

[![GitHub](https://img.shields.io/badge/GitHub-kriti2110-181717?style=flat-square&logo=github)](https://github.com/kriti2110)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Kriti%20Raj-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/kriti-raj-5b398236a)

</div>
