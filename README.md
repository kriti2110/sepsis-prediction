<div align="center">

<h1>🩺 Sepsis Prediction using Machine Learning</h1>

<p>A clinical machine learning pipeline for early sepsis detection from patient data — binary classification with explainability and production-ready evaluation.</p>

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

![Type](https://img.shields.io/badge/Task-Binary%20Classification-6366f1?style=flat-square)
![Metric](https://img.shields.io/badge/Metric-Accuracy-2ea44f?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Clinical%20ML-e11d48?style=flat-square)

</div>

---

## Overview

Sepsis is a life-threatening condition caused by the body's extreme response to infection. Early prediction is critical — this project builds a supervised classification model that identifies patients at risk of sepsis from structured clinical data, enabling timely intervention.

| Property | Details |
|----------|---------|
| ![](https://img.shields.io/badge/Problem-e11d48?style=flat-square) | Binary classification: sepsis-positive vs sepsis-negative |
| ![](https://img.shields.io/badge/Input-0ea5e9?style=flat-square) | Patient vitals, lab results, and clinical indicators |
| ![](https://img.shields.io/badge/Output-16a34a?style=flat-square) | Sepsis risk probability + binary label |
| ![](https://img.shields.io/badge/Primary%20Metric-f59e0b?style=flat-square) | Accuracy (+ Precision, Recall, F1, AUC-ROC) |

---

## Model Pipeline

```
Raw Clinical Data
      │
      ▼
┌─────────────────────┐
│   Preprocessing     │  ← missing value imputation, outlier handling
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Engineering│  ← normalization, encoding, feature selection
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model Training     │  ← classifier trained on labeled patient records
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Evaluation         │  ← Accuracy, Confusion Matrix, AUC-ROC curve
└────────┬────────────┘
         │
         ▼
   Sepsis Risk Score
```

### Training

- Supervised learning on labeled clinical records (sepsis / no sepsis)
- Cross-validation to prevent overfitting on imbalanced medical data
- Hyperparameter tuning via grid search

### Prediction

- Takes a patient's clinical feature vector as input
- Outputs a binary label and a continuous risk probability score
- Supports batch inference for hospital-scale datasets

---

## Evaluation

> **Primary metric:** ![Accuracy](https://img.shields.io/badge/Accuracy-2ea44f?style=flat-square)

**Full evaluation suite:**

![Accuracy](https://img.shields.io/badge/Accuracy-2ea44f?style=flat-square)
![Precision](https://img.shields.io/badge/Precision-3b82f6?style=flat-square)
![Recall](https://img.shields.io/badge/Recall-f97316?style=flat-square)
![F1 Score](https://img.shields.io/badge/F1%20Score-8b5cf6?style=flat-square)
![AUC--ROC](https://img.shields.io/badge/AUC--ROC-e11d48?style=flat-square)

---

## Repository Structure

```
sepsis-prediction/
├── sepsis_model.ipynb    # Full pipeline: EDA, preprocessing, training, evaluation
└── README.md
```

---

## Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=for-the-badge)

</div>

---

## Author

<div align="center">

**Kriti Raj** — B.Tech CSE (AI/ML), KIIT University

[![GitHub](https://img.shields.io/badge/GitHub-kriti2110-181717?style=flat-square&logo=github)](https://github.com/kriti2110)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Kriti%20Raj-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/kriti-raj-5b398236a)

</div>
