# Model Card — Sepsis Prediction Classifier

## Model Details
- **Type:** Binary classification (sepsis / no sepsis)
- **Algorithm:** XGBoost (default) / LightGBM / Random Forest / Logistic Regression
- **Input:** Hourly ICU clinical measurements (vitals + lab values)
- **Output:** Sepsis probability score [0, 1] + binary label

## Intended Use
- Research and prototyping for early sepsis detection
- NOT intended for direct clinical decision-making without further validation

## Performance (Expected Range)
| Metric | Expected |
|--------|---------|
| Accuracy | 0.88–0.93 |
| AUC-ROC | 0.85–0.92 |
| Sensitivity | 0.75–0.85 |
| Specificity | 0.90–0.96 |

## Limitations
- Performance highly dependent on data quality and imputation strategy
- Class imbalance requires careful handling (SMOTE recommended)
- Not validated on non-ICU populations
