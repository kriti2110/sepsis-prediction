# Feature Engineering Reference

## Engineered Features

| Feature | Formula | Clinical Significance |
|---------|---------|----------------------|
| `ShockIndex` | HR / SBP | >1.0 indicates shock risk |
| `RenalScore` | Creatinine bins [0-4] | SOFA renal sub-score proxy |
| `LiverScore` | Bilirubin bins [0-4] | SOFA hepatic sub-score proxy |
| `HR_Resp_ratio` | HR / Resp | Cardio-respiratory coupling |
| `Pulse_Pressure` | SBP − MAP | Vascular tone indicator |
| `EarlyICU` | ICULOS ≤ 6h | Binary: first 6 hours flag |

## Feature Selection
Columns with >40% missing values are dropped before training. Feature importance rankings (SHAP) are available after model training.
