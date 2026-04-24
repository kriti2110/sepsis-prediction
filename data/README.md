# Dataset Guide

## PhysioNet Challenge 2019 (Recommended)
The primary dataset used for this project. Contains ICU patient records with hourly clinical measurements.

**Download:** https://physionet.org/content/challenge-2019/1.0.0/

**Structure after placement:**
```
data/
├── training/
│   ├── training_setA/   ← ~20,000 PSV files
│   └── training_setB/   ← ~20,000 PSV files
└── README.md
```

**File format:** Pipe-separated values (PSV), one file per patient.

## Feature Columns

| Category | Features |
|----------|---------|
| Vital Signs | HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2 |
| Lab Values | BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, Glucose, Lactate |
| Demographics | Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS |
| **Target** | **SepsisLabel** (0 = no sepsis, 1 = sepsis) |

## Class Distribution
Sepsis is rare even in ICU populations — expect significant class imbalance (~8–12% positive rate). Use `--smote` flag in the training script to handle this.
