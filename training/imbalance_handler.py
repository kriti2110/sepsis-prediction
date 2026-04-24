from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np

def apply_smote(X_train, y_train, random_state=42):
    smote = SMOTE(sampling_strategy="minority", random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE: {y_train.sum()} → {y_res.sum()} minority class samples")
    return X_res, y_res

def apply_adasyn(X_train, y_train, random_state=42):
    adasyn = ADASYN(sampling_strategy="minority", random_state=random_state)
    X_res, y_res = adasyn.fit_resample(X_train, y_train)
    print(f"ADASYN: {y_train.sum()} → {y_res.sum()} minority class samples")
    return X_res, y_res

def apply_combined(X_train, y_train, over_ratio=0.3, under_ratio=0.7, random_state=42):
    pipeline = ImbPipeline([
        ("over",  SMOTE(sampling_strategy=over_ratio, random_state=random_state)),
        ("under", RandomUnderSampler(sampling_strategy=under_ratio, random_state=random_state)),
    ])
    X_res, y_res = pipeline.fit_resample(X_train, y_train)
    return X_res, y_res
