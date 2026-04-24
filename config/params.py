from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataConfig:
    target_col: str = "SepsisLabel"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    handle_imbalance: str = "smote"  # smote | class_weight | oversample

@dataclass
class FeatureConfig:
    vital_signs: List[str] = field(default_factory=lambda: [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"
    ])
    lab_values: List[str] = field(default_factory=lambda: [
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
        "Glucose", "Lactate", "Creatinine", "Bilirubin_total"
    ])
    drop_threshold: float = 0.4

@dataclass
class ModelConfig:
    model_type: str = "xgboost"  # xgboost | lightgbm | random_forest | logistic
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42
    save_dir: str = "outputs"
