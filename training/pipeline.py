from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from models.classifier import build_model

def build_pipeline(model_type="xgboost", **model_kwargs):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
        ("model",   build_model(model_type, **model_kwargs)),
    ])
