from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import joblib, os

def build_model(model_type="xgboost", **kwargs):
    models = {
        "xgboost": XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 300),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.05),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            scale_pos_weight=kwargs.get("scale_pos_weight", 5),
            eval_metric="logloss",
            random_state=42,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=kwargs.get("n_estimators", 300),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.05),
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 8),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "logistic": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models)}")
    return models[model_type]

def save_model(model, path="outputs/model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    return joblib.load(path)
