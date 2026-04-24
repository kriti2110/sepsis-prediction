import argparse, os, json
from utils.preprocessing import load_data, drop_high_missing, impute_features, scale_features, split_features_target
from utils.feature_engineering import engineer_all
from models.classifier import build_model, save_model
from training.imbalance_handler import apply_smote
from training.cross_validation import stratified_kfold_cv
from utils.evaluation import print_report
from sklearn.model_selection import train_test_split

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str, required=True, help="Path to CSV dataset")
    p.add_argument("--model",      type=str, default="xgboost")
    p.add_argument("--output",     type=str, default="outputs")
    p.add_argument("--cv",         action="store_true", help="Run cross-validation")
    p.add_argument("--smote",      action="store_true", help="Apply SMOTE oversampling")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    df = load_data(args.data)
    df = drop_high_missing(df)
    df = engineer_all(df)
    X, y = split_features_target(df)
    X, _ = impute_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    if args.smote:
        X_train, y_train = apply_smote(X_train, y_train)

    if args.cv:
        stratified_kfold_cv(lambda: build_model(args.model), X_train, y_train)

    model = build_model(args.model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print_report(y_test, y_pred, y_prob)

    save_model(model, os.path.join(args.output, "model.joblib"))

if __name__ == "__main__":
    main()
