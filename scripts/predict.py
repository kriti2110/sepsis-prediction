import argparse, pandas as pd
from models.classifier import load_model
from utils.preprocessing import impute_features, scale_features
from utils.feature_engineering import engineer_all

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str, required=True)
    p.add_argument("--model",      type=str, required=True)
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--output",     type=str, default="predictions.csv")
    return p.parse_args()

def main():
    args = parse_args()
    df    = pd.read_csv(args.data)
    df    = engineer_all(df)
    X, _  = impute_features(df)
    model = load_model(args.model)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= args.threshold).astype(int)

    out = df.copy()
    out["sepsis_probability"] = probs.round(4)
    out["sepsis_prediction"]  = preds
    out["risk_level"] = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0],
                                labels=["Low", "Medium", "High"])
    out.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"Sepsis positive: {preds.sum()} / {len(preds)}")

if __name__ == "__main__":
    main()
