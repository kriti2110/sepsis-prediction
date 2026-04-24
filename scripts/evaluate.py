import argparse, pandas as pd
from models.classifier import load_model
from utils.evaluation import print_report
from utils.visualization import plot_roc_curve, plot_confusion_matrix, plot_feature_importance
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",      type=str, required=True)
    p.add_argument("--labels",    type=str, required=True)
    p.add_argument("--model",     type=str, required=True)
    p.add_argument("--output",    type=str, default="outputs/plots")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    X = pd.read_csv(args.data)
    y = pd.read_csv(args.labels).squeeze()
    model = load_model(args.model)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    print_report(y, y_pred, y_prob)
    plot_roc_curve(y, y_prob, save_path=f"{args.output}/roc_curve.png")
    plot_confusion_matrix(y, y_pred, save_path=f"{args.output}/confusion_matrix.png")

if __name__ == "__main__":
    main()
