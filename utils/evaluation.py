import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, average_precision_score)

def evaluate_model(y_true, y_pred, y_prob=None):
    results = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity_score(y_true, y_pred),
    }
    if y_prob is not None:
        results["auc_roc"] = roc_auc_score(y_true, y_prob)
        results["auc_pr"]  = average_precision_score(y_true, y_prob)
    return results

def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp + 1e-6)

def print_report(y_true, y_pred, y_prob=None):
    metrics = evaluate_model(y_true, y_pred, y_prob)
    print("\n── Evaluation Report ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Sepsis", "Sepsis"]))
