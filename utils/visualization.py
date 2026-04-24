import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve

def plot_roc_curve(y_true, y_prob, title="ROC Curve", save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#e11d48", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Sepsis","Sepsis"], yticklabels=["No Sepsis","Sepsis"], ax=ax)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted"); ax.set_title("Confusion Matrix")
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([feature_names[i] for i in idx], importances[idx], color="#3b82f6")
    ax.set_title(f"Top {top_n} Feature Importances"); ax.grid(axis="x", alpha=0.3)
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
