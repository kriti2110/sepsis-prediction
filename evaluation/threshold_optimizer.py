import numpy as np
from sklearn.metrics import f1_score

def find_optimal_threshold(y_true, y_prob, metric="f1", step=0.01):
    thresholds = np.arange(0.1, 0.9, step)
    best_thresh, best_score = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0) if metric == "f1" else None
        if score and score > best_score:
            best_score, best_thresh = score, t
    print(f"Optimal threshold: {best_thresh:.2f} (F1 = {best_score:.4f})")
    return best_thresh, best_score
