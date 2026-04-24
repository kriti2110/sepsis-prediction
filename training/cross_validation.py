import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils.evaluation import evaluate_model

def stratified_kfold_cv(model_fn, X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = evaluate_model(y_val, y_pred, y_prob)
        fold_results.append(metrics)
        print(f"Fold {fold}/{n_splits} — Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    summary = {k: (np.mean([r[k] for r in fold_results]),
                   np.std([r[k]  for r in fold_results]))
               for k in fold_results[0]}
    print("\n── CV Summary ──────────────────────────")
    for k, (mean, std) in summary.items():
        print(f"  {k:12s}: {mean:.4f} ± {std:.4f}")
    return fold_results, summary
