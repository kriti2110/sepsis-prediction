from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np

PARAM_GRIDS = {
    "xgboost": {
        "n_estimators":    [100, 200, 300, 500],
        "max_depth":       [3, 5, 6, 8],
        "learning_rate":   [0.01, 0.05, 0.1],
        "subsample":       [0.6, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.8, 1.0],
        "scale_pos_weight":[3, 5, 7, 10],
    },
    "lightgbm": {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [4, 6, 8, -1],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves":    [31, 63, 127],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth":    [5, 8, 10, None],
        "min_samples_split": [2, 5, 10],
    },
}

def tune_model(model, model_type, X_train, y_train, n_iter=30, cv=5, scoring="roc_auc"):
    param_grid = PARAM_GRIDS.get(model_type, {})
    search = RandomizedSearchCV(
        model, param_grid, n_iter=n_iter, cv=cv,
        scoring=scoring, n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train, y_train)
    print(f"Best {scoring}: {search.best_score_:.4f}")
    print(f"Best params:   {search.best_params_}")
    return search.best_estimator_, search.best_params_
