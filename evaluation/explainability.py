import shap
import numpy as np
import matplotlib.pyplot as plt

def compute_shap_values(model, X, model_type="tree"):
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))
    shap_values = explainer.shap_values(X)
    return explainer, shap_values

def plot_shap_summary(shap_values, X, max_display=20, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def plot_shap_waterfall(explainer, shap_values, X, idx=0, save_path=None):
    shap.plots.waterfall(explainer.expected_value +
                         shap.Explanation(shap_values[idx], feature_names=X.columns.tolist()),
                         show=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

def get_top_features(shap_values, feature_names, top_n=10):
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]
    return [(feature_names[i], mean_abs_shap[i]) for i in top_idx]
