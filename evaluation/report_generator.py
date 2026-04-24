import json, os
from datetime import datetime
from utils.evaluation import evaluate_model

def generate_report(model_name, y_true, y_pred, y_prob, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    metrics = evaluate_model(y_true, y_pred, y_prob)
    report = {
        "model":     model_name,
        "timestamp": datetime.now().isoformat(),
        "metrics":   {k: round(v, 4) for k, v in metrics.items()},
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }
    path = os.path.join(save_dir, f"report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {path}")
    return report
