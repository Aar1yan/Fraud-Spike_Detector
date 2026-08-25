"""Final evaluation on the locked test set. Run exactly once, after model
selection and threshold selection are already finalized on val — no tuning
happens here, and this script does not feed back into train.py.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODELS_DIR / "model.joblib")
    with open(MODELS_DIR / "model_meta.json") as f:
        meta = json.load(f)

    threshold = meta["threshold"]
    feature_cols = meta["feature_cols"]

    test = pd.read_csv(PROCESSED_DIR / "test_scaled.csv")
    X_test, y_test = test[feature_cols], test["Class"]

    scores = model.predict_proba(X_test)[:, 1]
    preds = (scores >= threshold).astype(int)

    pr_auc = average_precision_score(y_test, scores)
    roc_auc = roc_auc_score(y_test, scores)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    report = {
        "model_name": meta["model_name"],
        "threshold": threshold,
        "test_pr_auc": pr_auc,
        "test_roc_auc": roc_auc,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "n_test_rows": len(test),
        "n_test_fraud": int(y_test.sum()),
    }

    with open(RESULTS_DIR / "test_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
