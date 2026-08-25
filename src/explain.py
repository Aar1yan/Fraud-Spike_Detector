"""Feature importance via SHAP. IMPORTANT CAVEAT: V1-V28 are PCA components
released by the dataset owner with no disclosed meaning (confidentiality).
This can tell us *which* components drive a fraud score and by how much, but
NOT what real-world transaction attribute a component like V14 represents.
Any "why was this flagged" explanation is therefore limited to numeric
feature contributions, not a business-readable reason.
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
SAMPLE_SIZE = 2000
RANDOM_STATE = 42


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODELS_DIR / "model.joblib")
    with open(MODELS_DIR / "model_meta.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    test = pd.read_csv(PROCESSED_DIR / "test_scaled.csv")
    sample = test.sample(
        n=min(SAMPLE_SIZE, len(test)), random_state=RANDOM_STATE
    )
    X_sample = sample[feature_cols]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # binary classifier: take the "fraud" class contributions
    fraud_shap = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values[1]

    mean_abs_shap = pd.Series(
        abs(fraud_shap).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)
    mean_abs_shap.to_csv(RESULTS_DIR / "feature_importance.csv", header=["mean_abs_shap"])

    plt.figure()
    shap.summary_plot(fraud_shap, X_sample, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=150)
    plt.close()

    print("Top 10 features by mean |SHAP value|:")
    print(mean_abs_shap.head(10))
    print(f"\nSaved: {RESULTS_DIR/'feature_importance.csv'}, {RESULTS_DIR/'shap_summary.png'}")
    print(
        "\nNote: V1-V28 are anonymized PCA components with no disclosed real-world "
        "meaning, so this ranks WHICH components drive fraud scores, not WHAT "
        "business attribute they represent."
    )


if __name__ == "__main__":
    main()
