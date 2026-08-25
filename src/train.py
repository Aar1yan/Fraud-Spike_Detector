"""Train candidate models and select the best one + decision threshold using
validation data only. The test set is never touched here.
"""

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

# macOS Accelerate BLAS emits spurious "divide by zero"/"overflow" RuntimeWarnings
# from intermediate matmul steps in LogisticRegression's lbfgs solver; verified
# the fitted coefficients/intercept are finite and the solver converges (~42
# iters), so this is a known platform false-positive, not a real fit problem.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RANDOM_STATE = 42

FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"


def load_xy(split_name: str):
    df = pd.read_csv(PROCESSED_DIR / f"{split_name}_scaled.csv")
    return df[FEATURE_COLS], df[TARGET_COL]


def best_threshold(y_true, y_scores) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = f1[:-1].argmax()  # thresholds has one fewer element than precision/recall
    return float(thresholds[best_idx]), float(f1[best_idx])


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_xy("train")
    X_val, y_val = load_xy("val")

    # Imbalance handled via class weighting (not SMOTE/undersampling): with
    # only 295 fraud rows in train, synthetic oversampling risks manufacturing
    # unrealistic examples, and undersampling would throw away legitimate
    # transaction data the non-fraud class needs to be well characterized.
    # Class weighting keeps the real distribution and just reweights the loss.
    candidates = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        # class_weight is a native param here (sklearn >=1.2) — no need for
        # manual sample_weight or a scale_pos_weight-style workaround.
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    leaderboard = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_scores = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, val_scores)
        leaderboard[name] = pr_auc
        fitted[name] = model
        print(f"{name}: val PR-AUC = {pr_auc:.4f}")

    best_name = max(leaderboard, key=leaderboard.get)
    best_model = fitted[best_name]
    val_scores = best_model.predict_proba(X_val)[:, 1]
    threshold, f1_at_threshold = best_threshold(y_val, val_scores)

    print(f"\nSelected model: {best_name} (val PR-AUC = {leaderboard[best_name]:.4f})")
    print(f"Selected threshold: {threshold:.4f} (val F1 = {f1_at_threshold:.4f})")

    joblib.dump(best_model, MODELS_DIR / "model.joblib")
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump(
            {
                "model_name": best_name,
                "threshold": threshold,
                "val_pr_auc": leaderboard[best_name],
                "val_f1_at_threshold": f1_at_threshold,
                "leaderboard": leaderboard,
                "feature_cols": FEATURE_COLS,
            },
            f,
            indent=2,
        )
    print(f"\nSaved model -> {MODELS_DIR/'model.joblib'}, metadata -> {MODELS_DIR/'model_meta.json'}")


if __name__ == "__main__":
    main()
