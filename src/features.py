"""Feature scaling. V1-V28 are already PCA components (pre-scaled); only
Time and Amount are on raw scales, so those are the only columns we transform.
The scaler is fit on train only and reused for val/test to avoid leakage.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
SCALE_COLS = ["Time", "Amount"]


def fit_scaler(train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train[SCALE_COLS])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    out[SCALE_COLS] = scaler.transform(out[SCALE_COLS])
    return out


def load_splits():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train, val, test


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = load_splits()

    scaler = fit_scaler(train)
    train_s = apply_scaler(train, scaler)
    val_s = apply_scaler(val, scaler)
    test_s = apply_scaler(test, scaler)

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    train_s.to_csv(PROCESSED_DIR / "train_scaled.csv", index=False)
    val_s.to_csv(PROCESSED_DIR / "val_scaled.csv", index=False)
    test_s.to_csv(PROCESSED_DIR / "test_scaled.csv", index=False)
    print("scaler fit on train, applied to train/val/test, saved to models/scaler.joblib")


if __name__ == "__main__":
    main()
