"""Load the raw dataset and produce a single, locked stratified train/val/test split."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DIR = Path("data/processed")
RANDOM_STATE = 42


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert df.isnull().sum().sum() == 0, "unexpected missing values in raw data"
    return df


def split(df: pd.DataFrame):
    # 60/20/20 train/val/test, stratified on Class so each split keeps the
    # same ~0.173% fraud rate. This split is created once and never redone.
    train, temp = train_test_split(
        df, test_size=0.4, stratify=df["Class"], random_state=RANDOM_STATE
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp["Class"], random_state=RANDOM_STATE
    )
    return train, val, test


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    train, val, test = split(df)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        n_fraud = int(part["Class"].sum())
        print(f"{name}: {len(part)} rows, {n_fraud} fraud ({n_fraud / len(part):.4%})")


if __name__ == "__main__":
    main()
