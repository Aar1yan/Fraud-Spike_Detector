# Fraud Spike Detector

A real-time transaction risk scoring system that flags potentially fraudulent card-not-present transactions, with per-prediction explainability built in.

## Live demo

**[fraud-spikedetector-ga4pltdeqctyzsxecsh5gb.streamlit.app](https://fraud-spikedetector-ga4pltdeqctyzsxecsh5gb.streamlit.app)**

The demo scores real, held-out test-set transactions — pick one from the curated dropdown, or pull a random one from the full test set, and see the model's risk score, flag, and SHAP-based explanation for that specific prediction.

## Problem

Card-not-present fraud — transactions made without the physical card present, e.g. online or over the phone — is one of the hardest fraud patterns to catch, because there's no chip, PIN, or signature to check against. Left undetected, it costs money directly (chargebacks, reimbursed losses), erodes customer trust in the platform, and creates ongoing operational cost for manual review teams. Catching it early, with enough confidence to act on, is the goal of this project.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle, `mlg-ulb/creditcardfraud`) — real, anonymized transactions from European cardholders, September 2013.

- 284,807 transactions, 492 confirmed fraud cases (**0.17% positive rate**)
- Features `V1`–`V28` are PCA-transformed components (see [Known limitation](#known-limitation-anonymized-features) below); `Time` and `Amount` are the only two features on their original scale

The raw CSV (~144MB) isn't checked into this repo — it's too large for git. Download it from the Kaggle link above and place it at `data/raw/creditcard.csv` before running the pipeline.

## Approach

### 1. Train/val/test split — 60/20/20, not 70/15/15

With only 492 fraud rows in the *entire* dataset, a standard 70/15/15 split would put roughly 74 fraud rows in each of val and test — few enough that flipping 3–4 cases from caught to missed would swing recall by several points on noise alone. 60/20/20 pushes that to ~98–99 fraud rows per split, giving meaningfully more stable metrics, at effectively no cost to training data (170k rows is far more than this model needs). The split is stratified on `Class` and created once, with a fixed random seed — the test set is touched exactly once, for final evaluation.

### 2. Feature scaling

Only `Time` and `Amount` are scaled (`StandardScaler`, fit on train only, applied to val/test). `V1`–`V28` arrive already PCA-transformed and pre-scaled, so they're left untouched.

### 3. Model comparison

Three model families were trained and compared on validation PR-AUC:

| Model | Val PR-AUC |
|---|---|
| Logistic Regression | 0.686 |
| **Random Forest** | **0.799** ← selected |
| HistGradientBoosting | 0.684 |

**Why PR-AUC and not accuracy or ROC-AUC:** at a 0.17% fraud rate, a classifier that predicts "not fraud" for every single transaction scores ~99.83% accuracy — so accuracy can't tell a good fraud model from a useless one. ROC-AUC is better but still overstates performance under this much imbalance, since the false-positive *rate* stays tiny even when the false-positive *count* (relative to the fraud count) is large. PR-AUC is directly sensitive to how many of the model's fraud flags are actually correct, which is what matters operationally.

### 4. Imbalance handling

`class_weight="balanced"` was used instead of SMOTE or undersampling. With only ~295 fraud rows in train, synthetic oversampling risks fabricating unrealistic examples, and undersampling would throw away real, legitimate transaction data for no benefit — 284k legitimate examples aren't a data-volume problem. Class weighting keeps the real distribution intact and just reweights each class's contribution to the training loss.

### 5. Threshold selection

The decision threshold (**0.30**) was chosen by maximizing F1 on the validation set — not on test, and not by eyeballing a "reasonable" cutoff like 0.5.

### 6. Final test-set evaluation (locked, one-time)

Run exactly once, after the model and threshold were both finalized on train/val alone — no retuning afterward.

| Metric | Value |
|---|---|
| PR-AUC | 0.876 |
| ROC-AUC | 0.962 |
| Precision | 0.953 |
| Recall | 0.818 |
| F1 | 0.880 |

Confusion matrix (99 actual fraud cases in test, out of 56,962 transactions):

| | Predicted genuine | Predicted fraud |
|---|---|---|
| **Actual genuine** | 56,859 (TN) | 4 (FP) |
| **Actual fraud** | 18 (FN) | 81 (TP) |

The model catches roughly 4 out of 5 fraud cases, and when it flags something, it's right about 95% of the time.

## Explainability

Per-prediction feature importance is computed with SHAP's `TreeExplainer`, so every flagged transaction comes with a breakdown of which features pushed its score up or down — not just a single opaque number. Across a sample of the test set, the strongest global drivers were `V12`, `V14`, `V4`, `V10`, and `V3`.

## Known limitation: anonymized features

`V1`–`V28` are PCA components released by the dataset owner to protect customer privacy — the original transaction details (merchant type, location, purchase category, etc.) were mathematically transformed before this data was ever made public. That means the model, and SHAP, can accurately identify **which** components are unusual for a given transaction and by how much — but that can't be traced back to **what** those components originally represented, since that mapping was intentionally discarded for privacy.

This is a known, well-documented trade-off of working with privacy-protected real-world financial data, not a gap in the modeling. A production deployment using a business's own (non-anonymized) transaction data would restore full interpretability — explanations like "unusual purchase location" or "amount far above typical spending" instead of "V14 = -8.2."

## Project structure

```
fraud-spike-detector/
├── data/
│   ├── raw/            # creditcard.csv (download separately, see Dataset)
│   └── processed/      # train/val/test splits, raw and scaled
├── notebooks/
│   ├── 01_eda.ipynb     # EDA — data shape, class balance, split rationale
│   └── 03_tuning.ipynb  # model comparison, threshold selection, final test results
├── src/
│   ├── data_prep.py     # stratified 60/20/20 split
│   ├── features.py      # Time/Amount scaling
│   ├── train.py         # model comparison + selection + threshold
│   ├── evaluate.py      # locked, one-time test-set evaluation
│   └── explain.py       # SHAP feature importance
├── models/               # trained model, scaler, metadata (tracked in git — small)
├── results/              # evaluation metrics, SHAP outputs
└── app/
    └── streamlit_app.py  # the live demo
```

## How to run locally

```bash
git clone <this-repo-url>
cd fraud-spike-detector

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The trained model (`models/`) and the test-set files the app needs (`data/processed/test.csv`, `test_scaled.csv`) are already committed, so you can jump straight to running the app:

```bash
streamlit run app/streamlit_app.py
```

To reproduce the full pipeline from scratch instead (e.g. to retrain):

1. Download the dataset from Kaggle and place it at `data/raw/creditcard.csv`
2. Run the pipeline in order:
   ```bash
   python -m src.data_prep   # split raw data into train/val/test
   python -m src.features    # scale Time/Amount, save scaler
   python -m src.train       # compare models, select + save the best one
   python -m src.evaluate    # one-time locked test-set evaluation
   python -m src.explain     # SHAP feature importance
   ```
3. Run the app as above

## Tech stack

Python 3.9+, [pandas](https://pandas.pydata.org/) 2.3, [scikit-learn](https://scikit-learn.org/) 1.6, [SHAP](https://shap.readthedocs.io/) 0.49, [Streamlit](https://streamlit.io/) 1.50, [Plotly](https://plotly.com/python/) 6.9 (SHAP visualization in the app), matplotlib/seaborn (EDA).

## Notebooks

- **[`01_eda.ipynb`](notebooks/01_eda.ipynb)** — dataset load and shape, null check, class balance and why 0.17% fraud counts as extreme imbalance, `Time`/`Amount` distributions, the PCA-anonymization note, and the reasoning behind the 60/20/20 split.
- **[`03_tuning.ipynb`](notebooks/03_tuning.ipynb)** — why PR-AUC over accuracy/ROC-AUC, why class-weighting over SMOTE/undersampling, the three-model comparison, threshold selection, and the final locked test-set results.

Both notebooks are executed end-to-end and load their numbers from the actual saved artifacts (`models/model_meta.json`, `results/test_metrics.json`) rather than restating them by hand — so they stay accurate if the underlying model ever changes.
