"""Demo UI for the fraud-spike detector. Loads the already-trained model and
scaler from models/, runs them on a handful of real held-out test-set
transactions, and shows the risk score + SHAP-based feature drivers for
whichever one the user picks. No training or tuning happens here.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features import SCALE_COLS, apply_scaler  # noqa: E402

MODELS_DIR = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
N_HIGH_FRAUD = 8
N_HIGH_GENUINE = 12
N_BORDERLINE = 8
HIGH_FRAUD_SCORE_MIN = 0.7
HIGH_GENUINE_SCORE_MAX = 0.02
SAMPLE_RANDOM_STATE = 42


@st.cache_resource
def load_model_bundle():
    model = joblib.load(MODELS_DIR / "model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "model_meta.json") as f:
        meta = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, scaler, meta, explainer


@st.cache_data
def load_full_test_set():
    return pd.read_csv(PROCESSED_DIR / "test.csv")


@st.cache_data
def load_full_test_set_scaled():
    return pd.read_csv(PROCESSED_DIR / "test_scaled.csv")


@st.cache_data
def build_curated_sample(_model, feature_cols, threshold):
    """A ~28-row curated pool spanning three buckets — high-confidence fraud,
    high-confidence genuine, and borderline/medium-risk cases near the
    decision threshold — so the demo isn't just a handful of easy calls.
    """
    test_raw = load_full_test_set()
    test_scaled = load_full_test_set_scaled()
    scores = _model.predict_proba(test_scaled[feature_cols])[:, 1]

    pool = test_raw.copy()
    pool["_score"] = scores

    high_fraud = pool[
        (pool["Class"] == 1) & (pool["_score"] >= HIGH_FRAUD_SCORE_MIN)
    ].sample(n=N_HIGH_FRAUD, random_state=SAMPLE_RANDOM_STATE)
    high_genuine = pool[
        (pool["Class"] == 0) & (pool["_score"] < HIGH_GENUINE_SCORE_MAX)
    ].sample(n=N_HIGH_GENUINE, random_state=SAMPLE_RANDOM_STATE)
    borderline = pool[
        (pool["_score"] >= threshold / 2) & (pool["_score"] <= threshold * 2)
    ].sample(n=N_BORDERLINE, random_state=SAMPLE_RANDOM_STATE)

    sample = (
        pd.concat([high_fraud, high_genuine, borderline])
        .drop(columns="_score")
        .sample(frac=1, random_state=SAMPLE_RANDOM_STATE)
        .reset_index(drop=True)
    )
    sample.insert(0, "txn_id", [f"TXN-{i + 1:03d}" for i in range(len(sample))])
    return sample


def risk_flag(score: float, threshold: float) -> str:
    # Bucketed around the val-selected threshold: High = at/above threshold,
    # Medium = the half-threshold band below it, Low = well below.
    if score >= threshold:
        return "High"
    if score >= threshold / 2:
        return "Medium"
    return "Low"


FLAG_COLORS = {
    "High": {"fg": "#ef4444", "bg": "rgba(239, 68, 68, 0.16)", "icon": "\U0001F534"},
    "Medium": {"fg": "#f59e0b", "bg": "rgba(245, 158, 11, 0.16)", "icon": "\U0001F7E1"},
    "Low": {"fg": "#22c55e", "bg": "rgba(34, 197, 94, 0.16)", "icon": "\U0001F7E2"},
}


def format_score(score: float) -> str:
    # A flat "%.3f" reads as "0.000" for genuinely tiny (but non-zero)
    # probabilities. Scale up precision as the score shrinks so it stays
    # legible instead of looking broken/zeroed-out.
    pct = score * 100
    if score == 0:
        return "0.00%"
    if pct < 0.01:
        return f"{pct:.4f}%"
    if pct < 1:
        return f"{pct:.3f}%"
    return f"{pct:.2f}%"


def render_score_panel(score: float, flag: str):
    colors = FLAG_COLORS[flag]
    pct_display = format_score(score)
    bar_pct = min(max(score * 100, 0), 100)
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between; gap:1rem;">
            <div>
                <div style="font-size:0.85rem; color:#9ca3af; letter-spacing:.04em; text-transform:uppercase;">Risk score</div>
                <div style="font-size:2.75rem; font-weight:700; color:{colors['fg']}; line-height:1.1;">{pct_display}</div>
            </div>
            <div style="background-color:{colors['bg']}; color:{colors['fg']}; padding:8px 20px; border-radius:999px; font-weight:700; font-size:1.1rem; white-space:nowrap;">
                {colors['icon']} {flag}
            </div>
        </div>
        <div style="background-color:rgba(255,255,255,0.08); border-radius:8px; height:14px; width:100%; overflow:hidden; margin-top:10px;">
            <div style="background-color:{colors['fg']}; width:{bar_pct}%; height:100%; border-radius:8px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Fraud Spike Detector", page_icon="🚨", layout="wide"
    )
    st.title("🚨 Fraud Spike Detector")
    st.caption("Live scoring demo over real, held-out test-set transactions.")

    model, scaler, meta, explainer = load_model_bundle()
    threshold = meta["threshold"]
    feature_cols = meta["feature_cols"]
    sample = build_curated_sample(model, feature_cols, threshold)
    full_test = load_full_test_set()

    if "active_source" not in st.session_state:
        st.session_state.active_source = "curated"
    if "random_row" not in st.session_state:
        st.session_state.random_row = None

    if st.button("🎲 Random transaction"):
        st.session_state.random_row = full_test.sample(n=1).iloc[0]
        st.session_state.active_source = "random"

    options = sample["txn_id"].tolist()
    labels = {
        row.txn_id: f"{row.txn_id} — Amount ${row.Amount:.2f}, Time {row.Time:.0f}s"
        for row in sample.itertuples()
    }
    st.caption(
        "These are real transactions from our held-out test set (not entered "
        "by you) — select one to see how the model scores it."
    )
    selected_id = st.selectbox(
        "Select a transaction", options, format_func=lambda x: labels[x]
    )
    if selected_id != st.session_state.get("_last_dropdown_id"):
        st.session_state.active_source = "curated"
        st.session_state["_last_dropdown_id"] = selected_id

    if st.session_state.active_source == "random" and st.session_state.random_row is not None:
        row = st.session_state.random_row
        st.caption(
            f"🎲 Randomly pulled from the full test set — Amount ${row.Amount:.2f}, "
            f"Time {row.Time:.0f}s"
        )
    else:
        row = sample[sample["txn_id"] == selected_id].iloc[0]

    # Apply the same scaler fit on train (src/features.py) so the model sees
    # features on the same scale it was trained on.
    row_df = pd.DataFrame([row[feature_cols]])
    row_scaled = apply_scaler(row_df, scaler)
    score = float(model.predict_proba(row_scaled[feature_cols])[0, 1])
    flag = risk_flag(score, threshold)

    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("Amount", f"${row.Amount:.2f}")
    col2.metric("Time (s)", f"{row.Time:.0f}")

    st.write("")
    render_score_panel(score, flag)

    st.divider()
    st.subheader("Top 5 contributing features (SHAP)")
    shap_values = explainer.shap_values(row_scaled[feature_cols])
    fraud_shap = (
        shap_values[0, :, 1] if shap_values.ndim == 3 else shap_values[1][0]
    )
    contrib = pd.Series(fraud_shap, index=feature_cols)
    # Sort by |SHAP value| so the strongest drivers (either direction) surface,
    # not just the largest positive push.
    top5 = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:5])
    raw_values = row[top5.index]

    bar_labels = [f"{feat} = {raw_values[feat]:.2f}" for feat in top5.index]
    bar_colors = ["#ef4444" if v > 0 else "#3b82f6" for v in top5.values]

    fig = go.Figure(
        go.Bar(
            x=top5.values[::-1],
            y=bar_labels[::-1],
            orientation="h",
            marker_color=bar_colors[::-1],
            text=[f"{v:+.3f}" for v in top5.values[::-1]],
            textposition="outside",
            hovertemplate="%{y}<br>SHAP value: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="SHAP value (push toward fraud →)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        margin=dict(l=10, r=30, t=10, b=40),
        height=320,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)", zerolinecolor="rgba(255,255,255,0.3)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "V1-V28 are anonymized PCA components — this shows statistical drivers "
        "behind the prediction, not plain-language business reasons."
    )
    with st.expander("Why can't I see what V1-V28 mean?"):
        st.write(
            "This dataset comes from real European bank transactions. To protect "
            "customer privacy, the bank mathematically transformed the original "
            "transaction details (like merchant type, location, or purchase "
            "category) into anonymized numerical components (V1 through V28) "
            "before releasing this data publicly. This means our model can "
            "accurately detect which components are unusual for a given "
            "transaction — but we can't trace those back to the original "
            "real-world meaning, since that information was intentionally "
            "scrambled for privacy. In a production deployment with a real "
            "merchant's own transaction data, this same model would be able to "
            "show fully readable explanations (e.g., 'unusual purchase location' "
            "or 'amount far above typical spending')."
        )

    st.divider()
    truth = "Fraud" if row.Class == 1 else "Genuine"
    correct = (score >= threshold) == (row.Class == 1)
    verdict = "✅ correct" if correct else "❌ missed"
    st.subheader("Ground truth")
    st.write(f"This transaction was actually **{truth}** — model prediction was {verdict}.")


if __name__ == "__main__":
    main()
