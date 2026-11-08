"""Higgs Boson classification demo (Streamlit).

Tabs: results of the project, an interactive AMS-threshold explorer on out-of-fold predictions,
and a scorer for new events with the final XGBoost model.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

HERE = Path(__file__).parent
FIGS = HERE.parent / "figures"
BR = 10.0

st.set_page_config(page_title="Higgs Boson Classifier", page_icon="⚛️", layout="wide")


# --------------------------------------------------------------------------- helpers
def ams(s, b, br=BR):
    s, b = np.asarray(s, float), np.asarray(b, float)
    return np.sqrt(np.maximum(2 * ((s + b + br) * np.log1p(s / (b + br)) - s), 0))


@st.cache_data
def load_meta():
    return json.loads((HERE / "data/meta.json").read_text())


@st.cache_data
def load_oof():
    d = np.load(HERE / "data/oof_scores.npz")
    order = np.argsort(-d["score"], kind="stable")
    y, w, sc = d["y"][order].astype(int), d["w"][order].astype(float), d["score"][order].astype(float)
    return sc, np.cumsum(w * (y == 1)), np.cumsum(w * (y == 0))


@st.cache_data
def load_tables():
    exp = pd.read_csv(HERE / "data/experiments_validation.csv")
    cv = pd.read_csv(HERE / "data/cv_summary.csv")
    return exp, cv


@st.cache_resource
def load_model():
    m = xgb.XGBClassifier()
    m.load_model(str(HERE / "model/xgb_higgs.json"))
    return m


meta = load_meta()
FEATURES = meta["features"]
sc, cum_s, cum_b = load_oof()
N = len(sc)

# --------------------------------------------------------------------------- header
st.title("⚛️ Higgs Boson classifier")
st.caption("ATLAS Higgs Boson Machine Learning Challenge · XGBoost with class-balanced weights · evaluated with the "
           "weighted Approximate Median Significance (AMS), not accuracy")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Out-of-fold AMS", f"{meta['oof_ams']:.2f}", help="250,000 training events, 5-fold stratified CV")
c2.metric("Events called signal", f"{meta['fraction']:.0%}", help="Top-scoring fraction chosen on out-of-fold data")
c3.metric("Probability cut", f"{meta['prob_threshold']:.3f}", help="Score of the event at the chosen fraction (not 0.5)")
c4.metric("All-signal AMS (reference)", "1.08")

tab_res, tab_ams, tab_score, tab_about = st.tabs(["📊 Results", "🎚️ AMS threshold explorer", "🧪 Score events", "ℹ️ About"])

# --------------------------------------------------------------------------- results
with tab_res:
    exp, cv = load_tables()
    st.subheader("Model comparison (validation split, best-cut AMS)")
    mods = exp[exp["Experiment"].str.startswith("MOD-")][["Model", "AUC", "AMS@0.5", "AMS_best"]].set_index("Model")
    left, right = st.columns([3, 2])
    with left:
        st.bar_chart(mods["AMS_best"].sort_values(), horizontal=True, height=340)
    with right:
        st.dataframe(mods.round(3), width="stretch")
    st.caption("The default 0.5 cut (AMS@0.5) is far below the best cut; ranking quality (AUC) and AMS broadly agree.")

    st.subheader("Training weights (HistGradientBoosting)")
    wexp = exp[exp["Experiment"].str.startswith("W-")][["Training weights", "AUC", "Acc@0.5", "AMS@0.5", "AMS_best"]]
    st.dataframe(wexp.round(3).set_index("Training weights"), width="stretch")

    st.subheader("5-fold cross-validation")
    st.dataframe(cv.round(4).set_index("model"), width="stretch")
    st.caption("The top models differ by about 0.04 AMS, less than the fold-to-fold standard deviation (0.15–0.29).")

    for name, cap in [("07_ams_vs_fraction_auc.png", "AMS vs selection fraction and AUC vs AMS"),
                      ("09_permutation_importance.png", "Permutation importance (a clue, not proof)")]:
        if (FIGS / name).exists():
            st.image(str(FIGS / name), caption=cap)

# --------------------------------------------------------------------------- AMS explorer
with tab_ams:
    st.subheader("Choose how many events to call signal")
    st.write("Out-of-fold predictions for all 250,000 training events, with their event weights. "
             "Move the slider: the AMS changes because signal purity changes.")
    frac = st.slider("Fraction of events called signal", 0.01, 0.60, float(meta["fraction"]), 0.005, format="%.3f")
    k = max(int(frac * N) - 1, 0)
    s_sel, b_sel = cum_s[k], cum_b[k]
    fr = np.arange(0.01, 0.60, 0.005)
    curve = ams(cum_s[(fr * N).astype(int) - 1], cum_b[(fr * N).astype(int) - 1])
    p05 = sc >= 0.5
    ams05 = float(ams(cum_s[p05.sum() - 1], cum_b[p05.sum() - 1]))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AMS at this fraction", f"{float(ams(s_sel, b_sel)):.3f}")
    m2.metric("Signal weight selected (s)", f"{s_sel:.1f}")
    m3.metric("Background weight selected (b)", f"{b_sel:,.0f}")
    m4.metric("Equivalent probability cut", f"{sc[k]:.3f}")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(fr, curve, color="#2f6fb0")
    ax.axvline(frac, color="#c2501a", ls="--", label="your choice")
    ax.axvline(meta["fraction"], color="grey", ls=":", label="chosen in the project")
    ax.set_xlabel("fraction of events called signal"); ax.set_ylabel("AMS (out-of-fold)")
    ax.legend(); ax.grid(alpha=.3)
    st.pyplot(fig)
    st.info(f"With the default probability cut of 0.5, {p05.mean():.1%} of events would be called signal "
            f"and the AMS would be only **{ams05:.2f}**, against {meta['oof_ams']:.2f} at the chosen cut.")

# --------------------------------------------------------------------------- scorer
with tab_score:
    st.subheader("Score events with the final model")
    st.write(f"Upload a CSV with the {len(FEATURES)} feature columns (`DER_*` and `PRI_*`; `-999` means undefined). "
             "`EventId` is optional and is kept only as an identifier; `Weight` and `Label` are ignored.")
    src = st.radio("Data", ["300 sample test events", "Upload a CSV"], horizontal=True)
    df = None
    if src.startswith("300"):
        df = pd.read_csv(HERE / "data/sample_events.csv")
    else:
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            df = pd.read_csv(up)
    if df is not None:
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error("Missing feature columns: " + ", ".join(missing))
        else:
            X = df[FEATURES].replace(-999.0, np.nan)
            p = load_model().predict_proba(X)[:, 1]
            mode = st.radio("Decision rule", ["Fixed probability cut (from training data)", "Top share of this batch"], horizontal=True)
            if mode.startswith("Fixed"):
                cut = st.number_input("Probability cut", 0.0, 1.0, float(meta["prob_threshold"]), 0.001, format="%.3f")
                called = p >= cut
            else:
                share = st.slider("Share of this batch called signal", 0.01, 0.60, float(meta["fraction"]), 0.005)
                called = p >= np.quantile(p, 1 - share)
            out = pd.DataFrame({"EventId": df["EventId"] if "EventId" in df.columns else np.arange(len(df)),
                                "signal_score": p.round(4), "class": np.where(called, "s", "b")})
            st.write(f"**{called.sum()}** of {len(out)} events called signal ({called.mean():.1%}).")
            st.dataframe(out.sort_values("signal_score", ascending=False), width="stretch", height=360)
            st.download_button("Download predictions (CSV)", out.to_csv(index=False), "predictions.csv", "text/csv")
            st.caption("Scores come from a model trained with class-balanced weights, so they are rankings, not calibrated "
                       "probabilities. The fixed cut keeps the project's selection rate on training-like data; on a different "
                       "batch the share of events called signal can differ.")

# --------------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**Task.** Classify ATLAS collision events as signal (Higgs → ττ) or background, scored by AMS:
`AMS = √(2[(s+b+10)·ln(1+s/(b+10)) − s])`, with `s`, `b` the *weighted* signal / background selected.

**Final model.** XGBoost on the 30 original features (`-999` → NaN, no scaling), class-balanced sample weights,
300 trees, learning rate 0.08, depth 6, seed 42. Trained on all 250,000 training events; the top 15 % of events by
score are called signal.

**Validation.** Stratified 80/20 split for all comparisons, then stratified 5-fold CV; the threshold was chosen on
out-of-fold predictions only. The competition test labels are private, so the leaderboard AMS cannot be computed offline.

**Limitations.** AMS is noisy (fold std 0.1–0.3); the 15 % cut was chosen on the same out-of-fold data used for the
estimate (slightly optimistic); hyper-parameters were not tuned; scores are not calibrated probabilities.

**Data.** Raw competition data are not part of this app. Sources: Kaggle `higgs-boson`, CERN Open Data record 328.
""")
