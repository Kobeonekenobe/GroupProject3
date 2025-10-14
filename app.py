import os, json, joblib, numpy as np, pandas as pd, streamlit as st
from src.io import load_csv

st.set_page_config(page_title="Horse Racing Predictive Analytics", page_icon="🏇", layout="wide")
st.title("🏇 Predictive Analytics in NZ Horse Racing")

REGISTRY = {
    "he": {"label":"Betting Odds","title": "Investigate the Predicitve Value of features compared to betting odds, in the context of race winners", "tag": "form_xgb"},
    "jw": {"label":"Frequent Pairings","title": "Do frequent trainer/jockey pairings outperform infrequent pairings", "tag": "lgbm"},
    "kns":{"label":"Lineage Features","title": "Does incorporating lineage-based features, such as sire and dam-sire performance, improve the model’s ability to predict whether a racehorse finishes in the top three?", "tag": "lineage_lgbm"},
    "lr": {"label":"Track/Going Effects","title": "What impact does the environment (weather and track conditions) and equipment impact on top 3 finishes in a horse race?", "tag": "track_ada"},
}

with st.sidebar:
    keys = ["he","jw", "kns", "lr"]  # optional: control the order
    initials = st.selectbox("Select analysis",options=keys,format_func=lambda k: REGISTRY[k]["label"]) # shows “Track & Weather Conditions” for lr
    tag = st.text_input("Artifact tag", value=REGISTRY[initials]["tag"])
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.toggle("Use sample data", value=(uploaded is None))
    sample_path = st.text_input("Sample CSV path", value="data/cleaned_data_v1.csv")

st.header(REGISTRY[initials]["title"])

# Load data
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample and os.path.exists(sample_path):
    df = load_csv(sample_path)
else:
    st.info("Upload a CSV or enable sample data.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(), use_container_width=True)

# Load artifact (initials_tag)
model_path  = f"models/{initials}_{tag}.joblib"
schema_path = f"models/{initials}_{tag}.schema.json"

if os.path.exists(model_path) and os.path.exists(schema_path):
    with open(schema_path) as f:
        feature_order = json.load(f)["feature_order"]
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        st.error(f"Data is missing model features: {missing}")
        st.write("Expected:", feature_order)
        st.stop()

    model = joblib.load(model_path)
    X = df[feature_order]
    scores = model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else model.predict(X)

    out = df.copy()
    out["score"] = scores
    thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    out["pred"] = (out["score"] >= thr).astype(int)
    st.subheader("Predictions")
    st.dataframe(out.head(200), use_container_width=True)

    label = next((c for c in ["top3","target","y","label","win"] if c in out.columns), None)
    if label:
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        y_true = out[label].astype(int)
        try: auc = roc_auc_score(y_true, scores)
        except Exception: auc = float("nan")
        c1,c2,c3 = st.columns(3)
        c1.metric("AUC", "N/A" if np.isnan(auc) else f"{auc:.3f}")
        c2.metric("Accuracy", f"{accuracy_score(y_true, out['pred']):.3f}")
        c3.metric("F1", f"{f1_score(y_true, out['pred']):.3f}")

if initials == "kns" and "out" in locals():
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.markdown("## 🧬 Lineage Feature Analysis")

    # === Background ===
    st.markdown("""
    The **Lineage Model** investigates whether genetic heritage — specifically sire and dam-sire performance —
    enhances the ability to predict whether a racehorse finishes in the top three.

    Horses often inherit key performance traits such as **stamina, acceleration, and race temperament**.
    This section explores how lineage features interact with other core variables, especially **age**,
    which emerged as the **most influential factor** in the base model and the **second-most important**
    in the lineage model.
    """)

    # === Age Performance Summary ===
    if "age" in out.columns and "top3" in out.columns:
        age_summary = (
            out.groupby("age")["top3"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"mean": "avg_top3_rate"})
        )

        # Filter out small sample sizes (<30)
        age_summary_filtered = age_summary[age_summary["count"] >= 30].copy()

        # Optional smoothing to make trend clearer
        age_summary_filtered["smooth_rate"] = (
            age_summary_filtered["avg_top3_rate"].rolling(window=3, center=True).mean()
        )

        # === Visualization: Average Top-3 Finish Rate by Age ===
        st.markdown("### 🐎 Age and Performance Relationship")

        fig, ax = plt.subplots()
        sns.lineplot(
            data=age_summary_filtered,
            x="age",
            y="avg_top3_rate",
            marker="o",
            color="red",
            ax=ax,
        )
        ax.set_title("Average Top-3 Finish Rate by Age (Filtered for n ≥ 30)")
        ax.set_xlabel("Horse Age")
        ax.set_ylabel("Proportion of Horses Finishing Top-3")
        st.pyplot(fig)

        st.caption(
            "Filtered to include only age groups with ≥30 samples, ensuring statistically reliable comparisons."
        )

        st.markdown("""
        #### Interpretation
        Below, a filtered model can be observed where age groups with fewer than 30 observations
        are excluded to prevent statistical noise.

        The analysis shows that horses aged **3 to 4** have the **highest probability of finishing in the top three**, 
        aligning with the athletic prime typical in racing.  

        A noticeable spike appeared around **age 11**, where the proportion of top-3 finishes rose to approximately **0.33**.
        Upon closer investigation, this was found to be due to **a very small sample size (n = 3)** — 
        a few outlier horses with strong results skewed the mean.
        Once filtered, the trend shows a consistent decline beyond age 7, 
        confirming that **older horses perform significantly below their younger counterparts**.
        """)

    # === Optional: Lineage Feature Correlation Visualization ===
    lineage_cols = [
        c for c in out.columns if any(k in c.lower() for k in ["sire", "dam_sire", "grandsire"])
    ]
    if lineage_cols:
        corr = out[["score"] + lineage_cols].corr()["score"].sort_values(ascending=False).drop("score")
        st.markdown("### 🧠 Lineage Feature Correlations with Model Score")
        st.bar_chart(corr)
        st.caption(
            "Lineage variables such as **sire_avg_top3** and **dam_sire_win_rate** show moderate positive correlation "
            "with predicted race success probabilities, supporting the hypothesis that superior bloodlines "
            "enhance performance potential."
        )

    # === Optional: Prediction Score Distribution ===
    st.markdown("### 🎯 Prediction Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(out["score"], bins=30, kde=True, color="royalblue", ax=ax)
    ax.set_title("Distribution of Predicted Top-3 Probabilities")
    ax.set_xlabel("Predicted Probability (score)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.caption(
        "The probability distribution shows how confidently the model distinguishes between likely top performers "
        "and lower-tier horses. A more polarized shape reflects stronger model discrimination."
    )

    # === Summary Paragraph ===
    st.markdown("""
    ---
    **Summary:**  
    Age remains the most influential driver of performance, peaking between **3–4 years old** and
    declining thereafter.  
    Incorporating lineage variables improves both **model interpretability** and **predictive accuracy**, 
    capturing hereditary performance signals that complement traditional race metrics.
    This analysis underscores the importance of **sample size validation** and **contextual feature interpretation**
    when evaluating predictive insights.
    """)
else:
    st.info(f"No artifact found for **{initials.upper()}_{tag}** yet. Train in the notebook and commit files to `/models`.")

