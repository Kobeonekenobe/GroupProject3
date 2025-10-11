cat > app.py << 'PY'
import os, json, joblib, numpy as np, pandas as pd, streamlit as st
from src.io import load_csv

st.set_page_config(page_title="Horse Racing Predictive Analytics", page_icon="🏇", layout="wide")
st.title("🏇 Predictive Analytics in NZ Horse Racing")

REGISTRY = {
    "jw": {"title": "Frequent Pairings", "tag": "lgbm"},
    "kns": {"title": "Lineage Features", "tag": "lineage_lgbm"},
    "lr": {"title": "Track/Going Effects", "tag": "track_ada"},
    "he": {"title": "Form & Ratings", "tag": "form_xgb"},
}

with st.sidebar:
    initials = st.selectbox("Select teammate", list(REGISTRY.keys()), format_func=lambda k: k.upper())
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
else:
    st.info(f"No artifact found for **{initials.upper()}_{tag}** yet. Train in the notebook and commit files to `/models`.")
PY
