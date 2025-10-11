<h1>🏇 GroupProject3 — Predictive Analytics in NZ Horse Racing</h1>

A collaborative data science project exploring predictive models for race outcomes using NZ horse racing data.
We use a shared Jupyter notebook for exploration and a Streamlit web app to demo trained models.

Stack: Python · pandas · scikit-learn · LightGBM · Streamlit

<h2>🤝 Team & Research Questions</h2>

- Harrye (HE): How do form & ratings features impact top-3 predictions? 
- Kobe (KNS): Does incorporating lineage (sire/dam) improve predicting top-3 placement?
- Lagi (LR): How do track/going and recent performance affect outcomes?
- Jason (JW): Do frequent jockey–trainer pairings outperform ad-hoc pairings?


Each teammate works with their own dataframe in the notebook:
```
jw_df  = cleaned_df.copy()
kns_df = cleaned_df.copy()
lr_df  = cleaned_df.copy()
he_df  = cleaned_df.copy()
```

Artifacts (models) are saved using initials as a prefix, e.g. models/jw_lgbm.joblib.

<h2>📂 Repository Structure</h2>
GroupProject3/
├─ app.py                    # Streamlit app entrypoint
├─ pages/                    # (optional) extra Streamlit pages
├─ src/                      # shared code used by notebook + app
│  ├─ __init__.py
│  ├─ io.py                  # load_csv(), ensure_columns()
│  └─ train_common.py        # holdout_train(), save_artifacts()
├─ models/                   # saved models (.joblib) + schemas/metrics (JSON)
├─ data/
│  ├─ cleaned_data_v1.csv
│  └─ nz_flat_2000_2025.csv
└─ notebooks/
   └─ GroupProject3.ipynb

🚀 Quick Start (Local)
# 1) create and activate a venv
python -m venv .venv
# Windows PowerShell:
. .\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# 2) install deps
pip install --upgrade pip
pip install -r requirements.txt

# 3) run the app
streamlit run app.py


Open the browser to the URL shown (usually http://localhost:8501
).

🧪 Notebook → Model Artifacts (per teammate)

In notebooks/GroupProject3.ipynb, use the helper:

from src.train_common import holdout_train, save_artifacts
from lightgbm import LGBMClassifier

def train_and_save(df, features, target, initials: str, tag: str, model=None):
    X = df[features].copy()
    y = df[target].astype(int)    # adjust if already 0/1
    model = model or LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=-1, random_state=42)
    res = holdout_train(model, X, y)
    print(f"{initials}_{tag} metrics:", res.metrics)
    save_artifacts(res, name=f"{initials}_{tag}")


Example runs

# Jason
train_and_save(
    jw_df,
    features=["going","draw","age","lbs","has_blinkers","has_tongue_tie"],
    target="top3",
    initials="jw",
    tag="lgbm",
)

# Kobe (lineage)
train_and_save(
    kns_df,
    features=["sire_top3_rate","dam_top3_rate","sire_runners_log","dam_runners","going","age"],
    target="top3",
    initials="kns",
    tag="lineage_lgbm",
)


This writes three files into models/:

models/<initials>_<tag>.joblib         # trained model
models/<initials>_<tag>.schema.json    # {"feature_order": [...]}
models/<initials>_<tag>.metrics.json   # {"auc": ..., "accuracy": ..., "f1": ...}


The Streamlit app auto-detects these by initials.

🌐 Using the Streamlit App

streamlit run app.py

In the left sidebar:

Pick teammate initials (JW / KNS / LR / HE)

Keep or change the artifact tag (e.g., lgbm, lineage_lgbm)

Upload a CSV or use the sample (data/cleaned_data_v1.csv)

The page:

Shows a preview of your data

Loads the matching model (if found) and outputs scores/predictions

Provides a threshold slider and (if labels exist) live AUC/Accuracy/F1

If an artifact isn’t present yet, the app shows a light exploratory fallback so you can still demo.

📦 Requirements

See requirements.txt. Key dependencies:

streamlit, pandas, numpy, scikit-learn, lightgbm, joblib, plotly

Install with:

pip install -r requirements.txt

🔄 Collaboration Workflow

Initials Convention: dataframes and artifacts use initials (jw, kns, lr, he).

Branches (recommended): feature/<initials>-<short-desc> → PR → review → merge.

Do commit: code, notebook, small CSVs, model JSONs (*.schema.json, *.metrics.json).

Avoid committing: very large data, secret keys, raw models if huge (consider Git LFS).

🧾 Data

data/cleaned_data_v1.csv — ⟨brief description of fields and label column⟩

data/nz_flat_2000_2025.csv — ⟨brief description⟩

Ensure the feature columns used for training exist in the CSV you score.

📊 Metrics & Reporting

Metrics computed on a held-out split (train_test_split stratified).

Stored in models/<artifact>.metrics.json.

The app can compute live metrics if the label column (e.g., top3) is present in the scored CSV.

☁️ Optional Deployment (Streamlit Community Cloud)

Push to GitHub (public repo is fine).

Go to streamlit.io → Community Cloud → New app.

Select this repo and app.py as the entry file.

Add any secrets under App settings → Secrets (not required for this project).

🐛 Troubleshooting

streamlit: command not found → activate your venv and reinstall requirements.

Missing columns → the app will list the exact expected feature_order.

Port in use → streamlit run app.py --server.port 8502.

📜 License & Acknowledgements

⟨License, if any⟩

Thanks to ⟨data source / provider⟩ for the datasets.

✅ Checklist (for teammates)

 Train and save your artifact with train_and_save(...).

 Confirm three files exist in /models: .joblib, .schema.json, .metrics.json.

 Run streamlit run app.py, pick your initials, and verify predictions.

 Commit your changes and open a PR.
