cat > src/train_common.py << 'PY'
from dataclasses import dataclass
from typing import Dict, Any
import joblib, json, os, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

@dataclass
class TrainResult:
    model: Any
    metrics: Dict[str, float]
    feature_order: list[str]

def holdout_train(model, X, Y, test_size=0.2, seed=42) -> TrainResult:
    Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=test_size, stratify=Y, random_state=seed)
    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xte)[:,1]
        auc = roc_auc_score(yte, proba)
        yhat = (proba >= 0.5).astype(int)
    else:
        yhat = model.predict(Xte); auc = float("nan")
    return TrainResult(model=model,
                       metrics={"accuracy": accuracy_score(yte,yhat),
                                "f1": f1_score(yte,yhat),
                                "auc": auc},
                       feature_order=list(X.columns))

def save_artifacts(res: TrainResult, name: str, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(res.model, f"{out_dir}/{name}.joblib")
    with open(f"{out_dir}/{name}.schema.json","w") as f: f.write(json.dumps({"feature_order": res.feature_order}, indent=2))
    with open(f"{out_dir}/{name}.metrics.json","w") as f: f.write(json.dumps(res.metrics, indent=2))
PY
