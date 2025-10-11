cat > src/io.py << 'PY'
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def ensure_columns(df: pd.DataFrame, expected: list[str]):
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
PY
