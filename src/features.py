"""Feature processing (Owner: B).

Keep minimal: scale Amount (fit on train only to avoid leakage). The fitted
scaler is persisted alongside the model so batch inference applies the exact
same transform as training.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"
_SCALER_PATH = MODELS_DIR / "scaler.joblib"


def fit_scaler(train: pd.DataFrame) -> StandardScaler:
    return StandardScaler().fit(train[["Amount"]])


def transform(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    df["Amount"] = scaler.transform(df[["Amount"]])
    return df


def xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]


def save_scaler(scaler: StandardScaler) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, _SCALER_PATH)
    return _SCALER_PATH


def load_scaler() -> StandardScaler:
    return joblib.load(_SCALER_PATH)
