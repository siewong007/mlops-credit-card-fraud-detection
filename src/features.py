"""Feature processing (Owner: B).

Keep minimal: scale Amount (and optionally Time-derived features).
Fit scaler on train only to avoid leakage.
"""
from sklearn.preprocessing import StandardScaler
import pandas as pd

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"


def fit_scaler(train: pd.DataFrame) -> StandardScaler:
    return StandardScaler().fit(train[["Amount"]])


def transform(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    df["Amount"] = scaler.transform(df[["Amount"]])
    return df


def xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]
