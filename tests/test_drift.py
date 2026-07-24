import numpy as np
import pandas as pd

from src import features
from src.drift import feature_drift, psi

PARAMS = {"drift": {"psi_threshold": 0.2, "ks_pvalue": 0.05}}


def _frame(rng, shift_amount=0.0, shift_v1=0.0, n=3000):
    data = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    data["V1"] = data["V1"] + shift_v1
    data["Amount"] = np.abs(rng.normal(50, 10, n)) + shift_amount
    data["Class"] = 0
    return pd.DataFrame(data)


def test_psi_zero_for_identical_distribution():
    x = np.random.RandomState(0).normal(size=5000)
    assert psi(x, x) == 0.0


def test_psi_large_under_mean_shift():
    rng = np.random.RandomState(0)
    assert psi(rng.normal(0, 1, 5000), rng.normal(3, 1, 5000)) > 0.2


def test_feature_drift_flags_only_shifted_features():
    rng = np.random.RandomState(1)
    ref = _frame(rng)
    cur = _frame(rng, shift_amount=40.0, shift_v1=3.0)
    result = feature_drift(ref, cur, PARAMS)
    assert result["per_feature"]["Amount"]["drifted"] is True
    assert result["per_feature"]["V1"]["drifted"] is True
    # An untouched feature should usually stay below threshold.
    assert result["per_feature"]["V15"]["drifted"] is False
    assert 0 <= result["pct_drifted_features"] <= 100
    assert result["n_features"] == len(features.FEATURES)
