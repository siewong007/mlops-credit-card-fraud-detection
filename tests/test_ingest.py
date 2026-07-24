import numpy as np
import pandas as pd

from src.ingest import inject_drift, split_by_time

PARAMS = {"data": {"train_frac": 0.5, "valid_frac": 0.2, "n_prod_batches": 3}}


def _df(n=100):
    return pd.DataFrame({"Time": range(n), "Class": [0] * n})


def test_split_is_time_ordered_and_disjoint():
    parts = split_by_time(_df(), PARAMS)
    assert set(parts) == {"train", "valid", "prod_1", "prod_2", "prod_3"}
    assert sum(len(p) for p in parts.values()) == 100
    assert parts["train"]["Time"].max() < parts["valid"]["Time"].min()
    assert parts["valid"]["Time"].max() < parts["prod_1"]["Time"].min()


def test_inject_drift_shifts_amount_and_signal_features():
    rng = np.random.RandomState(0)
    n = 200
    data = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    data["Amount"] = np.abs(rng.normal(50, 10, n))
    data["Class"] = [1] * 5 + [0] * (n - 5)
    df = pd.DataFrame(data)
    params = {"data": {"drift_amount_scale": 3.0, "drift_feature_shift": 2.5}}

    out = inject_drift(df, params)
    # Amount is scaled up; a signal feature (V14) shifts; a non-signal one (V1) does not.
    assert out["Amount"].mean() > df["Amount"].mean() * 2
    assert out["V14"].mean() > df["V14"].mean() + 1.0
    assert np.allclose(out["V1"], df["V1"])
    # Fraud rows shift twice as far as legit on a signal feature (concept drift).
    fraud, legit = out["Class"] == 1, out["Class"] == 0
    assert out.loc[fraud, "V14"].mean() > out.loc[legit, "V14"].mean() + 1.0
