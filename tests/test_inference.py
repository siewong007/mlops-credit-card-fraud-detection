import numpy as np
import pandas as pd

from src import features
from src.batch_inference import score_batch


class _FakeModel:
    """Predicts fraud probability as a monotone function of Amount."""

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(X["Amount"].to_numpy())))
        return np.column_stack([1 - p, p])


def _batch(n=6):
    data = {f"V{i}": np.zeros(n) for i in range(1, 29)}
    data["Amount"] = np.linspace(-3, 3, n)
    data["Time"] = np.arange(n)
    data["Class"] = [0, 0, 0, 1, 1, 1]
    return pd.DataFrame(data)


def test_score_batch_thresholding_and_columns():
    scaler = features.fit_scaler(_batch())  # identity-ish on Amount
    out = score_batch(_batch(), _FakeModel(), scaler, threshold=0.5)
    assert {"Time", "Amount", "proba", "pred", "Class"} <= set(out.columns)
    # pred must be exactly proba >= threshold
    assert (out["pred"] == (out["proba"] >= 0.5).astype(int)).all()
    assert out["proba"].between(0, 1).all()


def test_score_batch_without_labels_omits_class():
    df = _batch().drop(columns=["Class"])
    out = score_batch(df, _FakeModel(), features.fit_scaler(_batch()), threshold=0.5)
    assert "Class" not in out.columns
    assert len(out) == len(df)
