import numpy as np
import pandas as pd
import pytest

from src import features
from src.batch_inference import score_batch
from src.validate import DataValidationError


class _FakeModel:
    """Predicts fraud probability as a monotone function of Amount."""

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(X["Amount"].to_numpy())))
        return np.column_stack([1 - p, p])


def _batch(n=6):
    data = {f"V{i}": np.zeros(n) for i in range(1, 29)}
    data["Amount"] = np.linspace(0, 6, n)
    data["Time"] = np.arange(n)
    data["Class"] = [0, 0, 0, 1, 1, 1]
    return pd.DataFrame(data)


def test_score_batch_records_operating_contract():
    source = _batch()
    scaler = features.fit_scaler(source)
    out = score_batch(
        source,
        _FakeModel(),
        scaler,
        threshold=0.5,
        promoted_model_id="fraud-detector:v2",
    )
    assert {"Time", "Amount", "proba", "pred", "Class"} <= set(out)
    assert out["operating_threshold"].unique().tolist() == [0.5]
    assert out["promoted_model_id"].unique().tolist() == ["fraud-detector:v2"]


def test_unlabeled_batch_is_supported_and_audited():
    labeled = _batch()
    out = score_batch(
        labeled.drop(columns=["Class"]),
        _FakeModel(),
        features.fit_scaler(labeled),
        threshold=0.5,
        promoted_model_id="fraud-detector:v2",
    )
    assert "Class" not in out.columns
    assert len(out) == len(labeled)


def test_missing_feature_fails_before_prediction():
    class NeverCalled(_FakeModel):
        def predict_proba(self, X):
            raise AssertionError("model must not run")

    with pytest.raises(DataValidationError):
        score_batch(
            _batch().drop(columns=["V8", "Class"]),
            NeverCalled(),
            features.fit_scaler(_batch()),
            threshold=0.5,
            promoted_model_id="fraud-detector:v2",
        )
