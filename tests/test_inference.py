import numpy as np
import pandas as pd
import pytest

from src import features
from src import batch_inference
from src.batch_inference import score_batch
from src.evidence import sha256_file
from src.validate import DataValidationError

_BATCH_FINGERPRINT = "a" * 64


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
        batch_data_fingerprint=_BATCH_FINGERPRINT,
    )
    assert {
        "row_position",
        "Time",
        "Amount",
        "proba",
        "pred",
        "Class",
        "batch_data_fingerprint",
    } <= set(out)
    assert out["row_position"].tolist() == list(range(len(source)))
    assert out["operating_threshold"].unique().tolist() == [0.5]
    assert out["promoted_model_id"].unique().tolist() == ["fraud-detector:v2"]
    assert out["batch_data_fingerprint"].unique().tolist() == [_BATCH_FINGERPRINT]


def test_unlabeled_batch_is_supported_and_audited():
    labeled = _batch()
    out = score_batch(
        labeled.drop(columns=["Class"]),
        _FakeModel(),
        features.fit_scaler(labeled),
        threshold=0.5,
        promoted_model_id="fraud-detector:v2",
        batch_data_fingerprint=_BATCH_FINGERPRINT,
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
            batch_data_fingerprint=_BATCH_FINGERPRINT,
        )


def test_main_binds_predictions_to_source_file_fingerprint(tmp_path, monkeypatch):
    source_path = tmp_path / "prod_1.csv"
    _batch().to_csv(source_path, index=False)
    scaler = features.fit_scaler(_batch())
    monkeypatch.setattr(
        batch_inference,
        "load_params",
        lambda: {"data": {"n_prod_batches": 1}},
    )
    monkeypatch.setattr(batch_inference, "batch_dir", lambda params: tmp_path)
    monkeypatch.setattr(batch_inference, "load_model", lambda: _FakeModel())
    monkeypatch.setattr(batch_inference, "load_threshold", lambda: {
        "threshold": 0.5,
        "promoted_model_id": "fraud-detector:v2",
    })
    monkeypatch.setattr(batch_inference.features, "load_scaler", lambda: scaler)

    batch_inference.main()

    predictions = pd.read_csv(tmp_path / "preds_prod_1.csv")
    assert predictions["batch_data_fingerprint"].unique().tolist() == [
        sha256_file(source_path)
    ]


def test_main_rejects_source_mutation_while_reading(tmp_path, monkeypatch):
    source_path = tmp_path / "prod_1.csv"
    _batch().to_csv(source_path, index=False)
    scaler = features.fit_scaler(_batch())
    real_read_csv = pd.read_csv

    def read_then_mutate(path, *args, **kwargs):
        frame = real_read_csv(path, *args, **kwargs)
        if path == source_path:
            mutated = frame.copy()
            mutated.loc[0, "V1"] = 99.0
            mutated.to_csv(source_path, index=False)
        return frame

    monkeypatch.setattr(
        batch_inference,
        "load_params",
        lambda: {"data": {"n_prod_batches": 1}},
    )
    monkeypatch.setattr(batch_inference, "batch_dir", lambda params: tmp_path)
    monkeypatch.setattr(batch_inference, "load_model", lambda: _FakeModel())
    monkeypatch.setattr(
        batch_inference,
        "load_threshold",
        lambda: {
            "threshold": 0.5,
            "promoted_model_id": "fraud-detector:v2",
        },
    )
    monkeypatch.setattr(batch_inference.features, "load_scaler", lambda: scaler)
    monkeypatch.setattr(batch_inference.pd, "read_csv", read_then_mutate)

    with pytest.raises(ValueError, match="changed while being read"):
        batch_inference.main()

    assert not (tmp_path / "preds_prod_1.csv").exists()
