import json
import sys

import numpy as np
import pandas as pd
import pytest

from src import features
from src import drift
from src.drift import evidently_report, feature_drift, psi, summarize_batch
from src.evidence import sha256_file

PARAMS = {"drift": {"psi_threshold": 0.2, "ks_pvalue": 0.05}}
MODEL_ID = "fraud-detector:v2"
BATCH_FINGERPRINT = "a" * 64


def _frame(rng, shift_amount=0.0, shift_v1=0.0, n=3000):
    data = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    data["V1"] = data["V1"] + shift_v1
    data["Amount"] = np.abs(rng.normal(50, 10, n)) + shift_amount
    data["Time"] = np.arange(n, dtype=float)
    data["Class"] = 0
    return pd.DataFrame(data)


def _predictions(frame, labeled=True):
    out = pd.DataFrame(
        {
            "row_position": np.arange(len(frame)),
            "Time": frame["Time"].to_numpy(),
            "Amount": frame["Amount"].to_numpy(),
            "proba": np.linspace(0.01, 0.99, len(frame)),
            "pred": np.zeros(len(frame), dtype=int),
            "operating_threshold": 0.5,
            "promoted_model_id": MODEL_ID,
            "batch_data_fingerprint": BATCH_FINGERPRINT,
        }
    )
    if labeled:
        out["Class"] = frame["Class"].to_numpy()
    return out


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


def test_summary_marks_available_labels_and_cost():
    rng = np.random.RandomState(7)
    reference = _frame(rng, n=3000)
    reference.iloc[-6:, reference.columns.get_loc("Class")] = 1
    current = _frame(rng, n=3000)
    current.iloc[-6:, current.columns.get_loc("Class")] = 1

    summary = summarize_batch(
        reference,
        current,
        np.linspace(0.01, 0.99, len(reference)),
        _predictions(current),
        batch_name="prod_1",
        threshold=0.5,
        promoted_model_id=MODEL_ID,
        batch_data_fingerprint=BATCH_FINGERPRINT,
        params={
            **PARAMS,
            "threshold": {"cost_false_negative": 100, "cost_false_positive": 1},
        },
    )

    assert summary["label_status"] == "available"
    assert summary["pr_auc"] is not None
    assert summary["estimated_business_cost"] is not None


def test_summary_keeps_drift_when_labels_are_pending():
    rng = np.random.RandomState(8)
    reference = _frame(rng, n=3000)
    current = _frame(rng, shift_amount=40.0, n=3000).drop(columns=["Class"])

    summary = summarize_batch(
        reference,
        current,
        np.linspace(0.01, 0.99, len(reference)),
        _predictions(current, labeled=False),
        batch_name="prod_unlabeled",
        threshold=0.5,
        promoted_model_id=MODEL_ID,
        batch_data_fingerprint=BATCH_FINGERPRINT,
        params={
            **PARAMS,
            "threshold": {"cost_false_negative": 100, "cost_false_positive": 1},
        },
    )

    assert summary["label_status"] == "pending"
    assert summary["amount_psi"] > 0
    assert summary["pr_auc"] is None
    assert summary["recall"] is None
    assert summary["target_rate"] is None
    json.dumps(summary, allow_nan=False)


def _summarize(reference, current, predictions):
    return summarize_batch(
        reference,
        current,
        np.linspace(0.01, 0.99, len(reference)),
        predictions,
        batch_name="prod_1",
        threshold=0.5,
        promoted_model_id=MODEL_ID,
        batch_data_fingerprint=BATCH_FINGERPRINT,
        params={
            **PARAMS,
            "threshold": {"cost_false_negative": 100, "cost_false_positive": 1},
        },
    )


def test_summary_rejects_prediction_row_count_mismatch():
    rng = np.random.RandomState(11)
    reference = _frame(rng, n=20)
    current = _frame(rng, n=10)

    with pytest.raises(ValueError, match="row count"):
        _summarize(reference, current, _predictions(current).iloc[:-1])


@pytest.mark.parametrize("mutation", ["reordered", "time", "amount", "class"])
def test_summary_rejects_reordered_or_misaligned_rows(mutation):
    rng = np.random.RandomState(12)
    reference = _frame(rng, n=20)
    current = _frame(rng, n=10)
    predictions = _predictions(current)
    if mutation == "reordered":
        predictions = predictions.iloc[::-1]
    else:
        predictions.loc[0, mutation.capitalize()] += 1

    with pytest.raises(ValueError, match="row alignment"):
        _summarize(reference, current, predictions)


def test_summary_rejects_wrong_uniform_operating_threshold():
    rng = np.random.RandomState(13)
    reference = _frame(rng, n=20)
    current = _frame(rng, n=10)
    predictions = _predictions(current)
    predictions["operating_threshold"] = 0.4

    with pytest.raises(ValueError, match="operating threshold"):
        _summarize(reference, current, predictions)


def test_summary_rejects_wrong_promoted_model_id():
    rng = np.random.RandomState(14)
    reference = _frame(rng, n=20)
    current = _frame(rng, n=10)
    predictions = _predictions(current)
    predictions["promoted_model_id"] = "fraud-detector:v999"

    with pytest.raises(ValueError, match="promoted model"):
        _summarize(reference, current, predictions)


def test_summary_rejects_wrong_batch_data_fingerprint():
    rng = np.random.RandomState(15)
    reference = _frame(rng, n=20)
    current = _frame(rng, n=10)
    predictions = _predictions(current)
    predictions["batch_data_fingerprint"] = "b" * 64

    with pytest.raises(ValueError, match="batch data fingerprint"):
        _summarize(reference, current, predictions)


@pytest.mark.parametrize("probability", [np.nan, np.inf])
def test_summary_rejects_non_finite_probabilities(probability):
    rng = np.random.RandomState(16)
    reference = _frame(rng, n=20)
    current = _frame(rng, n=10)
    predictions = _predictions(current)
    predictions.loc[0, "proba"] = probability

    with pytest.raises(ValueError, match="finite probabilities"):
        _summarize(reference, current, predictions)


def test_summary_rejects_non_binary_predictions():
    rng = np.random.RandomState(17)
    reference = _frame(rng, n=20)
    current = _frame(rng, n=10)
    predictions = _predictions(current)
    predictions.loc[0, "pred"] = 2

    with pytest.raises(ValueError, match="binary predictions"):
        _summarize(reference, current, predictions)


def test_main_rejects_current_batch_mutation_while_reading(tmp_path, monkeypatch):
    bdir = tmp_path / "batches"
    reports = tmp_path / "reports"
    drift_dir = reports / "drift"
    figures = reports / "figures"
    bdir.mkdir()
    drift_dir.mkdir(parents=True)
    figures.mkdir()
    rng = np.random.RandomState(18)
    train = _frame(rng, n=40)
    calibration = _frame(rng, n=20)
    current = _frame(rng, n=20)
    train.to_csv(bdir / "train.csv", index=False)
    calibration.to_csv(bdir / "calibration.csv", index=False)
    source_path = bdir / "prod_1.csv"
    current.to_csv(source_path, index=False)
    mutated = current.copy()
    mutated.loc[0, "V1"] = 99.0
    mutated_path = tmp_path / "mutated.csv"
    mutated.to_csv(mutated_path, index=False)
    predictions = _predictions(current)
    predictions["batch_data_fingerprint"] = sha256_file(mutated_path)
    predictions.to_csv(bdir / "preds_prod_1.csv", index=False)
    scaler = features.fit_scaler(train)

    class FakeModel:
        def predict_proba(self, X):
            probability = np.linspace(0.01, 0.99, len(X))
            return np.column_stack([1 - probability, probability])

    real_read_csv = pd.read_csv

    def read_then_mutate(path, *args, **kwargs):
        frame = real_read_csv(path, *args, **kwargs)
        if path == source_path:
            mutated.to_csv(source_path, index=False)
        return frame

    monkeypatch.setattr(
        drift,
        "load_params",
        lambda: {
            "data": {"n_prod_batches": 1},
            "drift": {"psi_threshold": 0.2},
            "threshold": {
                "cost_false_negative": 100,
                "cost_false_positive": 1,
            },
        },
    )
    monkeypatch.setattr(drift, "batch_dir", lambda params: bdir)
    monkeypatch.setattr(drift, "DRIFT_DIR", drift_dir)
    monkeypatch.setattr(drift, "FIGURES_DIR", figures)
    monkeypatch.setattr(drift, "REPORTS_DIR", reports)
    monkeypatch.setattr(drift, "load_model", lambda: FakeModel())
    monkeypatch.setattr(drift.features, "load_scaler", lambda: scaler)
    monkeypatch.setattr(
        "src.artifacts.load_threshold",
        lambda: {"threshold": 0.5, "promoted_model_id": MODEL_ID},
    )
    monkeypatch.setattr(drift, "ensure_dirs", lambda: None)
    monkeypatch.setattr(drift, "evidently_report", lambda *args: {
        "status": "generated",
        "error_type": None,
        "message": None,
    })
    monkeypatch.setattr(drift, "_plot_psi", lambda *args: None)
    monkeypatch.setattr(drift.pd, "read_csv", read_then_mutate)

    with pytest.raises(ValueError, match="changed while being read"):
        drift.main()


def test_evidently_report_surfaces_optional_dependency_failure(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "evidently", None)
    target = tmp_path / "drift.html"
    target.write_text("stale successful report", encoding="utf-8")

    result = evidently_report(
        _frame(np.random.RandomState(9)),
        _frame(np.random.RandomState(10)),
        target,
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "ModuleNotFoundError"
    assert result["message"]
    assert not target.exists()
