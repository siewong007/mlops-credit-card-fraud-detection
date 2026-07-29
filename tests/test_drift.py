import json
import sys

import numpy as np
import pandas as pd

from src import features
from src.drift import evidently_report, feature_drift, psi, summarize_batch

PARAMS = {"drift": {"psi_threshold": 0.2, "ks_pvalue": 0.05}}


def _frame(rng, shift_amount=0.0, shift_v1=0.0, n=3000):
    data = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    data["V1"] = data["V1"] + shift_v1
    data["Amount"] = np.abs(rng.normal(50, 10, n)) + shift_amount
    data["Class"] = 0
    return pd.DataFrame(data)


def _predictions(frame, labeled=True):
    out = pd.DataFrame(
        {
            "proba": np.linspace(0.01, 0.99, len(frame)),
            "pred": np.zeros(len(frame), dtype=int),
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


def test_evidently_report_surfaces_optional_dependency_failure(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "evidently", None)

    result = evidently_report(_frame(np.random.RandomState(9)), _frame(np.random.RandomState(10)), tmp_path / "drift.html")

    assert result["status"] == "failed"
    assert result["error_type"] == "ModuleNotFoundError"
    assert result["message"]
