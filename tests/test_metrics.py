import json

import numpy as np
import pandas as pd
import pytest

from src import artifacts, config, evaluate, features, threshold
from src.evaluate import compute_metrics, evaluate_probabilities
from src.threshold import build_operating_point, select_threshold
from src.validate import DataValidationError


def _calibration_frame(n=12):
    data = {f"V{i}": np.zeros(n) for i in range(1, 29)}
    data["Amount"] = np.linspace(1, 12, n)
    data["Time"] = np.arange(n, dtype=float)
    data["Class"] = [0] * (n - 2) + [1, 1]
    return pd.DataFrame(data)


def test_perfect_classifier():
    m = compute_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        threshold=0.5,
        cost_false_negative=100,
        cost_false_positive=1,
    )
    assert m["recall"] == 1.0 and m["precision"] == 1.0 and m["pr_auc"] == 1.0


def test_metrics_include_hand_derived_confusion_and_cost():
    """Catches swapped confusion cells or omitted FN/FP business-cost terms."""
    metrics = compute_metrics(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.2, 0.9],
        0.5,
        cost_false_negative=100,
        cost_false_positive=1,
    )
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5
    assert metrics["estimated_business_cost"] == 101.0


def test_metrics_use_none_for_auc_when_a_class_is_absent():
    """Catches strict-JSON-breaking NaN AUC values for one-class batches."""
    metrics = compute_metrics(
        [0, 0],
        [0.1, 0.9],
        0.5,
        cost_false_negative=100,
        cost_false_positive=1,
    )
    assert metrics["pr_auc"] is None
    assert metrics["roc_auc"] is None


def test_evaluation_returns_default_and_operating_views():
    """Catches treating the 0.5 comparison as the operating-point evidence."""
    operating_point = {
        "threshold": 0.3,
        "cost_assumptions": {"false_negative": 100.0, "false_positive": 1.0},
        "promoted_model_id": "fraud-detector:v4",
        "calibration_data_fingerprint": "c" * 64,
    }
    default, operating = evaluate_probabilities(
        [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.9], operating_point
    )
    assert default["threshold"] == 0.5
    assert operating["threshold"] == 0.3
    assert operating["evidence_role"] == "primary_operating_point"
    assert default["evidence_role"] == "comparison_default_threshold"


def test_operating_point_binds_calibration_costs_and_promotion():
    """Catches an operating point detached from its calibration or promoted model."""
    operating_point = build_operating_point(
        [0, 0, 0, 1],
        [0.1, 0.2, 0.4, 0.45],
        cost_false_negative=100,
        cost_false_positive=1,
        promotion_record={"promoted_model_id": "fraud-detector:v4"},
        calibration_data_fingerprint="c" * 64,
    )
    assert operating_point["threshold"] == pytest.approx(0.41)
    assert operating_point["estimated_business_cost"] == 0.0
    assert operating_point["cost_assumptions"] == {"false_negative": 100.0, "false_positive": 1.0}
    assert operating_point["promoted_model_id"] == "fraud-detector:v4"
    assert operating_point["calibration_data_fingerprint"] == "c" * 64


def test_threshold_prefers_recall_when_fn_costly():
    y = [0, 0, 0, 1]
    proba = [0.1, 0.2, 0.4, 0.45]
    t, _ = select_threshold(y, proba, c_fn=100, c_fp=1)
    assert t <= 0.45  # low threshold catches the fraud case


def test_threshold_main_rejects_calibration_mutation_while_reading(
    tmp_path, monkeypatch
):
    reports = tmp_path / "reports"
    figures = reports / "figures"
    models = tmp_path / "models"
    batches = tmp_path / "batches"
    for directory in (reports, figures, models, batches):
        directory.mkdir(parents=True, exist_ok=True)
    calibration_path = batches / "calibration.csv"
    calibration = _calibration_frame()
    calibration.to_csv(calibration_path, index=False)
    (reports / "promotion_record.json").write_text(
        json.dumps({"promoted_model_id": "fraud-detector:v4"}),
        encoding="utf-8",
    )
    scaler = features.fit_scaler(calibration)

    class FakeModel:
        def predict_proba(self, X):
            probability = np.linspace(0.01, 0.99, len(X))
            return np.column_stack([1 - probability, probability])

    real_read_csv = pd.read_csv

    def read_then_mutate(path, *args, **kwargs):
        frame = real_read_csv(path, *args, **kwargs)
        if path == calibration_path:
            mutated = frame.copy()
            mutated.loc[0, "V1"] = 99.0
            mutated.to_csv(calibration_path, index=False)
        return frame

    monkeypatch.setattr(
        config,
        "load_params",
        lambda: {
            "data": {},
            "threshold": {
                "cost_false_negative": 100,
                "cost_false_positive": 1,
            },
        },
    )
    monkeypatch.setattr(config, "batch_dir", lambda params: batches)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    monkeypatch.setattr(config, "FIGURES_DIR", figures)
    monkeypatch.setattr(artifacts, "load_model", lambda: FakeModel())
    monkeypatch.setattr(artifacts, "_THRESHOLD_PATH", models / "threshold.json")
    monkeypatch.setattr(artifacts, "REPORTS_DIR", reports)
    monkeypatch.setattr(features, "load_scaler", lambda: scaler)
    monkeypatch.setattr(threshold, "_plot_tradeoff", lambda *args: None)
    monkeypatch.setattr("src.validate.pd.read_csv", read_then_mutate)

    with pytest.raises(DataValidationError, match="changed while being read"):
        threshold.main()

    assert not (reports / "operating_point.json").exists()


def test_evaluate_main_rejects_calibration_outside_operating_point(
    tmp_path, monkeypatch
):
    batches = tmp_path / "batches"
    batches.mkdir()
    _calibration_frame().to_csv(batches / "calibration.csv", index=False)
    monkeypatch.setattr(config, "load_params", lambda: {"data": {}})
    monkeypatch.setattr(config, "batch_dir", lambda params: batches)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    monkeypatch.setattr(
        artifacts,
        "load_threshold",
        lambda: {
            "threshold": 0.5,
            "promoted_model_id": "fraud-detector:v4",
            "calibration_data_fingerprint": "0" * 64,
            "cost_assumptions": {
                "false_negative": 100,
                "false_positive": 1,
            },
        },
    )
    monkeypatch.setattr(
        artifacts,
        "load_model",
        lambda: (_ for _ in ()).throw(AssertionError("model must not load")),
    )

    with pytest.raises(DataValidationError, match="expected SHA-256"):
        evaluate.main()
