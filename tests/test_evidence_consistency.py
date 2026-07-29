import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

pytestmark = pytest.mark.integration


def _load(name):
    return json.loads((REPORTS / name).read_text())


def test_generated_evidence_is_internally_consistent():
    comparison = _load("experiment_comparison.json")
    promotion = _load("promotion_record.json")
    operating_point = _load("operating_point.json")
    default = _load("metrics_default.json")
    operating = _load("metrics_operating.json")
    drift = _load("drift_summary.json")
    trigger = _load("trigger_decisions.json")
    manifest = _load("run_manifest.json")
    validation = _load("validation_report.json")

    promoted = [item for item in comparison["candidates"] if item["promoted"]]
    assert len(promoted) == 1
    assert promoted[0]["run_id"] == promotion["mlflow_run_id"]
    assert promoted[0]["model_name"] == promotion["model_name"]
    assert comparison["promoted_run_id"] == promotion["mlflow_run_id"]

    promoted_model_id = promotion["promoted_model_id"]
    assert operating_point["promoted_model_id"] == promoted_model_id
    assert default["promoted_model_id"] == promoted_model_id
    assert operating["promoted_model_id"] == promoted_model_id
    assert operating["threshold"] == operating_point["threshold"]
    assert default["threshold"] == 0.5
    assert default["evidence_role"] == "comparison_default_threshold"
    assert operating["evidence_role"] == "primary_operating_point"

    costs = operating_point["cost_assumptions"]
    for metrics in (default, operating):
        assert metrics["estimated_business_cost"] == (
            metrics["fn"] * costs["false_negative"]
            + metrics["fp"] * costs["false_positive"]
        )
        assert metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"] > 0

    drift_batches = [item["batch"] for item in drift]
    decision_batches = [item["batch"] for item in trigger["decisions"]]
    assert decision_batches == drift_batches
    severity = {"ok": 0, "warning": 1, "retrain": 2}
    expected_overall = max(
        (item["status"] for item in trigger["decisions"]),
        key=severity.__getitem__,
    )
    assert trigger["overall_status"] == expected_overall

    for summary, decision in zip(drift, trigger["decisions"], strict=True):
        assert summary["promoted_model_id"] == promoted_model_id
        assert summary["operating_threshold"] == operating_point["threshold"]
        assert re.fullmatch(
            r"[0-9a-f]{64}", summary["batch_data_fingerprint"]
        )
        assert summary["label_status"] == decision["label_status"]
        assert decision["reason_codes"]
        for key in (
            "promoted_model_id",
            "operating_threshold",
            "batch_data_fingerprint",
        ):
            assert decision["observed"][key] == summary[key]

        native = _load(f"drift/{summary['batch']}.json")
        for key, value in summary.items():
            assert native[key] == value
        assert native["label_status"] == summary["label_status"]
        assert native["evidently"] == summary["evidently"]

    assert manifest["python_version"] == "3.13.9"
    assert validation["raw_data_fingerprint"] == manifest["data"][
        "fingerprint_sha256"
    ]
    assert manifest["data"]["fingerprint_sha256"] == promotion["data_fingerprint"]
    assert manifest["parameters"]["fingerprint_sha256"] == promotion[
        "parameter_fingerprint"
    ]
    for metrics in (default, operating):
        assert metrics["calibration_data_fingerprint"] == operating_point[
            "calibration_data_fingerprint"
        ]

    html = (ROOT / "site" / "index.html").read_text()
    for marker in (
        f"Model {promoted_model_id}",
        f"MLflow run {promotion['mlflow_run_id']}",
        f"registry version {promotion['registered_model_version']}",
        f"operating threshold {operating_point['threshold']:.2f}",
        f"status {trigger['overall_status']}",
    ):
        assert marker in html
    for summary in drift:
        assert (
            f"Native drift {summary['batch']} · model "
            f"{summary['promoted_model_id']} · operating threshold "
            f"{summary['operating_threshold']:.2f} · batch fingerprint "
            f"{summary['batch_data_fingerprint']} · labels "
            f"{summary['label_status']} · Evidently "
            f"{summary['evidently']['status']}"
        ) in html
