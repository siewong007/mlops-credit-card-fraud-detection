import json

import numpy as np
import pandas as pd
import pytest

from src.validate import (
    DataValidationError,
    check_dataframe,
    require_valid_dataframe,
    require_validation_gate,
    validate_file,
)


def _frame(n=1000, labeled=True):
    data = {"Time": np.arange(n, dtype=float)}
    data.update({f"V{i}": np.zeros(n) for i in range(1, 29)})
    data["Amount"] = np.ones(n)
    if labeled:
        target = np.zeros(n, dtype=int)
        target[-2:] = 1
        data["Class"] = target
    return pd.DataFrame(data)


def test_labeled_contract_checks_distribution_and_finite_values():
    assert require_valid_dataframe(
        _frame(), contract="labeled", require_target_distribution=True
    ).shape == (1000, 31)
    with pytest.raises(DataValidationError):
        require_valid_dataframe(
            _frame().assign(V7=np.inf),
            contract="labeled",
            require_target_distribution=True,
        )
    with pytest.raises(DataValidationError):
        require_valid_dataframe(
            _frame().assign(Class=0),
            contract="labeled",
            require_target_distribution=True,
        )
    high_rate = _frame()
    high_rate.loc[:50, "Class"] = 1
    with pytest.raises(DataValidationError):
        require_valid_dataframe(
            high_rate, contract="labeled", require_target_distribution=True
        )


def test_inference_contract_accepts_missing_target_only():
    report = check_dataframe(_frame(labeled=False), contract="inference")
    assert report["status"] == "passed"
    assert report["fraud_count"] is None
    assert report["fraud_rate"] is None
    with pytest.raises(DataValidationError):
        require_valid_dataframe(
            _frame(labeled=False).drop(columns=["V12"]), contract="inference"
        )


def test_inference_with_target_uses_labeled_contract():
    with pytest.raises(DataValidationError):
        require_valid_dataframe(_frame().assign(Class=2), contract="inference")


def test_validate_file_writes_failed_evidence_before_raising(tmp_path):
    raw_path = tmp_path / "raw.csv"
    report_path = tmp_path / "validation_report.json"
    _frame().to_csv(raw_path, index=False)
    params = {
        "data": {
            "train_frac": 0.5,
            "model_valid_frac": 0.1,
            "calibration_frac": 0.1,
            "n_prod_batches": 3,
        }
    }

    with pytest.raises(DataValidationError):
        validate_file(raw_path, params, report_path)

    report = json.loads(report_path.read_text())
    assert report["overall_status"] == "failed"
    assert set(report["datasets"]) == {"raw", "train", "model_valid", "calibration"}
    assert "NaN" not in report_path.read_text()


def test_validate_file_reports_unsplittable_raw_schema_before_raising(tmp_path):
    raw_path = tmp_path / "raw.csv"
    report_path = tmp_path / "validation_report.json"
    _frame().drop(columns=["Time"]).to_csv(raw_path, index=False)
    params = {
        "data": {
            "train_frac": 0.5,
            "model_valid_frac": 0.1,
            "calibration_frac": 0.1,
            "n_prod_batches": 3,
        }
    }

    with pytest.raises(DataValidationError):
        validate_file(raw_path, params, report_path)

    report = json.loads(report_path.read_text())
    assert report["overall_status"] == "failed"
    assert report["datasets"]["raw"]["status"] == "failed"
    for name in ("train", "model_valid", "calibration"):
        assert report["datasets"][name]["status"] == "failed"
        assert report["datasets"][name]["checks"][0]["name"] == "slice_unavailable"


def test_validation_gate_rejects_missing_failed_or_incomplete_report(tmp_path):
    path = tmp_path / "validation_report.json"
    with pytest.raises(DataValidationError):
        require_validation_gate(path)


def test_validation_gate_rejects_empty_or_failed_dataset_evidence(tmp_path):
    path = tmp_path / "validation_report.json"
    required = ("raw", "train", "model_valid", "calibration")
    for datasets in (
        {name: {} for name in required},
        {
            name: {"status": "failed" if name == "calibration" else "passed"}
            for name in required
        },
    ):
        path.write_text(json.dumps({"overall_status": "passed", "datasets": datasets}))
        with pytest.raises(DataValidationError):
            require_validation_gate(path)
    path.write_text(json.dumps({"overall_status": "failed", "datasets": {}}))
    with pytest.raises(DataValidationError):
        require_validation_gate(path)
    path.write_text(
        json.dumps(
            {
                "overall_status": "passed",
                "datasets": {"raw": {}, "train": {}, "model_valid": {}},
            }
        )
    )
    with pytest.raises(DataValidationError):
        require_validation_gate(path)
