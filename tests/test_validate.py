import json
import hashlib
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.validate import (
    DataValidationError,
    check_dataframe,
    read_stable_csv,
    require_valid_dataframe,
    require_validation_gate,
    validate_file,
)

PARAMS = {
    "data": {
        "train_frac": 0.5,
        "model_valid_frac": 0.1,
        "calibration_frac": 0.1,
        "n_prod_batches": 3,
    }
}


def _frame(n=1000, labeled=True):
    data = {"Time": np.arange(n, dtype=float)}
    data.update({f"V{i}": np.zeros(n) for i in range(1, 29)})
    data["Amount"] = np.ones(n)
    if labeled:
        target = np.zeros(n, dtype=int)
        target[-2:] = 1
        data["Class"] = target
    return pd.DataFrame(data)


def _valid_frame():
    frame = _frame()
    frame["Class"] = 0
    frame.loc[[10, 510, 610], "Class"] = 1
    return frame


def test_stable_csv_read_returns_verified_digest(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("value\n1\n", encoding="utf-8")

    frame, digest = read_stable_csv(path)

    assert frame["value"].tolist() == [1]
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_stable_csv_read_rejects_expected_fingerprint_mismatch(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("value\n1\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="expected SHA-256"):
        read_stable_csv(path, expected_sha256="0" * 64)


def test_stable_csv_read_rejects_mutation_during_read(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    path.write_text("value\n1\n", encoding="utf-8")
    real_read_csv = pd.read_csv

    def read_then_mutate(source, *args, **kwargs):
        frame = real_read_csv(source, *args, **kwargs)
        path.write_text("value\n2\n", encoding="utf-8")
        return frame

    monkeypatch.setattr("src.validate.pd.read_csv", read_then_mutate)

    with pytest.raises(DataValidationError, match="changed while being read"):
        read_stable_csv(path)


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
    #_frame().to_csv(raw_path, index=False)
    _frame().to_csv(raw_path, index=False, lineterminator="\n")

    with pytest.raises(DataValidationError):
        validate_file(raw_path, PARAMS, report_path)

    report = json.loads(report_path.read_text())
    assert report["overall_status"] == "failed"
    assert report["raw_data_fingerprint"] == (
        "453c58699473bf7c42d44e85f1ef1d991016d0767c51085da3e5b984841ae6b1"
    )
    assert len(report["split_parameter_fingerprint"]) == 64
    assert set(report["datasets"]) == {"raw", "train", "model_valid", "calibration"}
    assert "NaN" not in report_path.read_text()


def test_validate_file_reports_unsplittable_raw_schema_before_raising(tmp_path):
    raw_path = tmp_path / "raw.csv"
    report_path = tmp_path / "validation_report.json"
    _frame().drop(columns=["Time"]).to_csv(raw_path, index=False)

    with pytest.raises(DataValidationError):
        validate_file(raw_path, PARAMS, report_path)

    report = json.loads(report_path.read_text())
    assert report["overall_status"] == "failed"
    assert report["datasets"]["raw"]["status"] == "failed"
    for name in ("train", "model_valid", "calibration"):
        assert report["datasets"][name]["status"] == "failed"
        assert report["datasets"][name]["checks"][0]["name"] == "slice_unavailable"


def test_validation_gate_rejects_missing_failed_or_incomplete_report(tmp_path):
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text("raw", encoding="utf-8")
    report_path = tmp_path / "validation_report.json"
    with pytest.raises(DataValidationError):
        require_validation_gate(raw_path, PARAMS, report_path)


def test_validation_gate_rejects_empty_or_failed_dataset_evidence(tmp_path):
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text("raw", encoding="utf-8")
    report_path = tmp_path / "validation_report.json"
    required = ("raw", "train", "model_valid", "calibration")
    for datasets in (
        {name: {} for name in required},
        {
            name: {"status": "failed" if name == "calibration" else "passed"}
            for name in required
        },
    ):
        report_path.write_text(
            json.dumps({"overall_status": "passed", "datasets": datasets})
        )
        with pytest.raises(DataValidationError):
            require_validation_gate(raw_path, PARAMS, report_path)
    report_path.write_text(
        json.dumps({"overall_status": "failed", "datasets": {}})
    )
    with pytest.raises(DataValidationError):
        require_validation_gate(raw_path, PARAMS, report_path)
    report_path.write_text(
        json.dumps(
            {
                "overall_status": "passed",
                "datasets": {"raw": {}, "train": {}, "model_valid": {}},
            }
        )
    )
    with pytest.raises(DataValidationError):
        require_validation_gate(raw_path, PARAMS, report_path)


def test_validation_gate_rejects_raw_mutation_after_passing_report(tmp_path):
    raw_path = tmp_path / "raw.csv"
    report_path = tmp_path / "validation_report.json"
    frame = _valid_frame()
    frame.to_csv(raw_path, index=False)
    validate_file(raw_path, PARAMS, report_path)
    require_validation_gate(raw_path, PARAMS, report_path)

    frame.loc[0, "Amount"] = 2.0
    frame.to_csv(raw_path, index=False)

    with pytest.raises(DataValidationError, match="raw data fingerprint"):
        require_validation_gate(raw_path, PARAMS, report_path)


def test_validation_gate_rejects_split_parameter_mutation(tmp_path):
    raw_path = tmp_path / "raw.csv"
    report_path = tmp_path / "validation_report.json"
    _valid_frame().to_csv(raw_path, index=False)
    validate_file(raw_path, PARAMS, report_path)
    changed = deepcopy(PARAMS)
    changed["data"]["train_frac"] = 0.4

    with pytest.raises(DataValidationError, match="split parameter fingerprint"):
        require_validation_gate(raw_path, changed, report_path)


def test_read_failure_replaces_preexisting_success_evidence(
    tmp_path, monkeypatch
):
    raw_path = tmp_path / "raw.csv"
    report_path = tmp_path / "validation_report.json"
    _valid_frame().to_csv(raw_path, index=False)
    validate_file(raw_path, PARAMS, report_path)
    assert json.loads(report_path.read_text())["overall_status"] == "passed"

    def fail_read(*args, **kwargs):
        raise OSError("read failed")

    monkeypatch.setattr(pd, "read_csv", fail_read)

    with pytest.raises(DataValidationError, match="read raw data"):
        validate_file(raw_path, PARAMS, report_path)

    text = report_path.read_text()
    report = json.loads(text)
    assert report["overall_status"] == "failed"
    assert report["datasets"]["raw"]["status"] == "failed"
    assert "read failed" in report["datasets"]["raw"]["checks"][0]["details"]
    assert "NaN" not in text


def test_missing_raw_replaces_preexisting_success_evidence(tmp_path):
    raw_path = tmp_path / "missing.csv"
    report_path = tmp_path / "validation_report.json"
    report_path.write_text(
        json.dumps({"overall_status": "passed"}),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="read raw data"):
        validate_file(raw_path, PARAMS, report_path)

    text = report_path.read_text(encoding="utf-8")
    report = json.loads(text)
    assert report["overall_status"] == "failed"
    assert report["raw_data_fingerprint"] is None
    assert report["datasets"]["raw"]["status"] == "failed"
    assert "NaN" not in text
