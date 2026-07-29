"""Labeled-development and inference validation contracts."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR, ROOT, load_params
from src.evidence import write_json

try:
    import pandera.pandas as pa
    from pandera.pandas import Check, Column
except ImportError:  # pragma: no cover - older Pandera versions
    import pandera as pa
    from pandera import Check, Column

ContractName = Literal["labeled", "inference"]
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
LABELED_COLUMNS = [*FEATURE_COLUMNS, "Class"]
DEVELOPMENT_SLICES = ("train", "model_valid", "calibration")

_FINITE = Check(lambda series: bool(np.isfinite(series).all()))
_FEATURE_SCHEMA_COLUMNS = {
    "Time": Column(float, [Check.ge(0), _FINITE], coerce=True),
    **{
        f"V{i}": Column(float, _FINITE, nullable=False, coerce=True)
        for i in range(1, 29)
    },
    "Amount": Column(float, [Check.ge(0), _FINITE], coerce=True),
}
INFERENCE_SCHEMA = pa.DataFrameSchema(_FEATURE_SCHEMA_COLUMNS, strict=True)
LABELED_SCHEMA = pa.DataFrameSchema(
    {
        **_FEATURE_SCHEMA_COLUMNS,
        "Class": Column(int, Check.isin([0, 1]), coerce=True),
    },
    strict=True,
)


class DataValidationError(ValueError):
    """Raised when a data contract or its evidence gate is not satisfied."""


def _check(name: str, passed: bool, details: str = "") -> dict:
    return {"name": name, "passed": bool(passed), "details": details}


def _finite_range(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return {"min": None, "max": None}
    return {"min": float(finite.min()), "max": float(finite.max())}


def _is_non_negative_time_and_amount(df: pd.DataFrame) -> bool:
    if "Time" not in df or "Amount" not in df:
        return False
    values = df[["Time", "Amount"]].apply(pd.to_numeric, errors="coerce")
    return bool((values >= 0).all().all())


def _unavailable_slice_report(details: str) -> dict:
    return {
        "contract": "labeled",
        "status": "failed",
        "row_count": None,
        "columns": [],
        "dtypes": {},
        "null_counts": {},
        "ranges": {},
        "fraud_count": None,
        "fraud_rate": None,
        "checks": [_check("slice_unavailable", False, details)],
    }


def check_dataframe(
    df: pd.DataFrame,
    *,
    contract: ContractName,
    require_target_distribution: bool = False,
) -> dict:
    """Return evidence for one explicit validation contract without raising."""
    if contract not in {"labeled", "inference"}:
        raise DataValidationError(f"unknown validation contract: {contract}")

    target_present = "Class" in df.columns
    required = LABELED_COLUMNS if contract == "labeled" or target_present else FEATURE_COLUMNS
    pandera_contract = (
        LABELED_SCHEMA if contract == "labeled" or target_present else INFERENCE_SCHEMA
    )
    try:
        pandera_contract.validate(df, lazy=True)
        schema_passed = True
        schema_details = ""
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as error:
        schema_passed = False
        schema_details = str(error).splitlines()[0]

    target = pd.to_numeric(df["Class"], errors="coerce") if target_present else None
    fraud_count = int((target == 1).sum()) if target is not None else None
    fraud_rate = fraud_count / len(df) if target is not None and len(df) else None
    numeric = df.select_dtypes(include=[np.number])
    checks = [
        _check("pandera_schema", schema_passed, schema_details),
        _check(
            "complete_column_set",
            len(df.columns) == len(required) and set(df.columns) == set(required),
            f"expected {required}",
        ),
        _check(
            "numeric_and_finite",
            numeric.shape[1] == len(required) and bool(np.isfinite(numeric).all().all()),
        ),
        _check("no_nulls", not df.isna().any().any()),
        _check("non_negative_time_and_amount", _is_non_negative_time_and_amount(df)),
        _check(
            "binary_target",
            not target_present or set(df["Class"].dropna().unique()).issubset({0, 1}),
        ),
    ]
    if require_target_distribution:
        checks.extend(
            [
                _check("both_target_classes", target_present and df["Class"].nunique() == 2),
                _check(
                    "fraud_rate_in_range",
                    fraud_rate is not None and 0 < fraud_rate <= 0.05,
                    "required range is 0 < fraud_rate <= 0.05",
                ),
            ]
        )

    ranges = {
        column: _finite_range(df[column])
        for column in ("Time", "Amount", "Class")
        if column in df
    }
    return {
        "contract": contract,
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "row_count": len(df),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "null_counts": {column: int(count) for column, count in df.isna().sum().items()},
        "ranges": ranges,
        "fraud_count": fraud_count,
        "fraud_rate": fraud_rate,
        "checks": checks,
    }


def require_valid_dataframe(
    df: pd.DataFrame,
    *,
    contract: ContractName,
    require_target_distribution: bool = False,
) -> pd.DataFrame:
    report = check_dataframe(
        df,
        contract=contract,
        require_target_distribution=require_target_distribution,
    )
    if report["status"] != "passed":
        failed = [item["name"] for item in report["checks"] if not item["passed"]]
        raise DataValidationError(f"validation failed: {', '.join(failed)}")
    return df


def validate_file(
    path: str | Path,
    params: dict,
    report_path: Path = REPORTS_DIR / "validation_report.json",
) -> dict:
    """Write all required validation evidence before failing invalid data."""
    from src.ingest import split_by_time

    raw = pd.read_csv(path)
    datasets = {
        "raw": check_dataframe(
            raw, contract="labeled", require_target_distribution=True
        )
    }
    try:
        slices = split_by_time(raw, params)
    except Exception as error:
        details = f"time split unavailable: {error}"
        for name in DEVELOPMENT_SLICES:
            datasets[name] = _unavailable_slice_report(details)
    else:
        for name in DEVELOPMENT_SLICES:
            datasets[name] = check_dataframe(
                slices[name], contract="labeled", require_target_distribution=True
            )
    payload = {
        "overall_status": (
            "passed"
            if all(report["status"] == "passed" for report in datasets.values())
            else "failed"
        ),
        "datasets": datasets,
    }
    write_json(report_path, payload)
    if payload["overall_status"] != "passed":
        raise DataValidationError("raw or development-slice validation failed")
    return payload


def require_validation_gate(
    report_path: Path = REPORTS_DIR / "validation_report.json",
) -> dict:
    if not report_path.exists():
        raise DataValidationError(f"validation evidence missing: {report_path}")
    report = json.loads(report_path.read_text())
    required = {"raw", *DEVELOPMENT_SLICES}
    datasets = report.get("datasets")
    completed = isinstance(datasets, Mapping) and all(
        isinstance(datasets.get(name), Mapping)
        and datasets[name].get("status") == "passed"
        for name in required
    )
    if (
        report.get("overall_status") != "passed"
        or not isinstance(datasets, Mapping)
        or not required.issubset(datasets)
        or not completed
    ):
        raise DataValidationError("validation report is failed or incomplete")
    return report


def main() -> None:
    params = load_params()
    path = ROOT / params["data"]["raw_path"]
    report = validate_file(path, params)
    print(
        f"OK: {path} and development slices passed validation "
        f"({sum(item['row_count'] for item in report['datasets'].values()):,} checked rows)"
    )


if __name__ == "__main__":
    main()
