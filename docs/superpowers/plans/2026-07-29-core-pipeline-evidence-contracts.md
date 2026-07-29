# Core Pipeline Evidence Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a validation-gated, six-slice fraud pipeline whose model, operating point, predictions, monitoring, retraining-review decisions, and dashboard are all auditable through stable JSON contracts.

**Architecture:** Keep the existing stage modules and replace ambiguous or duplicated contracts in place. Model selection uses `model_valid`, threshold selection and baseline evaluation use `calibration`, production monitoring uses three later slices, and the dashboard renders structured evidence without recomputing decisions.

**Tech Stack:** Python 3.13.9, pandas, NumPy, Pandera, scikit-learn, XGBoost, MLflow 3.14.0, Matplotlib, Evidently, Pytest, and JSON.

## Global Constraints

- The split is exactly 50% train, 10% model validation, 10% calibration, and three 10% production batches.
- Preserve all rows; production slice sizes may differ by at most one row.
- Time ordering is non-decreasing because genuine `Time` values may repeat.
- Labeled train, model-validation, and calibration data must contain both classes and a fraud rate greater than 0 and no more than 5%.
- Inference accepts the full feature schema with or without `Class`.
- Model promotion is based only on model-validation PR-AUC.
- Threshold selection is based only on the calibration slice and costs false negatives at 100 versus false positives at 1 unless `params.yaml` explicitly changes them.
- Operating-threshold metrics are primary; threshold 0.5 is comparison-only.
- MLflow registry failure is fatal and must occur before local promotion evidence is published.
- Pending labels make target/performance metrics unavailable; feature and prediction drift remain available.
- Retraining thresholds are PR-AUC drop greater than 10%, recall below 0.75, and feature drift greater than 50%; 25–50% drift is warning.
- Monitoring recommends human review and never automatically starts training or replaces a model.
- JSON writers must use `allow_nan=False`.
- The full test suite must pass with zero skips.
- Do not edit `dvc.yaml` or `.github/workflows/ci.yml` in this plan; their final paths belong to the reproducibility plan.

---

## File Structure

**Create**

- `src/evidence.py` — stable SHA-256 and strict JSON primitives; the reproducibility plan later adds the run manifest.
- `tests/test_train.py` — MLflow candidate and registry integration tests.
- `tests/test_trigger.py` — structured decision and annotation tests.

**Modify**

- `params.yaml` — exact six-slice fractions and existing thresholds.
- `src/validate.py` — labeled/inference contracts and validation report.
- `src/ingest.py` — six chronological slices and validation gate.
- `src/train.py` — candidate run traceability and promotion evidence.
- `src/artifacts.py` — runtime model/operating-point helpers.
- `src/threshold.py` — calibration-only operating point.
- `src/evaluate.py` — cost metrics and default/operating evidence.
- `src/batch_inference.py` — audited labeled/unlabeled scoring.
- `src/drift.py` — label-aware monitoring and visible Evidently errors.
- `src/retrain_trigger.py` — decision JSON, Markdown renderer, and annotations.
- `src/build_dashboard.py` — evidence-only rendering with injectable directories.
- Existing unit-test files under `tests/`.

**Generate, never hand-author**

- `reports/validation_report.json`
- `reports/experiment_comparison.json`
- `reports/promotion_record.json`
- `reports/operating_point.json`
- `reports/metrics_default.json`
- `reports/metrics_operating.json`
- `reports/figures/confusion_matrix_default.png`
- `reports/figures/confusion_matrix_operating.png`
- `reports/trigger_decisions.json`
- `reports/trigger_log.md`

**Retire**

- `data/batches/valid.csv`
- `models/baseline.json`
- `reports/metrics_valid.json`
- `reports/figures/confusion_matrix.png`

### Task 1: Strict Evidence Primitives

**Files:**
- Create: `src/evidence.py`
- Create: `tests/test_evidence.py`

**Interfaces:**
- Consumes: `pathlib.Path` and JSON-compatible dictionaries
- Produces: `sha256_file(path: str | Path) -> str`
- Produces: `write_json(path: str | Path, payload: dict | list) -> Path`

- [ ] **Step 1: Write the failing evidence tests**

```python
# tests/test_evidence.py
import json

import pytest

from src.evidence import sha256_file, write_json


def test_sha256_file_is_stable(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"fraud-mlops\n")
    assert sha256_file(source) == (
        "bd69cbe6d0148a4dbec170d60a316d5dbad2da4bf8b0584f1dcb095e3d2bdd33"
    )


def test_write_json_is_sorted_and_rejects_nan(tmp_path):
    target = tmp_path / "nested" / "evidence.json"
    assert write_json(target, {"z": 2, "a": 1}) == target
    assert target.read_text() == '{\n  "a": 1,\n  "z": 2\n}\n'
    assert json.loads(target.read_text()) == {"a": 1, "z": 2}
    with pytest.raises(ValueError):
        write_json(target, {"metric": float("nan")})
```

- [ ] **Step 2: Run the tests and confirm the missing module**

Run: `pytest tests/test_evidence.py -q`

Expected: collection fails because `src.evidence` does not exist.

- [ ] **Step 3: Implement the strict helpers**

```python
# src/evidence.py
"""Stable, portable evidence helpers shared by pipeline stages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict | list) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.write_text(text, encoding="utf-8")
    return target
```

- [ ] **Step 4: Verify the digest fixture and pass the tests**

Run:

```bash
python -c "from hashlib import sha256; print(sha256(b'fraud-mlops\n').hexdigest())"
pytest tests/test_evidence.py -q
```

Expected: the printed digest matches the fixture and both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/evidence.py tests/test_evidence.py
git commit -m "feat: add strict evidence primitives"
```

### Task 2: Six-Way Chronological Segmentation

**Files:**
- Modify: `params.yaml:1-15`
- Modify: `src/ingest.py:12-31`
- Modify: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `params["data"]` containing `train_frac`, `model_valid_frac`, `calibration_frac`, and `n_prod_batches`
- Produces: `SPLIT_NAMES: tuple[str, ...]`
- Produces: `split_by_time(df: pd.DataFrame, params: dict) -> dict[str, pd.DataFrame]`

- [ ] **Step 1: Replace the split test with exact boundaries and remainder checks**

```python
# tests/test_ingest.py
import numpy as np
import pandas as pd

from src.ingest import SPLIT_NAMES, inject_drift, split_by_time

PARAMS = {
    "data": {
        "train_frac": 0.5,
        "model_valid_frac": 0.1,
        "calibration_frac": 0.1,
        "n_prod_batches": 3,
    }
}


def _df(n=100):
    return pd.DataFrame(
        {"row_id": np.arange(n), "Time": np.repeat(np.arange((n + 1) // 2), 2)[:n], "Class": 0}
    )


def test_split_has_six_chronological_disjoint_slices():
    parts = split_by_time(_df(100).sample(frac=1, random_state=4), PARAMS)
    assert tuple(parts) == SPLIT_NAMES
    assert [len(parts[name]) for name in SPLIT_NAMES] == [50, 10, 10, 10, 10, 10]
    joined = pd.concat(parts.values(), ignore_index=True)
    assert len(joined) == 100
    assert joined["row_id"].is_unique
    assert joined["Time"].is_monotonic_increasing


def test_split_preserves_remainder_rows():
    parts = split_by_time(_df(103), PARAMS)
    assert sum(map(len, parts.values())) == 103
    production_sizes = [len(parts[f"prod_{i}"]) for i in range(1, 4)]
    assert max(production_sizes) - min(production_sizes) <= 1
```

Retain the existing drift-injection test beneath these tests.

- [ ] **Step 2: Run the two split tests and confirm failure**

Run: `pytest tests/test_ingest.py -q`

Expected: the old `valid` key and five-slice lengths fail.

- [ ] **Step 3: Change the parameters and splitting function**

Replace the old `valid_frac` setting:

```yaml
data:
  raw_path: data/raw/creditcard.csv
  batch_dir: data/batches
  train_frac: 0.5
  model_valid_frac: 0.1
  calibration_frac: 0.1
  n_prod_batches: 3
```

Replace `split_by_time` with:

```python
SPLIT_NAMES = (
    "train",
    "model_valid",
    "calibration",
    "prod_1",
    "prod_2",
    "prod_3",
)


def split_by_time(df: pd.DataFrame, params: dict) -> dict[str, pd.DataFrame]:
    ordered = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n_rows = len(ordered)
    data_params = params["data"]
    train_end = int(n_rows * data_params["train_frac"])
    model_valid_end = train_end + int(n_rows * data_params["model_valid_frac"])
    calibration_end = model_valid_end + int(n_rows * data_params["calibration_frac"])
    parts = {
        "train": ordered.iloc[:train_end],
        "model_valid": ordered.iloc[train_end:model_valid_end],
        "calibration": ordered.iloc[model_valid_end:calibration_end],
    }
    production = ordered.iloc[calibration_end:]
    n_batches = data_params["n_prod_batches"]
    for index in range(n_batches):
        start = index * len(production) // n_batches
        end = (index + 1) * len(production) // n_batches
        parts[f"prod_{index + 1}"] = production.iloc[start:end]
    if tuple(parts) != SPLIT_NAMES or sum(map(len, parts.values())) != n_rows:
        raise ValueError("six-way split did not preserve the complete dataset")
    return {name: frame.copy().reset_index(drop=True) for name, frame in parts.items()}
```

- [ ] **Step 4: Run the segmentation tests**

Run: `pytest tests/test_ingest.py -q`

Expected: all ingestion tests pass.

- [ ] **Step 5: Commit**

```bash
git add params.yaml src/ingest.py tests/test_ingest.py
git commit -m "feat: add six-way chronological split"
```

### Task 3: Labeled and Inference Validation Contracts

**Files:**
- Modify: `src/validate.py`
- Modify: `src/ingest.py`
- Modify: `tests/test_validate.py`

**Interfaces:**
- Consumes: `ContractName = Literal["labeled", "inference"]`
- Produces: `DataValidationError(ValueError)`
- Produces: `check_dataframe(df, *, contract, require_target_distribution=False) -> dict`
- Produces: `require_valid_dataframe(df, *, contract, require_target_distribution=False) -> pd.DataFrame`
- Produces: `validate_file(path, params, report_path=REPORTS_DIR / "validation_report.json") -> dict`
- Produces: `require_validation_gate(report_path=REPORTS_DIR / "validation_report.json") -> dict`

- [ ] **Step 1: Write failing contract and report tests**

```python
# tests/test_validate.py
import json

import numpy as np
import pandas as pd
import pytest

from src.validate import (
    DataValidationError,
    check_dataframe,
    require_valid_dataframe,
    require_validation_gate,
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


def test_validation_gate_rejects_failed_or_incomplete_report(tmp_path):
    path = tmp_path / "validation_report.json"
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
```

- [ ] **Step 2: Run validation tests and confirm the old schema is insufficient**

Run: `pytest tests/test_validate.py -q`

Expected: imports for the new contract functions fail.

- [ ] **Step 3: Implement report-first validation**

Implement these constants and public functions in `src/validate.py`. Retain
Pandera as the executable schema layer, with separate labeled and inference
schemas rather than one ambiguous global `schema`:

```python
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR
from src.evidence import write_json

try:
    import pandera.pandas as pa
    from pandera.pandas import Check, Column
except ImportError:
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
    pass


def _check(name: str, passed: bool, details: str = "") -> dict:
    return {"name": name, "passed": bool(passed), "details": details}


def check_dataframe(
    df: pd.DataFrame,
    *,
    contract: ContractName,
    require_target_distribution: bool = False,
) -> dict:
    target_present = "Class" in df.columns
    required = LABELED_COLUMNS if contract == "labeled" or target_present else FEATURE_COLUMNS
    numeric = df.select_dtypes(include=[np.number])
    schema = LABELED_SCHEMA if contract == "labeled" or target_present else INFERENCE_SCHEMA
    try:
        schema.validate(df, lazy=True)
        schema_passed = True
        schema_details = ""
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as error:
        schema_passed = False
        schema_details = str(error).splitlines()[0]
    fraud_count = int(df["Class"].sum()) if target_present else None
    fraud_rate = float(df["Class"].mean()) if target_present and len(df) else None
    checks = [
        _check("pandera_schema", schema_passed, schema_details),
        _check(
            "complete_column_set",
            len(df.columns) == len(required) and set(df.columns) == set(required),
            f"expected {required}",
        ),
        _check("numeric_and_finite", numeric.shape[1] == len(required) and np.isfinite(numeric).all().all()),
        _check("no_nulls", not df.isna().any().any()),
        _check(
            "non_negative_time_and_amount",
            "Time" in df and "Amount" in df and bool((df[["Time", "Amount"]] >= 0).all().all()),
        ),
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
    status = "passed" if all(item["passed"] for item in checks) else "failed"
    ranges = {}
    for column in ("Time", "Amount", "Class"):
        if column in df:
            ranges[column] = {
                "min": float(df[column].min()),
                "max": float(df[column].max()),
            }
    return {
        "contract": contract,
        "status": status,
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


def require_validation_gate(
    report_path: Path = REPORTS_DIR / "validation_report.json",
) -> dict:
    if not report_path.exists():
        raise DataValidationError(f"validation evidence missing: {report_path}")
    report = json.loads(report_path.read_text())
    required = {"raw", *DEVELOPMENT_SLICES}
    if report.get("overall_status") != "passed" or not required.issubset(
        report.get("datasets", {})
    ):
        raise DataValidationError("validation report is failed or incomplete")
    return report
```

Implement `validate_file` by reading raw data, calling `split_by_time` through a
local import, building reports for `raw`, `train`, `model_valid`, and
`calibration`, writing the combined report before raising, and returning the
report when passed:

```python
def validate_file(
    path: str | Path,
    params: dict,
    report_path: Path = REPORTS_DIR / "validation_report.json",
) -> dict:
    from src.ingest import split_by_time

    raw = pd.read_csv(path)
    slices = split_by_time(raw, params)
    datasets = {
        "raw": check_dataframe(
            raw, contract="labeled", require_target_distribution=True
        )
    }
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
```

Remove the unused `math` import. Keep Pandera as the schema enforcement layer;
the explicit report checks exist to produce assessor-readable evidence, not to
replace Pandera.

- [ ] **Step 4: Gate ingestion on completed validation evidence**

At the start of `src.ingest.main`, before reading or writing batches:

```python
from src.validate import require_validation_gate

require_validation_gate()
```

Update `src.validate.main` to call:

```python
params = load_params()
path = ROOT / params["data"]["raw_path"]
report = validate_file(path, params)
print(
    f"OK: {path} and development slices passed validation "
    f"({sum(item['row_count'] for item in report['datasets'].values()):,} checked rows)"
)
```

- [ ] **Step 5: Run the validation and ingestion tests**

Run:

```bash
pytest tests/test_validate.py tests/test_ingest.py -q
```

Expected: all tests pass and failed report fixtures contain no `NaN`.

- [ ] **Step 6: Commit**

```bash
git add src/validate.py src/ingest.py tests/test_validate.py
git commit -m "feat: add validation contracts and evidence gate"
```

### Task 4: Traceable MLflow Candidate Promotion

**Files:**
- Modify: `src/train.py`
- Modify: `src/features.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: train/model-validation frames, provenance, data fingerprint, parameter fingerprint
- Produces: `features.save_scaler(scaler) -> Path`
- Produces: `normalise_mlflow_param(value) -> str | int | float | bool`
- Produces: `log_candidate(name, model, X_train, y_train, X_valid, y_valid, metadata, scaler_path) -> dict`
- Produces: `choose_promoted_candidate(candidates: list[dict]) -> dict`
- Produces: `register_winner(candidate: dict, registered_model_name=REGISTERED_MODEL) -> str`

- [ ] **Step 1: Write candidate-selection and real-registry tests**

```python
# tests/test_train.py
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.train import (
    choose_promoted_candidate,
    log_candidate,
    register_winner,
)


def test_choose_promoted_candidate_uses_pr_auc():
    candidates = [
        {"model_name": "first", "model_validation_metrics": {"pr_auc": 0.71}},
        {"model_name": "second", "model_validation_metrics": {"pr_auc": 0.82}},
    ]
    assert choose_promoted_candidate(candidates)["model_name"] == "second"


def test_candidate_run_is_registered_without_a_second_run(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-fraud")
    X = np.array([[0.0], [0.2], [0.8], [1.0], [1.2], [1.4]])
    y = np.array([0, 0, 0, 1, 1, 1])
    scaler_path = tmp_path / "scaler.joblib"
    joblib.dump(StandardScaler().fit(X), scaler_path)
    model = LogisticRegression(random_state=42, class_weight="balanced")
    result = log_candidate(
        "logistic_regression",
        model,
        X,
        y,
        X,
        y,
        {
            "source": "unit-test",
            "data_fingerprint": "a" * 64,
            "parameter_fingerprint": "b" * 64,
            "imbalance": "balanced",
            "cost_false_negative": 100,
            "cost_false_positive": 1,
        },
        scaler_path,
    )
    version = register_winner(result, registered_model_name="test-fraud-detector")
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(result["run_id"])
    registered = client.get_model_version("test-fraud-detector", version)
    assert result["run_id"] == registered.run_id
    assert "estimator.C" in run.data.params
    assert run.data.tags["data_fingerprint"] == "a" * 64
    assert client.get_model_version_by_alias(
        "test-fraud-detector", "champion"
    ).version == version
```

- [ ] **Step 2: Run training tests and confirm missing interfaces**

Run: `pytest tests/test_train.py -q`

Expected: imports fail for the new training helpers.

- [ ] **Step 3: Implement candidate logging and registration**

Add to `src/train.py`:

```python
from mlflow import MlflowClient


def normalise_mlflow_param(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return "None" if value is None else value
    return json.dumps(value, sort_keys=True, default=str)


def log_candidate(
    name,
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    metadata,
    scaler_path,
) -> dict:
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_valid)[:, 1]
        metrics = compute_metrics(
            y_valid,
            probability,
            threshold=0.5,
            cost_false_negative=metadata["cost_false_negative"],
            cost_false_positive=metadata["cost_false_positive"],
        )
        params = {
            f"estimator.{key}": normalise_mlflow_param(value)
            for key, value in model.get_params(deep=False).items()
        }
        mlflow.log_params(params)
        mlflow.log_param("model_name", name)
        mlflow.log_param("imbalance", metadata["imbalance"])
        mlflow.set_tags(
            {
                "source_provenance": metadata["source"],
                "data_fingerprint": metadata["data_fingerprint"],
                "parameter_fingerprint": metadata["parameter_fingerprint"],
            }
        )
        mlflow.log_metrics(
            {
                f"model_validation.{key}": value
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and np.isfinite(value)
            }
        )
        mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")
        model_info = mlflow.sklearn.log_model(
            model, name="model", serialization_format="cloudpickle"
        )
        return {
            "model_name": name,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "model_validation_metrics": metrics,
            "model": model,
        }


def choose_promoted_candidate(candidates: list[dict]) -> dict:
    return max(
        candidates,
        key=lambda candidate: candidate["model_validation_metrics"]["pr_auc"],
    )


def register_winner(
    candidate: dict,
    registered_model_name: str = REGISTERED_MODEL,
) -> str:
    registered = mlflow.register_model(candidate["model_uri"], registered_model_name)
    client = MlflowClient()
    version = str(registered.version)
    client.set_registered_model_alias(registered_model_name, "champion", version)
    return version
```

Add `import numpy as np`. Do not wrap `register_winner` in `try/except`.

- [ ] **Step 4: Rework `main` around the two evidence contracts**

Make `features.save_scaler` return its written path:

```python
def save_scaler(scaler: StandardScaler) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, _SCALER_PATH)
    return _SCALER_PATH
```

Add `from pathlib import Path` to `src/features.py`. In training, read
`model_valid.csv` and derive metadata exactly:

```python
raw_path = ROOT / params["data"]["raw_path"]
provenance = read_provenance(params)
metadata = {
    "source": json.dumps(provenance, sort_keys=True),
    "data_fingerprint": sha256_file(raw_path),
    "parameter_fingerprint": sha256_file(ROOT / "params.yaml"),
    "imbalance": params["train"]["class_weight"],
    "cost_false_negative": params["threshold"]["cost_false_negative"],
    "cost_false_positive": params["threshold"]["cost_false_positive"],
}
```

Use a temporary scaler artifact for candidate runs. Only publish the runtime
model/scaler and evidence after registry success:

```python
with tempfile.TemporaryDirectory() as staging_dir:
    staged_scaler = Path(staging_dir) / "scaler.joblib"
    joblib.dump(scaler, staged_scaler)
    candidates = [
        log_candidate(
            name,
            build_model(name, params, pos_weight),
            X_train,
            y_train,
            X_model_valid,
            y_model_valid,
            metadata,
            staged_scaler,
        )
        for name in params["train"]["models"]
    ]
    winner = choose_promoted_candidate(candidates)
    version = register_winner(winner)

joblib.dump(winner["model"], MODELS_DIR / "model.joblib")
features.save_scaler(scaler)
```

Add `tempfile`, `Path`, `read_provenance`, `sha256_file`, and `write_json`
imports. Then create:

```python
comparison = {
    "selection_metric": "model_validation.pr_auc",
    "candidates": [
        {
            **{
                key: value
                for key, value in candidate.items()
                if key != "model"
            },
            "promoted": candidate["run_id"] == winner["run_id"],
        }
        for candidate in candidates
    ],
    "promoted_run_id": winner["run_id"],
}
promotion = {
    "promoted_model_id": f"{REGISTERED_MODEL}:v{version}",
    "model_name": winner["model_name"],
    "mlflow_run_id": winner["run_id"],
    "registered_model_name": REGISTERED_MODEL,
    "registered_model_version": version,
    "model_validation_metrics": winner["model_validation_metrics"],
    "source_provenance": provenance,
    "data_fingerprint": metadata["data_fingerprint"],
    "parameter_fingerprint": metadata["parameter_fingerprint"],
    "selection_reason": "highest model-validation PR-AUC",
}
```

Call `register_winner` before dumping the local model or writing either JSON.
Then write `reports/experiment_comparison.json` and
`reports/promotion_record.json` with `write_json`. Remove the old third
promotion run and `models/baseline.json`.

- [ ] **Step 5: Verify registry failure propagates**

Add:

```python
def test_registry_failure_propagates(monkeypatch):
    def fail_registration(model_uri, registered_model_name):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(mlflow, "register_model", fail_registration)
    with pytest.raises(RuntimeError, match="registry unavailable"):
        register_winner(
            {"model_uri": "runs:/missing/model"},
            registered_model_name="test-fraud-detector",
        )
```

Run `pytest tests/test_train.py -q`, and expect all tests to pass.

- [ ] **Step 6: Commit**

```bash
git add src/train.py src/features.py tests/test_train.py
git commit -m "feat: add traceable MLflow promotion"
```

### Task 5: Calibration-Only Operating Point and Dual Evaluation

**Files:**
- Modify: `src/evaluate.py`
- Modify: `src/threshold.py`
- Modify: `src/artifacts.py`
- Modify: `tests/test_metrics.py`
- Delete: `reports/metrics_valid.json`
- Delete: `reports/figures/confusion_matrix.png`

**Interfaces:**
- Produces: `compute_metrics(y_true, proba, threshold, *, cost_false_negative, cost_false_positive) -> dict`
- Produces: `build_operating_point(y_true, proba, *, cost_false_negative, cost_false_positive, promotion_record, calibration_data_fingerprint) -> dict`
- Produces: `evaluate_probabilities(y_true, proba, operating_point) -> tuple[dict, dict]`
- Produces: `save_operating_point(payload: dict) -> None`

- [ ] **Step 1: Write hand-derived cost and dual-view tests**

```python
# tests/test_metrics.py
from src.evaluate import compute_metrics, evaluate_probabilities
from src.threshold import build_operating_point, select_threshold


def test_metrics_include_hand_derived_confusion_and_cost():
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


def test_evaluation_returns_default_and_operating_views():
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
```

- [ ] **Step 2: Run metrics tests and observe signature failures**

Run: `pytest tests/test_metrics.py -q`

Expected: `compute_metrics` rejects the new cost arguments and
`evaluate_probabilities` is missing.

- [ ] **Step 3: Extend shared metrics and evaluation views**

Change `_safe_auc` so strict JSON never receives `NaN`, then replace
`compute_metrics` with the existing metrics plus:

```python
def _safe_auc(fn, y_true, proba) -> float | None:
    return float(fn(y_true, proba)) if len(np.unique(y_true)) > 1 else None


def compute_metrics(
    y_true,
    proba,
    threshold: float,
    *,
    cost_false_negative: float,
    cost_false_positive: float,
) -> dict:
    probability = np.asarray(proba)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "pr_auc": _safe_auc(average_precision_score, y_true, probability),
        "roc_auc": _safe_auc(roc_auc_score, y_true, probability),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold": float(threshold),
        "estimated_business_cost": float(
            cost_false_negative * fn + cost_false_positive * fp
        ),
    }


def evaluate_probabilities(y_true, proba, operating_point: dict) -> tuple[dict, dict]:
    costs = operating_point["cost_assumptions"]
    common = {
        "cost_false_negative": costs["false_negative"],
        "cost_false_positive": costs["false_positive"],
    }
    default = compute_metrics(y_true, proba, 0.5, **common)
    operating = compute_metrics(y_true, proba, operating_point["threshold"], **common)
    default.update(
        {
            "evidence_role": "comparison_default_threshold",
            "promoted_model_id": operating_point["promoted_model_id"],
            "calibration_data_fingerprint": operating_point[
                "calibration_data_fingerprint"
            ],
        }
    )
    operating.update(
        {
            "evidence_role": "primary_operating_point",
            "promoted_model_id": operating_point["promoted_model_id"],
            "calibration_data_fingerprint": operating_point[
                "calibration_data_fingerprint"
            ],
        }
    )
    return default, operating
```

Update every `compute_metrics` caller and existing metric test with configured
costs. In particular, the perfect-classifier test calls:

```python
m = compute_metrics(
    [0, 0, 1, 1],
    [0.1, 0.2, 0.8, 0.9],
    threshold=0.5,
    cost_false_negative=100,
    cost_false_positive=1,
)
```

- [ ] **Step 4: Build and save the operating-point contract**

Add to `src/threshold.py`:

```python
def build_operating_point(
    y_true,
    proba,
    *,
    cost_false_negative: float,
    cost_false_positive: float,
    promotion_record: dict,
    calibration_data_fingerprint: str,
) -> dict:
    threshold, cost = select_threshold(
        y_true,
        proba,
        cost_false_negative,
        cost_false_positive,
    )
    metrics = compute_metrics(
        y_true,
        proba,
        threshold,
        cost_false_negative=cost_false_negative,
        cost_false_positive=cost_false_positive,
    )
    return {
        "threshold": threshold,
        "cost_assumptions": {
            "false_negative": float(cost_false_negative),
            "false_positive": float(cost_false_positive),
        },
        "estimated_business_cost": cost,
        "calibration_metrics": metrics,
        "promoted_model_id": promotion_record["promoted_model_id"],
        "calibration_data_fingerprint": calibration_data_fingerprint,
    }
```

Change `main` to read `calibration.csv`, build the payload, and call
`save_operating_point`. In `src/artifacts.py`:

```python
def save_operating_point(payload: dict) -> None:
    from src.config import REPORTS_DIR
    from src.evidence import write_json

    write_json(_THRESHOLD_PATH, payload)
    write_json(REPORTS_DIR / "operating_point.json", payload)
```

- [ ] **Step 5: Emit both evaluation views and figures**

Change `src.evaluate.main` to read `calibration.csv`, call
`evaluate_probabilities`, and write:

```python
write_json(REPORTS_DIR / "metrics_default.json", default_metrics)
write_json(REPORTS_DIR / "metrics_operating.json", operating_metrics)
_plot_confusion(
    y,
    (proba >= 0.5).astype(int),
    FIGURES_DIR / "confusion_matrix_default.png",
    "Confusion matrix — default threshold 0.5",
)
_plot_confusion(
    y,
    (proba >= operating_point["threshold"]).astype(int),
    FIGURES_DIR / "confusion_matrix_operating.png",
    f"Confusion matrix — operating threshold {operating_point['threshold']:.2f}",
)
```

Add a `title` argument to `_plot_confusion`. Keep one calibration PR curve and
the threshold trade-off figure, both clearly labeled as calibration evidence.

- [ ] **Step 6: Remove ambiguous tracked evidence and pass tests**

Run:

```bash
git rm reports/metrics_valid.json reports/figures/confusion_matrix.png
pytest tests/test_metrics.py tests/test_train.py -q
```

Expected: all tests pass; code contains no `valid.csv`, `metrics_valid.json`, or
unnamed `confusion_matrix.png` references.

- [ ] **Step 7: Commit**

```bash
git add src/evaluate.py src/threshold.py src/artifacts.py tests/test_metrics.py
git commit -m "feat: make operating-point evidence primary"
```

### Task 6: Audited Labeled and Unlabeled Inference

**Files:**
- Modify: `src/batch_inference.py`
- Modify: `tests/test_inference.py`

**Interfaces:**
- Consumes: inference frame, model, scaler, threshold, promoted model ID
- Produces: `score_batch(df, model, scaler, *, threshold, promoted_model_id) -> pd.DataFrame`

- [ ] **Step 1: Replace inference tests with audit and gate assertions**

```python
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
    assert "Class" not in out
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
```

Add the required `pytest` and validation imports.

- [ ] **Step 2: Run inference tests and confirm signature failures**

Run: `pytest tests/test_inference.py -q`

Expected: `score_batch` rejects the keyword-only audit arguments.

- [ ] **Step 3: Validate inside the scoring boundary and add audit columns**

```python
def score_batch(
    df: pd.DataFrame,
    model,
    scaler,
    *,
    threshold: float,
    promoted_model_id: str,
) -> pd.DataFrame:
    contract = "labeled" if features.TARGET in df else "inference"
    require_valid_dataframe(
        df,
        contract=contract,
        require_target_distribution=False,
    )
    transformed = features.transform(df, scaler)
    proba = model.predict_proba(transformed[features.FEATURES])[:, 1]
    out = df[["Time", "Amount"]].copy()
    out["proba"] = proba
    out["pred"] = (proba >= threshold).astype(int)
    out["operating_threshold"] = float(threshold)
    out["promoted_model_id"] = promoted_model_id
    if features.TARGET in df:
        out[features.TARGET] = df[features.TARGET].values
    return out
```

In `main`, load the model ID from `reports/promotion_record.json` and remove the
old direct labeled-schema validation.

- [ ] **Step 4: Run inference and validation tests**

Run: `pytest tests/test_inference.py tests/test_validate.py -q`

Expected: all tests pass and the fake model is never called on invalid data.

- [ ] **Step 5: Commit**

```bash
git add src/batch_inference.py tests/test_inference.py
git commit -m "feat: audit labeled and unlabeled inference"
```

### Task 7: Label-Aware Monitoring with Visible Optional Failures

**Files:**
- Modify: `src/drift.py`
- Modify: `tests/test_drift.py`

**Interfaces:**
- Produces: `summarize_batch(reference, current, reference_proba, predictions, *, batch_name, threshold, params) -> dict`
- Produces: `evidently_report(reference, current, path) -> dict`

- [ ] **Step 1: Add labeled and pending-label monitoring tests**

```python
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
```

- [ ] **Step 2: Run drift tests and confirm the summary helper is missing**

Run: `pytest tests/test_drift.py -q`

Expected: import or name failure for `summarize_batch`.

- [ ] **Step 3: Implement the label-aware summary**

```python
def summarize_batch(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_proba: np.ndarray,
    predictions: pd.DataFrame,
    *,
    batch_name: str,
    threshold: float,
    params: dict,
) -> dict:
    drift = feature_drift(reference, current, params)
    labels_available = features.TARGET in predictions
    performance = None
    if labels_available:
        costs = params["threshold"]
        performance = compute_metrics(
            predictions[features.TARGET],
            predictions["proba"],
            threshold,
            cost_false_negative=costs["cost_false_negative"],
            cost_false_positive=costs["cost_false_positive"],
        )
    target_rate = (
        float(predictions[features.TARGET].mean()) if labels_available else None
    )
    baseline_target_rate = float(reference[features.TARGET].mean())
    return {
        "batch": batch_name,
        "n_rows": len(current),
        "label_status": "available" if labels_available else "pending",
        "pct_drifted_features": drift["pct_drifted_features"],
        "n_drifted": drift["n_drifted"],
        "amount_psi": drift["per_feature"]["Amount"]["psi"],
        "prediction_psi": round(
            psi(reference_proba, predictions["proba"].to_numpy()), 4
        ),
        "target_rate": target_rate,
        "baseline_target_rate": baseline_target_rate if labels_available else None,
        "target_rate_delta": (
            target_rate - baseline_target_rate if labels_available else None
        ),
        "pr_auc": performance["pr_auc"] if performance else None,
        "precision": performance["precision"] if performance else None,
        "recall": performance["recall"] if performance else None,
        "estimated_business_cost": (
            performance["estimated_business_cost"] if performance else None
        ),
        "per_feature": drift["per_feature"],
    }
```

Use the calibration slice for `reference_proba` and performance baseline; keep
the train slice as the feature-drift reference.

- [ ] **Step 4: Return Evidently status and errors instead of a Boolean**

Replace `_evidently_html` with:

```python
def evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    path,
) -> dict:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        columns = features.FEATURES.copy()
        if features.TARGET in reference and features.TARGET in current:
            columns.append(features.TARGET)
        snapshot = Report([DataDriftPreset()]).run(
            reference_data=reference[columns],
            current_data=current[columns],
        )
        snapshot.save_html(str(path))
        return {"status": "generated", "error_type": None, "message": None}
    except Exception as error:
        return {
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
        }
```

Attach the returned object to each per-batch JSON as `evidently`. Native JSON
generation remains required even when this object reports failure. Write the
full result, including `per_feature`, to `reports/drift/<batch>.json`; write a
copy without `per_feature` to the top-level `reports/drift_summary.json` list:

```python
full_summary["evidently"] = evidently_report(train, current, html_path)
write_json(DRIFT_DIR / f"{name}.json", full_summary)
summaries.append(
    {key: value for key, value in full_summary.items() if key != "per_feature"}
)
```

- [ ] **Step 5: Run monitoring tests**

Run: `pytest tests/test_drift.py -q`

Expected: all tests pass; serializing the pending-label summary with
`allow_nan=False` succeeds.

- [ ] **Step 6: Commit**

```bash
git add src/drift.py tests/test_drift.py
git commit -m "feat: make monitoring label aware"
```

### Task 8: Structured Retraining-Review Decisions

**Files:**
- Modify: `src/retrain_trigger.py`
- Create: `tests/test_trigger.py`
- Modify: `tests/test_metrics.py`

**Interfaces:**
- Produces: `evaluate_batch(baseline_pr_auc, batch, params) -> dict`
- Produces: `build_decision_report(batches, operating_point, params) -> dict`
- Produces: `render_markdown(report: dict) -> str`
- Produces: `github_annotation(report: dict) -> str | None`

- [ ] **Step 1: Write boundary, pending-label, and overall-status tests**

```python
# tests/test_trigger.py
from src.retrain_trigger import (
    build_decision_report,
    evaluate_batch,
    github_annotation,
)

PARAMS = {
    "trigger": {
        "pr_auc_drop_pct": 10,
        "recall_floor": 0.75,
        "drifted_features_pct": 50,
    }
}


def _batch(drift=0, pr_auc=0.9, recall=0.9, labels="available"):
    return {
        "batch": "prod_1",
        "label_status": labels,
        "pct_drifted_features": drift,
        "pr_auc": pr_auc if labels == "available" else None,
        "recall": recall if labels == "available" else None,
    }


def test_feature_drift_boundaries_are_calibrated():
    assert evaluate_batch(0.9, _batch(drift=24.9), PARAMS)["status"] == "ok"
    assert evaluate_batch(0.9, _batch(drift=25), PARAMS)["status"] == "warning"
    assert evaluate_batch(0.9, _batch(drift=50), PARAMS)["status"] == "warning"
    result = evaluate_batch(0.9, _batch(drift=50.1), PARAMS)
    assert result["status"] == "retrain"
    assert "FEATURE_DRIFT_RETRAIN" in result["reason_codes"]


def test_pending_labels_ignore_performance_rules():
    result = evaluate_batch(
        0.9,
        _batch(drift=10, labels="pending"),
        PARAMS,
    )
    assert result["status"] == "ok"
    assert result["reason_codes"] == ["LABELS_PENDING"]


def test_report_and_annotations_use_highest_severity():
    batches = [
        {**_batch(drift=0), "batch": "prod_1"},
        {**_batch(drift=30), "batch": "prod_2"},
        {**_batch(drift=60), "batch": "prod_3"},
    ]
    report = build_decision_report(
        batches,
        {
            "threshold": 0.42,
            "promoted_model_id": "fraud-detector:v3",
            "calibration_metrics": {"pr_auc": 0.9},
        },
        PARAMS,
    )
    assert report["overall_status"] == "retrain"
    assert github_annotation({"overall_status": "ok"}) is None
    assert github_annotation({"overall_status": "warning"}).startswith("::notice")
    assert github_annotation(report).startswith("::warning")
    assert "human approval" in github_annotation(report)
```

- [ ] **Step 2: Run trigger tests and confirm missing functions**

Run: `pytest tests/test_trigger.py -q`

Expected: imports fail for the structured interfaces.

- [ ] **Step 3: Implement stable reason codes and severity**

```python
_SEVERITY = {"ok": 0, "warning": 1, "retrain": 2}


def evaluate_batch(baseline_pr_auc: float, batch: dict, params: dict) -> dict:
    thresholds = params["trigger"]
    codes = []
    reasons = []
    if batch["label_status"] == "available":
        if batch["pr_auc"] is None:
            raise ValueError("PR-AUC is unavailable despite available batch labels")
        drop = 100 * (baseline_pr_auc - batch["pr_auc"]) / baseline_pr_auc
        if drop > thresholds["pr_auc_drop_pct"]:
            codes.append("PR_AUC_DROP")
            reasons.append(
                f"PR-AUC drop {drop:.1f}% exceeds {thresholds['pr_auc_drop_pct']}%"
            )
        if batch["recall"] < thresholds["recall_floor"]:
            codes.append("RECALL_BELOW_FLOOR")
            reasons.append(
                f"recall {batch['recall']:.3f} is below {thresholds['recall_floor']}"
            )
    else:
        codes.append("LABELS_PENDING")
        reasons.append("labels pending; performance rules were not evaluated")

    drift = batch["pct_drifted_features"]
    retrain_drift = thresholds["drifted_features_pct"]
    warning_drift = retrain_drift / 2
    if drift > retrain_drift:
        codes.append("FEATURE_DRIFT_RETRAIN")
        reasons.append(f"{drift:.1f}% feature drift exceeds {retrain_drift}%")
    elif drift >= warning_drift:
        codes.append("FEATURE_DRIFT_WARNING")
        reasons.append(f"{drift:.1f}% feature drift is in the warning band")

    if any(
        code in codes
        for code in ("PR_AUC_DROP", "RECALL_BELOW_FLOOR", "FEATURE_DRIFT_RETRAIN")
    ):
        status = "retrain"
    elif "FEATURE_DRIFT_WARNING" in codes:
        status = "warning"
    else:
        status = "ok"
    if not codes:
        codes = ["WITHIN_THRESHOLDS"]
        reasons = ["all available signals are within thresholds"]
    return {
        "batch": batch["batch"],
        "status": status,
        "label_status": batch["label_status"],
        "reason_codes": codes,
        "reasons": reasons,
        "observed": {
            "pr_auc": batch["pr_auc"],
            "recall": batch["recall"],
            "pct_drifted_features": drift,
        },
    }
```

- [ ] **Step 4: Build the report, Markdown, and GitHub annotation**

```python
def build_decision_report(
    batches: list[dict],
    operating_point: dict,
    params: dict,
) -> dict:
    baseline_pr_auc = operating_point["calibration_metrics"]["pr_auc"]
    decisions = [
        evaluate_batch(baseline_pr_auc, batch, params) for batch in batches
    ]
    overall = max(
        (decision["status"] for decision in decisions),
        key=_SEVERITY.__getitem__,
    )
    trigger = params["trigger"]
    return {
        "baseline": {
            "pr_auc": baseline_pr_auc,
            "operating_threshold": operating_point["threshold"],
            "promoted_model_id": operating_point["promoted_model_id"],
        },
        "thresholds": {
            "pr_auc_drop_pct": float(trigger["pr_auc_drop_pct"]),
            "recall_floor": float(trigger["recall_floor"]),
            "warning_drift_pct": float(trigger["drifted_features_pct"] / 2),
            "retrain_drift_pct": float(trigger["drifted_features_pct"]),
        },
        "decisions": decisions,
        "overall_status": overall,
    }


def github_annotation(report: dict) -> str | None:
    status = report.get("overall_status")
    if status == "ok":
        return None
    if status == "warning":
        return (
            "::notice title=Monitoring warning::Signals entered the warning band; "
            "continue monitoring."
        )
    if status == "retrain":
        return (
            "::warning title=Retraining review recommended::At least one batch "
            "met the review criteria; human approval is required before retraining."
        )
    raise ValueError(f"unknown overall status: {status!r}")


def _display(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_markdown(report: dict) -> str:
    baseline = report["baseline"]
    thresholds = report["thresholds"]
    lines = [
        "# Retraining review decision log",
        "",
        f"Promoted model: **{baseline['promoted_model_id']}**  ",
        f"Calibration PR-AUC baseline: **{baseline['pr_auc']:.4f}**  ",
        f"Operating threshold: **{baseline['operating_threshold']:.2f}**",
        "",
        "A `retrain` result is a recommendation for human review; it does not "
        "start training or replace a model.",
        "",
        "| Batch | Labels | PR-AUC | Recall | Drifted | Status | Reason codes | Reasons |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for decision in report["decisions"]:
        observed = decision["observed"]
        lines.append(
            f"| {decision['batch']} | {decision['label_status']} | "
            f"{_display(observed['pr_auc'])} | {_display(observed['recall'])} | "
            f"{observed['pct_drifted_features']:.1f}% | {decision['status']} | "
            f"{', '.join(decision['reason_codes'])} | "
            f"{'; '.join(decision['reasons'])} |"
        )
    lines.extend(
        [
            "",
            f"Overall status: **{report['overall_status']}**.",
            "",
            "Thresholds: PR-AUC drop > "
            f"{thresholds['pr_auc_drop_pct']:.1f}%; recall < "
            f"{thresholds['recall_floor']:.2f}; warning drift "
            f"{thresholds['warning_drift_pct']:.1f}–"
            f"{thresholds['retrain_drift_pct']:.1f}%; retrain drift > "
            f"{thresholds['retrain_drift_pct']:.1f}%.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_decision_stage() -> dict:
    params = load_params()
    operating_point = json.loads(
        (REPORTS_DIR / "operating_point.json").read_text()
    )
    batches = json.loads((REPORTS_DIR / "drift_summary.json").read_text())
    report = build_decision_report(batches, operating_point, params)
    write_json(REPORTS_DIR / "trigger_decisions.json", report)
    (REPORTS_DIR / "trigger_log.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


def main(argv=None) -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--github-annotation", type=Path)
    args = parser.parse_args(argv)
    if args.github_annotation:
        report = json.loads(args.github_annotation.read_text())
        annotation = github_annotation(report)
        if annotation:
            print(annotation)
        return
    report = run_decision_stage()
    print(f"overall monitoring status: {report['overall_status']}")
```

Add the required `write_json` import and retain the module's
`if __name__ == "__main__": main()` entrypoint. The CLI command is:

```bash
python -m src.retrain_trigger \
  --github-annotation reports/trigger_decisions.json
```

It prints nothing for `ok`, a notice for `warning`, a warning for `retrain`,
and exits non-zero on malformed status.

- [ ] **Step 5: Remove old decision tests and run the focused suite**

Remove `decide` imports/assertions from `tests/test_metrics.py`, then run:

```bash
pytest tests/test_trigger.py tests/test_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/retrain_trigger.py tests/test_trigger.py tests/test_metrics.py
git commit -m "feat: emit structured retraining review decisions"
```

### Task 9: Evidence-Only Self-Contained Dashboard

**Files:**
- Modify: `src/build_dashboard.py`
- Replace: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: JSON and figure/drift assets under an injected `reports_dir`
- Produces: `build(reports_dir: Path = REPORTS_DIR, site_dir: Path = SITE_DIR) -> dict`

- [ ] **Step 1: Replace skipped tests with a complete temporary fixture**

```python
# tests/test_dashboard.py
import json
from pathlib import Path

from src.build_dashboard import build


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _evidence(tmp_path):
    reports = tmp_path / "reports"
    site = tmp_path / "site"
    _write(
        reports / "promotion_record.json",
        {
            "promoted_model_id": "fraud-detector:v3",
            "model_name": "logistic_regression",
            "mlflow_run_id": "run-123",
            "registered_model_version": "3",
            "data_fingerprint": "a" * 64,
        },
    )
    _write(
        reports / "operating_point.json",
        {
            "threshold": 0.42,
            "cost_assumptions": {"false_negative": 100, "false_positive": 1},
            "promoted_model_id": "fraud-detector:v3",
        },
    )
    _write(
        reports / "metrics_operating.json",
        {
            "pr_auc": 0.84,
            "roc_auc": 0.96,
            "recall": 0.81,
            "precision": 0.40,
            "threshold": 0.42,
            "evidence_role": "primary_operating_point",
        },
    )
    _write(
        reports / "metrics_default.json",
        {
            "pr_auc": 0.84,
            "roc_auc": 0.96,
            "recall": 0.55,
            "precision": 0.70,
            "threshold": 0.5,
            "evidence_role": "comparison_default_threshold",
        },
    )
    _write(
        reports / "drift_summary.json",
        [
            {
                "batch": "prod_1",
                "label_status": "pending",
                "n_rows": 100,
                "pr_auc": None,
                "recall": None,
                "precision": None,
                "pct_drifted_features": 10,
                "amount_psi": 0.03,
                "prediction_psi": 0.05,
            }
        ],
    )
    _write(
        reports / "trigger_decisions.json",
        {
            "overall_status": "ok",
            "decisions": [
                {
                    "batch": "prod_1",
                    "status": "ok",
                    "label_status": "pending",
                    "reason_codes": ["LABELS_PENDING"],
                    "reasons": ["labels pending"],
                }
            ],
        },
    )
    _write(
        reports / "run_manifest.json",
        {
            "data": {"source": "REAL — test", "fingerprint_sha256": "a" * 64}
        },
    )
    (reports / "figures").mkdir()
    (reports / "drift").mkdir()
    return reports, site


def test_build_uses_operating_evidence_and_handles_pending_labels(tmp_path):
    reports, site = _evidence(tmp_path)
    out = build(reports, site)
    html = (site / "index.html").read_text()
    assert out["status"] == "ok"
    assert html.index("Operating threshold") < html.index("Default threshold")
    for expected in (
        "fraud-detector:v3",
        "run-123",
        "0.42",
        "LABELS_PENDING",
        "pending",
        "n/a",
    ):
        assert expected in html


def test_build_is_self_contained(tmp_path):
    reports, site = _evidence(tmp_path)
    build(reports, site)
    html = (site / "index.html").read_text()
    for forbidden in ("http://", "https://cdn", "<script"):
        assert forbidden not in html
```

- [ ] **Step 2: Run dashboard tests and confirm the old fixed paths fail**

Run: `pytest tests/test_dashboard.py -q -ra`

Expected: no test is skipped; tests fail because `build` lacks injected paths
and still reads the old evidence.

- [ ] **Step 3: Refactor dashboard inputs**

Change the signature:

```python
def build(
    reports_dir: Path = REPORTS_DIR,
    site_dir: Path = SITE_DIR,
) -> dict:
```

Derive figures and drift directories from `reports_dir`. Load only:

```python
promotion = json.loads((reports_dir / "promotion_record.json").read_text())
operating_point = json.loads((reports_dir / "operating_point.json").read_text())
operating = json.loads((reports_dir / "metrics_operating.json").read_text())
default = json.loads((reports_dir / "metrics_default.json").read_text())
summaries = json.loads((reports_dir / "drift_summary.json").read_text())
trigger = json.loads((reports_dir / "trigger_decisions.json").read_text())
manifest = json.loads((reports_dir / "run_manifest.json").read_text())
```

Do not import `decide`, `_reasons`, or any trigger implementation. Join
`summaries` and `trigger["decisions"]` by `batch`; raise `ValueError` if their
batch sets differ.

- [ ] **Step 4: Render the stable evidence contract**

Show operating metrics before a clearly titled “Default threshold comparison”.
Display promoted model ID, MLflow run/version, data source/fingerprint, operating
threshold, cost rationale, label status, reason codes, and overall status. Use:

```python
def _metric(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"
```

Copy only local assets from `reports/figures` and `reports/drift`. Ensure the
dashboard footer states that a `retrain` status is review-only and requires
human approval.

- [ ] **Step 5: Run dashboard and core regression tests**

Run:

```bash
pytest tests/test_dashboard.py -q -ra
pytest tests/test_ingest.py tests/test_validate.py tests/test_metrics.py \
  tests/test_train.py tests/test_inference.py tests/test_drift.py \
  tests/test_trigger.py -q -ra
```

Expected: all tests pass with no skipped section.

- [ ] **Step 6: Commit**

```bash
git add src/build_dashboard.py tests/test_dashboard.py
git commit -m "feat: render dashboard from evidence contracts"
```

### Task 10: Core Contract Regression Gate

**Files:**
- Inspect: every file changed in Tasks 1–9
- Update: module docstrings that still describe old paths or behavior

**Interfaces:**
- Consumes: complete core plan
- Produces: stable contract handoff for Make, DVC, Docker, CI, and documentation

- [ ] **Step 1: Scan for retired names and unsafe behavior**

Run:

```bash
rg -n "valid\.csv|metrics_valid|models/baseline|schema\.validate|Best-effort registry|grep -qi" \
  src tests params.yaml
```

Expected: no live core-code match. Historical prose is handled later.

- [ ] **Step 2: Run the entire unit suite**

Run: `pytest -q -ra`

Expected: all tests pass and the output contains no `SKIPPED` section.

- [ ] **Step 3: Verify JSON type consistency**

Run:

```bash
python -m compileall -q src tests
python -c "import json; json.dumps({'value': None}, allow_nan=False)"
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 4: Record the finalized interfaces**

In the delivery checklist, record these immutable names before the
reproducibility plan begins:

```text
train.csv
model_valid.csv
calibration.csv
prod_1.csv
prod_2.csv
prod_3.csv
validation_report.json
experiment_comparison.json
promotion_record.json
operating_point.json
metrics_default.json
metrics_operating.json
drift_summary.json
trigger_decisions.json
trigger_log.md
```

- [ ] **Step 5: Commit docstring or checklist corrections**

```bash
git add src tests params.yaml docs/delivery/95-plus-checklist.md
git commit -m "chore: freeze core evidence contracts"
```

## Core Plan Acceptance Gate

Do not proceed to authoritative evidence generation until all statements are
true:

- Model selection reads only `model_valid.csv`.
- Threshold selection and baseline evaluation read only `calibration.csv`.
- Production batches are unseen by training, model selection, and calibration.
- Validation evidence is written before ingestion and failed evidence blocks it.
- Candidate run ID and registered-model version refer to the same MLflow run.
- Registry failure leaves no new local promotion evidence.
- Both threshold views contain complete confusion counts and business cost.
- Unlabeled inference and monitoring succeed with explicit pending fields.
- Structured decisions contain stable reason codes and highest-severity status.
- Healthy status produces no retraining warning.
- Dashboard tests generate their own evidence and never skip.
- Full Pytest output contains zero skips.
