# Reproducibility, CI, and Evidence Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the finalized core pipeline reproducible and independently verifiable through one Make contract, a safe clean-clone DVC graph, an immutable Docker runtime, GitHub Actions, a run manifest, and cross-output consistency tests.

**Architecture:** Make remains the authoritative real-data orchestrator. DVC reproduces deterministic synthetic lineage only inside a disposable clone, Docker executes the same `make verify` contract, and CI runs all three paths before publishing success evidence. Stable JSON contracts are checked against each other rather than trusted independently.

**Tech Stack:** Python 3.13.9, standard-library hashing and package metadata, Make, DVC 3.67.1, Docker, GitHub Actions, Pytest, MLflow, and JSON.

## Global Constraints

- Begin only after the core plan freezes all slice and evidence filenames.
- Make gate order is `data → validate → ingest → train → threshold → evaluate → inference → drift → trigger → manifest`.
- `make verify` must build the dashboard before running the full zero-skip test suite.
- DVC is pinned exactly to 3.67.1.
- DVC must not claim or configure a remote for the 144 MB real CSV.
- DVC synthetic reproduction must never execute in the authoritative working tree.
- The caller's `data/raw/creditcard.csv` must be byte-identical before and after DVC verification.
- MLflow run IDs and registry versions are genuinely non-deterministic governance identifiers; do not fake or canonicalize them.
- The run manifest must report Python 3.13.9, exact direct dependencies, source provenance, data fingerprint, parameter fingerprint, and source commit.
- The Docker base is `python:3.13.9-slim@sha256:326df678c20c78d465db501563f3492d17c42a4afe33a1f2bf5406a1d56b0e86`.
- CI and Docker use Python 3.13.9, not a floating `3.13`.
- Quantitative evidence uploads occur only after successful generation; failed jobs must not upload stale committed metrics as current results.
- Scheduled monitoring parses `reports/trigger_decisions.json`; it never greps Markdown and never dispatches training.
- Scheduled execution is an ephemeral course verification run. Its `retrain` result is review-only and does not persist a replacement production model.
- Preserve the pre-existing untracked briefing DOCX and `data.zip`.

---

## File Structure

**Create**

- `pytest.ini` — registered integration marker.
- `tests/test_reproducibility.py` — static orchestration/runtime assertions.
- `tests/test_evidence_consistency.py` — generated-contract consistency gate.
- `.dvc/config`, `.dvc/.gitignore`, `.dvcignore`, `dvc.lock` — initialized DVC metadata.
- `scripts/verify_dvc_clean_clone.sh` — isolated synthetic reproduction with raw checksum guard.
- `.dockerignore` — excludes raw/private/large local files from the image.

**Modify**

- `src/evidence.py` and `tests/test_evidence.py` — exact pins and run manifest.
- `Makefile` — dependency graph and verification targets.
- `dvc.yaml` — finalized synthetic stage graph.
- `params.yaml` and `src/train.py` — deterministic configured worker count.
- `requirements.txt` — exact DVC and PyArrow pins.
- `Dockerfile` — immutable base and shared verification command.
- `.github/workflows/ci.yml` — unit, DVC, full, consistency, and Docker proof.
- `.github/workflows/monitoring.yml` — JSON status, success-only evidence, dashboard.
- `.gitignore` — private bundle and local large-file protections.

### Task 1: Make as the Single Verification Contract

**Files:**
- Modify: `Makefile`
- Create: `pytest.ini`
- Modify: `tests/test_train.py`
- Create: `tests/test_reproducibility.py`

**Interfaces:**
- Produces: `pipeline`, `dashboard`, `manifest`, `test-fast`, `verify`, and `verify-dvc` targets
- Consumes: the finalized core stage CLIs and evidence paths

- [ ] **Step 1: Write failing orchestration tests**

```python
# tests/test_reproducibility.py
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_make_verify_has_required_gate_order():
    result = subprocess.run(
        ["make", "-n", "verify"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout
    ordered = [
        "src.validate",
        "src.ingest",
        "src.train",
        "src.threshold",
        "src.evaluate",
        "src.batch_inference",
        "src.drift",
        "src.retrain_trigger",
        "src.evidence",
        "src.build_dashboard",
        "pytest",
    ]
    positions = [output.index(token) for token in ordered]
    assert positions == sorted(positions)


def test_no_test_has_a_skip_marker():
    source = "\n".join(
        path.read_text() for path in sorted((ROOT / "tests").glob("test_*.py"))
    )
    skip_marker = "pytest.mark." + "skip"
    skip_call = "pytest." + "skip("
    assert skip_marker not in source
    assert skip_call not in source
```

- [ ] **Step 2: Run the orchestration tests and confirm failure**

Run: `pytest tests/test_reproducibility.py -q`

Expected: `make -n verify` fails because `verify` does not exist.

- [ ] **Step 3: Replace the flat Make pipeline with dependencies**

Use this target structure; preserve the existing safe `fetch-data` and `clean`
recipes:

```make
.PHONY: fetch-data data validate ingest train threshold evaluate inference drift \
	trigger manifest pipeline dashboard test-fast test verify verify-dvc clean

fetch-data:
	python -m src.fetch_data

data:
	@test -f data/raw/creditcard.csv || python -m src.simulate_data

validate: data
	python -m src.validate

ingest: validate
	python -m src.ingest

train: ingest
	python -m src.train

threshold: train
	python -m src.threshold

evaluate: threshold
	python -m src.evaluate

inference: evaluate
	python -m src.batch_inference

drift: inference
	python -m src.drift

trigger: drift
	python -m src.retrain_trigger

manifest: trigger
	python -m src.evidence

pipeline: manifest

dashboard: manifest
	python -m src.build_dashboard

test-fast:
	python -m pytest -q -m "not integration"

test:
	python -m pytest -q -ra

verify: dashboard
	python -m pytest -q -ra

verify-dvc:
	scripts/verify_dvc_clean_clone.sh
```

`verify` has a recipe rather than `verify: dashboard test`; this guarantees the
dashboard is complete before Pytest even when a caller uses `make -j`.

- [ ] **Step 4: Register the integration marker**

```ini
# pytest.ini
[pytest]
markers =
    integration: requires generated pipeline evidence or a filesystem-backed service
```

Do not skip integration tests. `test-fast` deselects them; `verify` runs them.
Insert this line immediately above
`test_candidate_run_is_registered_without_a_second_run` in
`tests/test_train.py`:

```python
@pytest.mark.integration
```

- [ ] **Step 5: Run the dry-run tests**

Run:

```bash
pytest tests/test_reproducibility.py -q
make -n verify
```

Expected: tests pass and the printed command order matches the required chain.

- [ ] **Step 6: Commit**

```bash
git add Makefile pytest.ini tests/test_train.py tests/test_reproducibility.py
git commit -m "feat: make verification dependency ordered"
```

### Task 2: Exact Dependency Inventory and Run Manifest

**Files:**
- Modify: `src/evidence.py`
- Modify: `tests/test_evidence.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: exact requirement lines, provenance, raw CSV, params, source commit
- Produces: `load_exact_requirements(path: Path) -> dict[str, str]`
- Produces: `build_run_manifest(source_commit: str | None = None) -> dict`
- Produces: `write_run_manifest(path=REPORTS_DIR / "run_manifest.json") -> Path`
- CLI: `SOURCE_COMMIT=<40-hex> python -m src.evidence`

- [ ] **Step 1: Add failing exact-pin and manifest tests**

```python
# append to tests/test_evidence.py
import platform

from src.evidence import build_run_manifest, load_exact_requirements


def test_exact_requirement_parser_rejects_ranges(tmp_path):
    exact = tmp_path / "exact.txt"
    exact.write_text("pandas==2.3.3\n# comment\npyyaml==6.0.3\n")
    assert load_exact_requirements(exact) == {
        "pandas": "2.3.3",
        "pyyaml": "6.0.3",
    }
    ranged = tmp_path / "ranged.txt"
    ranged.write_text("dvc>=3.50,<4\n")
    with pytest.raises(ValueError, match="exact == pin"):
        load_exact_requirements(ranged)


def test_manifest_contains_invariant_reproducibility_fields(
    tmp_path, monkeypatch
):
    raw = tmp_path / "creditcard.csv"
    raw.write_text("Time,V1,Amount,Class\n0,0,1,0\n")
    params = tmp_path / "params.yaml"
    params.write_text("data: {}\n")
    provenance = {
        "source": "REAL — unit test",
        "rows": 1,
        "fraud": 0,
        "fraud_rate": 0.0,
    }
    monkeypatch.setattr("src.evidence.RAW_PATH", raw)
    monkeypatch.setattr("src.evidence.PARAMS_PATH", params)
    monkeypatch.setattr("src.evidence.read_provenance", lambda _: provenance)
    monkeypatch.setattr(
        "src.evidence.installed_direct_dependencies",
        lambda _: {"pandas": "2.3.3"},
    )
    manifest = build_run_manifest(source_commit="a" * 40)
    assert manifest["schema_version"] == 1
    assert manifest["source_commit"] == "a" * 40
    assert manifest["python_version"] == platform.python_version()
    assert manifest["data"]["fingerprint_sha256"] == sha256_file(raw)
    assert manifest["parameters"]["fingerprint_sha256"] == sha256_file(params)
    assert "generated_at" not in manifest
```

- [ ] **Step 2: Run evidence tests and confirm missing interfaces**

Run: `pytest tests/test_evidence.py -q`

Expected: imports fail for the manifest helpers.

- [ ] **Step 3: Make every direct requirement exact**

Retain all existing exact pins, replace the DVC range, and add the direct
Parquet runtime used by `src.fetch_data`:

```text
dvc==3.67.1
pyarrow==25.0.0
```

No non-comment requirement line may use `>=`, `<=`, `~=`, `>`, `<`, or an
unpinned name.

Install the finalized pins before the manifest checks:

```bash
python -m pip install -r requirements.txt
```

- [ ] **Step 4: Implement exact-pin parsing and installed-version checks**

Add to `src/evidence.py`:

```python
import os
import platform
import re
import subprocess
from importlib import metadata

from src.config import REPORTS_DIR, ROOT, load_params, read_provenance

PARAMS_PATH = ROOT / "params.yaml"
RAW_PATH = ROOT / load_params()["data"]["raw_path"]
_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)$")


def load_exact_requirements(path: Path) -> dict[str, str]:
    result = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN.fullmatch(line)
        if not match:
            raise ValueError(f"requirement is not an exact == pin: {line}")
        result[match.group(1)] = match.group(2)
    return result


def installed_direct_dependencies(path: Path) -> dict[str, str]:
    pins = load_exact_requirements(path)
    installed = {name: metadata.version(name) for name in pins}
    mismatches = {
        name: {"required": pins[name], "installed": installed[name]}
        for name in pins
        if installed[name] != pins[name]
    }
    if mismatches:
        raise RuntimeError(f"direct dependency mismatch: {mismatches}")
    return installed
```

- [ ] **Step 5: Implement the invariant run manifest**

```python
def _source_commit(explicit: str | None) -> str:
    commit = explicit or os.environ.get("SOURCE_COMMIT")
    if not commit:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source commit must be a full 40-character SHA")
    return commit


def build_run_manifest(source_commit: str | None = None) -> dict:
    params = load_params()
    provenance = read_provenance(params)
    return {
        "schema_version": 1,
        "source_commit": _source_commit(source_commit),
        "python_version": platform.python_version(),
        "dependencies": installed_direct_dependencies(ROOT / "requirements.txt"),
        "data": {
            "source": provenance["source"],
            "rows": int(provenance["rows"]),
            "fraud": int(provenance["fraud"]),
            "fraud_rate": float(provenance["fraud_rate"]),
            "fingerprint_sha256": sha256_file(RAW_PATH),
        },
        "parameters": {
            "path": "params.yaml",
            "fingerprint_sha256": sha256_file(PARAMS_PATH),
        },
    }


def write_run_manifest(
    path: Path = REPORTS_DIR / "run_manifest.json",
) -> Path:
    return write_json(path, build_run_manifest())


if __name__ == "__main__":
    output = write_run_manifest()
    print(f"wrote {output}")
```

Do not add a timestamp or MLflow run ID; those would make invariant
reproducibility evidence volatile.

- [ ] **Step 6: Run tests and the CLI**

Run:

```bash
pytest tests/test_evidence.py tests/test_reproducibility.py -q
SOURCE_COMMIT="$(git rev-parse HEAD)" python -m src.evidence
python -c "import json; p=json.load(open('reports/run_manifest.json')); print(p['python_version'])"
```

Expected: tests pass and the final command prints `3.13.9`.

- [ ] **Step 7: Commit**

```bash
git add src/evidence.py tests/test_evidence.py requirements.txt
git commit -m "feat: add exact run manifest"
```

### Task 3: Honest DVC Graph in a Disposable Clone

**Files:**
- Create: `.dvc/config`
- Create: `.dvc/.gitignore`
- Create: `.dvcignore`
- Replace: `dvc.yaml`
- Create: `scripts/verify_dvc_clean_clone.sh`
- Modify: `params.yaml`
- Modify: `src/train.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: committed `HEAD`; no positional arguments
- Produces: `scripts/verify_dvc_clean_clone.sh`
- Optional: `DVC_REPRO_EXPORT_DIR=/absolute/path` exports only runtime lock and synthetic verification evidence

- [ ] **Step 1: Configure deterministic model workers**

Add:

```yaml
train:
  models: [logistic_regression, xgboost]
  class_weight: balanced
  random_state: 42
  n_jobs: 1
```

Use `params["train"]["n_jobs"]` for XGBoost and the RandomForest fallback.
Log it with every candidate.

- [ ] **Step 2: Initialize DVC metadata without a remote**

Run:

```bash
dvc init
dvc remote list
```

Expected: `.dvc/config` and `.dvc/.gitignore` exist; `dvc remote list` prints
nothing. Do not run `dvc add data/raw/creditcard.csv`.

Create `.dvcignore`:

```text
submission-private/
dist/
site/
mlflow.db
mlruns/
reports/drift/*.html
```

- [ ] **Step 3: Replace `dvc.yaml` with the finalized graph**

Use the following stage names and contracts:

```yaml
stages:
  data:
    cmd: python -m src.simulate_data
    deps:
      - src/simulate_data.py
      - src/config.py
    params:
      - simulate
      - data.raw_path
    outs:
      - data/raw/creditcard.csv
      - data/raw/PROVENANCE.json

  validate:
    cmd: python -m src.validate
    deps:
      - src/validate.py
      - src/ingest.py
      - src/evidence.py
      - data/raw/creditcard.csv
    params:
      - data
    outs:
      - reports/validation_report.json:
          cache: false

  ingest:
    cmd: python -m src.ingest
    deps:
      - src/ingest.py
      - src/validate.py
      - reports/validation_report.json
      - data/raw/creditcard.csv
    params:
      - data
    outs:
      - data/batches/train.csv
      - data/batches/model_valid.csv
      - data/batches/calibration.csv
      - data/batches/prod_1.csv
      - data/batches/prod_2.csv
      - data/batches/prod_3.csv

  train:
    cmd: python -m src.train
    deps:
      - src/train.py
      - src/features.py
      - src/evaluate.py
      - src/evidence.py
      - data/batches/train.csv
      - data/batches/model_valid.csv
      - data/raw/PROVENANCE.json
    params:
      - train
      - threshold
    outs:
      - models/model.joblib
      - models/scaler.joblib
      - reports/experiment_comparison.json:
          cache: false
      - reports/promotion_record.json:
          cache: false

  threshold:
    cmd: python -m src.threshold
    deps:
      - src/threshold.py
      - src/artifacts.py
      - models/model.joblib
      - models/scaler.joblib
      - reports/promotion_record.json
      - data/batches/calibration.csv
    params:
      - threshold
    outs:
      - models/threshold.json
      - reports/operating_point.json:
          cache: false
    plots:
      - reports/figures/threshold_tradeoff.png:
          cache: false

  evaluate:
    cmd: python -m src.evaluate
    deps:
      - src/evaluate.py
      - models/model.joblib
      - models/scaler.joblib
      - models/threshold.json
      - data/batches/calibration.csv
    metrics:
      - reports/metrics_default.json:
          cache: false
      - reports/metrics_operating.json:
          cache: false
    plots:
      - reports/figures/confusion_matrix_default.png:
          cache: false
      - reports/figures/confusion_matrix_operating.png:
          cache: false
      - reports/figures/pr_curve.png:
          cache: false

  inference:
    cmd: python -m src.batch_inference
    deps:
      - src/batch_inference.py
      - src/validate.py
      - models/model.joblib
      - models/scaler.joblib
      - models/threshold.json
      - reports/promotion_record.json
      - data/batches/prod_1.csv
      - data/batches/prod_2.csv
      - data/batches/prod_3.csv
    outs:
      - data/batches/preds_prod_1.csv
      - data/batches/preds_prod_2.csv
      - data/batches/preds_prod_3.csv

  drift:
    cmd: python -m src.drift
    deps:
      - src/drift.py
      - data/batches/train.csv
      - data/batches/calibration.csv
      - data/batches/prod_1.csv
      - data/batches/prod_2.csv
      - data/batches/prod_3.csv
      - data/batches/preds_prod_1.csv
      - data/batches/preds_prod_2.csv
      - data/batches/preds_prod_3.csv
    params:
      - drift
      - threshold
    metrics:
      - reports/drift_summary.json:
          cache: false
    plots:
      - reports/figures/psi_prod_1.png:
          cache: false
      - reports/figures/psi_prod_2.png:
          cache: false
      - reports/figures/psi_prod_3.png:
          cache: false

  trigger:
    cmd: python -m src.retrain_trigger
    deps:
      - src/retrain_trigger.py
      - reports/drift_summary.json
      - reports/operating_point.json
    params:
      - trigger
    outs:
      - reports/trigger_decisions.json:
          cache: false
      - reports/trigger_log.md:
          cache: false
```

Do not add `run_manifest.json` to DVC outputs: its source commit would create a
self-reference when the resulting lockfile is committed.

- [ ] **Step 4: Write the raw-safe disposable-clone verifier**

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
raw_path="$repo_root/data/raw/creditcard.csv"
scratch="$(mktemp -d)"
cleanup() {
  rm -rf "$scratch"
}
trap cleanup EXIT

checksum() {
  python - "$1" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.exists():
    print("missing")
else:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    print(digest.hexdigest())
PY
}

before="$(checksum "$raw_path")"
source_commit="$(git -C "$repo_root" rev-parse HEAD)"
mkdir -p "$scratch/repo"
git -C "$repo_root" archive HEAD | tar -x -C "$scratch/repo"

(
  cd "$scratch/repo"
  git init --quiet
  git config user.name "DVC verification"
  git config user.email "dvc-verification@example.invalid"
  git add --all
  git commit --quiet -m "isolated verification snapshot"
  export SOURCE_COMMIT="$source_commit"
  python -m dvc repro
  python -m dvc status
  python -m src.evidence
  python -m src.build_dashboard
  if [[ -f tests/test_evidence_consistency.py ]]; then
    python -m pytest -q tests/test_evidence_consistency.py
  else
    python -m pytest -q tests/test_evidence.py
  fi
)

after="$(checksum "$raw_path")"
if [[ "$before" != "$after" ]]; then
  echo "caller raw data changed during DVC verification" >&2
  exit 1
fi

if [[ -n "${DVC_REPRO_EXPORT_DIR:-}" ]]; then
  mkdir -p "$DVC_REPRO_EXPORT_DIR"
  cp "$scratch/repo/dvc.lock" "$DVC_REPRO_EXPORT_DIR/dvc.lock"
  cp -R "$scratch/repo/reports" "$DVC_REPRO_EXPORT_DIR/reports"
fi

echo "DVC clean-clone verification passed; caller raw checksum: $after"
```

Save as `scripts/verify_dvc_clean_clone.sh`, then run
`chmod +x scripts/verify_dvc_clean_clone.sh`.

- [ ] **Step 5: Protect large and private local files**

Append to `.gitignore`:

```text
/submission-private/
/dist/
/data.zip
/MLOpsMaster_Final_Project_Briefing_Credit_Card_Fraud.docx
```

These ignores protect the existing files; do not delete, move, or stage them.

- [ ] **Step 6: Commit the graph before generating the lock**

```bash
git add .dvc .dvcignore dvc.yaml scripts/verify_dvc_clean_clone.sh \
  params.yaml src/train.py .gitignore
git commit -m "feat: initialize safe synthetic DVC graph"
```

- [ ] **Step 7: Generate and commit only the runtime lock**

Run:

```bash
export_dir="$(mktemp -d)"
DVC_REPRO_EXPORT_DIR="$export_dir" scripts/verify_dvc_clean_clone.sh
cp "$export_dir/dvc.lock" dvc.lock
rm -rf "$export_dir"
git add dvc.lock
git commit -m "chore: lock synthetic DVC pipeline"
scripts/verify_dvc_clean_clone.sh
dvc dag
dvc remote list
```

Expected: both verifier runs exit 0; `dvc dag` shows the complete chain;
`dvc remote list` is empty. Do not copy synthetic reports from the export
directory into the authoritative working tree.

### Task 4: Immutable Python and Docker Runtime

**Files:**
- Modify: `Dockerfile`
- Create: `.dockerignore`
- Modify: `tests/test_reproducibility.py`

**Interfaces:**
- Consumes: build argument `SOURCE_COMMIT=<40-character SHA>`
- Produces: an image whose default command is `make verify`

- [ ] **Step 1: Add failing static runtime tests**

```python
def test_runtime_versions_and_docker_contract_are_exact():
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    direct = [
        line.strip()
        for line in requirements
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert all("==" in line for line in direct)
    assert "dvc==3.67.1" in direct
    assert "pyarrow==25.0.0" in direct
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert (
        "FROM python:3.13.9-slim@sha256:"
        "326df678c20c78d465db501563f3492d17c42a4afe33a1f2bf5406a1d56b0e86"
    ) in dockerfile
    assert 'CMD ["make", "verify"]' in dockerfile
    assert "ARG SOURCE_COMMIT" in dockerfile
```

- [ ] **Step 2: Run the static test and confirm the floating image fails**

Run: `pytest tests/test_reproducibility.py -q`

Expected: the Docker contract assertion fails.

- [ ] **Step 3: Pin the image and shared verification command**

```dockerfile
FROM python:3.13.9-slim@sha256:326df678c20c78d465db501563f3492d17c42a4afe33a1f2bf5406a1d56b0e86

RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ARG SOURCE_COMMIT
ENV SOURCE_COMMIT=${SOURCE_COMMIT} \
    MPLBACKEND=Agg \
    DISABLE_PANDERA_IMPORT_WARNING=True \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["make", "verify"]
```

The build command must supply `SOURCE_COMMIT`; an omitted value deliberately
causes the manifest gate to fail.

- [ ] **Step 4: Exclude local data and private artifacts from the build**

```text
# .dockerignore
.git
.venv
__pycache__
*.pyc
mlflow.db
mlruns
mlartifacts
models
site
data/raw/*.csv
data/raw/PROVENANCE.json
data/batches/*.csv
submission-private
dist
data.zip
MLOpsMaster_Final_Project_Briefing_Credit_Card_Fraud.docx
```

- [ ] **Step 5: Build and run the exact runtime**

Run:

```bash
source_commit="$(git rev-parse HEAD)"
docker build --pull --build-arg SOURCE_COMMIT="$source_commit" \
  --tag fraud-mlops:verify .
docker run --rm --entrypoint python fraud-mlops:verify --version
docker run --rm fraud-mlops:verify
```

Expected: version output is exactly `Python 3.13.9`; the default run exits 0
after the full verification suite with zero skips.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore tests/test_reproducibility.py
git commit -m "build: pin immutable verification runtime"
```

### Task 5: Cross-Output Evidence Consistency

**Files:**
- Create: `tests/test_evidence_consistency.py`
- Modify: `src/build_dashboard.py` only if it omits a stable evidence marker

**Interfaces:**
- Consumes: generated reports and `site/index.html`
- Produces: an integration test that fails on any cross-output contradiction

- [ ] **Step 1: Write the consistency test**

```python
# tests/test_evidence_consistency.py
import json
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

    promoted = [
        item for item in comparison["candidates"] if item["promoted"]
    ]
    assert len(promoted) == 1
    assert promoted[0]["run_id"] == promotion["mlflow_run_id"]
    assert promoted[0]["model_name"] == promotion["model_name"]
    assert comparison["promoted_run_id"] == promotion["mlflow_run_id"]

    assert operating_point["promoted_model_id"] == promotion["promoted_model_id"]
    assert operating["promoted_model_id"] == promotion["promoted_model_id"]
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
        assert summary["label_status"] == decision["label_status"]
        assert decision["reason_codes"]

    assert manifest["python_version"] == "3.13.9"
    assert manifest["data"]["fingerprint_sha256"] == promotion["data_fingerprint"]
    assert manifest["parameters"]["fingerprint_sha256"] == promotion[
        "parameter_fingerprint"
    ]
    assert operating_point["calibration_data_fingerprint"] == operating[
        "calibration_data_fingerprint"
    ]

    html = (ROOT / "site" / "index.html").read_text()
    for marker in (
        promotion["promoted_model_id"],
        promotion["mlflow_run_id"],
        str(promotion["registered_model_version"]),
        f"{operating_point['threshold']:.2f}",
        trigger["overall_status"],
    ):
        assert marker in html
```

- [ ] **Step 2: Run it before evidence exists and confirm the integration gate**

Run: `pytest tests/test_evidence_consistency.py -q`

Expected: failure naming the first missing or stale core artefact. This is
intentional until `make dashboard` regenerates all contracts.

- [ ] **Step 3: Add a stable dashboard marker if numeric formatting is ambiguous**

Ensure `site/index.html` contains one hidden-free, human-readable evidence line:

```html
<p class="evidence-id">Model fraud-detector:v3 · MLflow run run-id · registry version 3 · operating threshold 0.42 · status ok</p>
```

Generate it from JSON fields; do not hard-code the example values.

- [ ] **Step 4: Generate synthetic evidence in a disposable clone**

Commit the test first so the clean clone contains it:

```bash
git add tests/test_evidence_consistency.py src/build_dashboard.py
git commit -m "test: enforce cross-output evidence consistency"
scripts/verify_dvc_clean_clone.sh
```

Expected: the verifier generates its own evidence and the consistency test
passes inside the clone without touching the caller's real raw file.

### Task 6: CI and Scheduled Monitoring Consume the Same Contracts

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/monitoring.yml`

**Interfaces:**
- Consumes: `make test-fast`, `make verify`, `make verify-dvc`, trigger annotation CLI
- Produces: success-only evidence uploads and review-only monitoring annotations

- [ ] **Step 1: Replace CI with the exact verification sequence**

The primary job must run:

```yaml
name: CI
on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    env:
      MPLBACKEND: Agg
      DISABLE_PANDERA_IMPORT_WARNING: "True"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13.9"
          cache: pip
      - run: pip install -r requirements.txt
      - name: Fast unit tests
        run: make test-fast
      - name: DVC clean-clone reproduction
        run: |
          export DVC_REPRO_EXPORT_DIR="$RUNNER_TEMP/dvc-repro"
          make verify-dvc
      - name: Complete verification
        run: SOURCE_COMMIT="$GITHUB_SHA" make verify
      - name: Evidence consistency
        run: pytest -q tests/test_evidence_consistency.py
      - name: Upload successful evidence
        uses: actions/upload-artifact@v4
        with:
          name: pipeline-evidence
          path: |
            reports/*.json
            reports/trigger_log.md
            reports/figures/**
            reports/drift/**
            site/**
            dvc.yaml
            dvc.lock
            params.yaml
            requirements.txt
            ${{ runner.temp }}/dvc-repro/**
```

Do not set `if: always()` on quantitative evidence upload.

- [ ] **Step 2: Add a parallel Docker proof job**

```yaml
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build immutable verification image
        run: |
          docker build --build-arg SOURCE_COMMIT="$GITHUB_SHA" \
            --tag fraud-mlops:verify .
      - name: Assert exact Python runtime
        run: |
          test "$(docker run --rm --entrypoint python fraud-mlops:verify --version)" \
            = "Python 3.13.9"
      - name: Run shared verification contract
        run: docker run --rm fraud-mlops:verify
```

- [ ] **Step 3: Replace scheduled Markdown grep with structured parsing**

Use Python 3.13.9, pinned dependencies, best-effort real fetch, then:

```yaml
      - name: Run verified monitoring simulation
        run: SOURCE_COMMIT="$GITHUB_SHA" make verify

      - name: Parse structured review decision
        id: decision
        run: |
          python -m src.retrain_trigger \
            --github-annotation reports/trigger_decisions.json
          status="$(python -c 'import json; print(json.load(open("reports/trigger_decisions.json"))["overall_status"])')"
          case "$status" in
            ok|warning|retrain) ;;
            *) echo "Unknown monitoring status: $status" >&2; exit 1 ;;
          esac
          echo "overall_status=$status" >> "$GITHUB_OUTPUT"

      - name: Publish decision to run summary
        run: |
          {
            echo "## Monitoring run"
            echo
            echo "Overall status: **${{ steps.decision.outputs.overall_status }}**"
            echo
            cat reports/trigger_log.md
          } >> "$GITHUB_STEP_SUMMARY"
```

Delete the `grep -qi 'retrain'` step completely. Keep Pages deployment and
success-only artifact upload. The workflow must not call or dispatch a training
workflow based on status.

- [ ] **Step 4: Check workflow syntax and stale paths**

Run:

```bash
rg -n "metrics_valid|confusion_matrix\\.png|grep -qi|if: always|python-version: \"3\\.13\"" \
  .github/workflows
python -c "import yaml; [yaml.safe_load(open(p)) for p in ['.github/workflows/ci.yml', '.github/workflows/monitoring.yml']]"
```

Expected: the search returns no match and both workflows parse.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml .github/workflows/monitoring.yml
git commit -m "ci: verify reproducibility and parse structured status"
```

### Task 7: Full Local Reproducibility Gate

**Files:**
- Inspect: all files changed in this plan
- Update: `docs/delivery/95-plus-checklist.md`

**Interfaces:**
- Consumes: the complete core and reproducibility plans
- Produces: local evidence for Gates G1 and G2

- [ ] **Step 1: Verify the current working-tree path without replacing raw data**

If the authoritative real CSV is already present, do not run the simulator.
Run:

```bash
source_commit="$(git rev-parse HEAD)"
SOURCE_COMMIT="$source_commit" make verify
```

Expected: the full pipeline, dashboard, and zero-skip suite pass.

- [ ] **Step 2: Verify the isolated DVC path**

Run:

```bash
before="$(python -c 'from src.evidence import sha256_file; from pathlib import Path; p=Path("data/raw/creditcard.csv"); print(sha256_file(p) if p.exists() else "missing")')"
scripts/verify_dvc_clean_clone.sh
after="$(python -c 'from src.evidence import sha256_file; from pathlib import Path; p=Path("data/raw/creditcard.csv"); print(sha256_file(p) if p.exists() else "missing")')"
test "$before" = "$after"
```

Expected: DVC succeeds and the caller checksum is unchanged.

- [ ] **Step 3: Verify Docker**

Run:

```bash
source_commit="$(git rev-parse HEAD)"
docker build --pull --build-arg SOURCE_COMMIT="$source_commit" \
  --tag fraud-mlops:verify .
test "$(docker run --rm --entrypoint python fraud-mlops:verify --version)" \
  = "Python 3.13.9"
docker run --rm fraud-mlops:verify
```

Expected: all commands exit 0.

- [ ] **Step 4: Run final consistency and repository checks**

Run:

```bash
pytest -q tests/test_evidence_consistency.py
pytest -q -ra
dvc remote list
git diff --check
git status --short
```

Expected: zero skips, empty DVC remote list, no whitespace errors, and no
unexpected tracked/private/raw files.

- [ ] **Step 5: Record evidence**

Add the exact command results, test count, Docker image ID, DVC lock SHA-256,
source commit, and reviewer to `docs/delivery/95-plus-checklist.md`.

- [ ] **Step 6: Commit the gate record**

```bash
git add docs/delivery/95-plus-checklist.md
git commit -m "docs: record reproducibility gate"
```

## Reproducibility Plan Acceptance Gate

- `make -n verify` proves validation precedes ingestion.
- `make verify` generates every evidence contract, dashboard, and zero-skip test result.
- `scripts/verify_dvc_clean_clone.sh` leaves caller raw data byte-identical.
- `.dvc/`, `dvc.yaml`, and `dvc.lock` are committed; `dvc remote list` is empty.
- Requirements contain exact `==` pins, including DVC 3.67.1 and PyArrow 25.0.0.
- Docker reports Python 3.13.9 and executes `make verify`.
- CI runs fast tests, clean-clone DVC, full verification, consistency, and Docker.
- Scheduled monitoring parses structured JSON and issues no retraining warning for `ok`.
- Evidence upload cannot present stale metrics from a failed run.
- Manifest fingerprints agree with promotion evidence and reports the exact source commit.
