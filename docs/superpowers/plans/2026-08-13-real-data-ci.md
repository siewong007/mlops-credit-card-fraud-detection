# Real-data-only CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every GitHub Actions pipeline and evidence run use the real ULB dataset and fail when it is unavailable, while retaining synthetic data for local development and unit tests.

**Architecture:** GitHub Actions will explicitly run the existing OpenML fetch before pipeline verification. The DVC clean-clone script will gain an explicit `DVC_REPRO_DATA_MODE=real` path that validates and copies the fetched real inputs into its disposable clone; its default local path remains synthetic. Static workflow tests plus a small behavioral provenance-guard test lock the contract in place.

**Tech Stack:** GitHub Actions YAML, GNU Make, Bash, Python 3.13, Pytest, DVC.

## Global Constraints

- Every GitHub Actions pipeline or evidence run must use OpenML dataset 1597 and fail if it cannot be fetched or verified.
- Synthetic data remains available for local development and isolated unit tests.
- GitHub Actions must never catch a real-data failure and invoke `src.simulate_data`.
- No new data mirror, cache service, secret, external dependency, model change, or schema change.
- Preserve the resumable OpenML download behavior.
- Do not modify or stage the user's untracked `docs/TECHNICAL_REPORT.docx`.

## File map

- `scripts/require_real_data.py`: validate that a raw CSV exists and its provenance identifies a real dataset.
- `scripts/verify_dvc_clean_clone.sh`: select real or synthetic seed data for the disposable DVC clone.
- `.github/workflows/ci.yml`: fetch real data before DVC/pipeline verification and run Docker verification on real data.
- `.github/workflows/monitoring.yml`: make the real-data fetch mandatory and remove the fallback summary.
- `tests/test_reproducibility.py`: behavioral and workflow-contract regression tests.
- `README.md`, `Makefile`, `params.yaml`, `src/simulate_data.py`, `notebooks/README.md`, `docs/MODEL_CARD.md`, `docs/DEMO_SCRIPT.md`, `docs/TECHNICAL_REPORT.md`: describe synthetic data as local/offline test support rather than CI input.

---

### Task 1: Add a testable real-data guard for DVC reproduction

**Files:**
- Create: `scripts/require_real_data.py`
- Modify: `scripts/verify_dvc_clean_clone.sh:4-60`
- Modify: `tests/test_reproducibility.py:151-163`

**Interfaces:**
- Consumes: raw CSV path and `PROVENANCE.json` path as two command-line arguments.
- Produces: exit code 0 only when the CSV exists, provenance is valid JSON, and `source` starts with `REAL`; `DVC_REPRO_DATA_MODE=real` selects this guarded path.

- [ ] **Step 1: Write failing behavioral tests for the provenance guard**

Add a subprocess helper and focused test to `tests/test_reproducibility.py`:

```python
import json
import sys


def _require_real_data(raw_path, provenance_path):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "require_real_data.py"),
            str(raw_path),
            str(provenance_path),
        ],
        capture_output=True,
        text=True,
    )


def test_real_data_guard_accepts_only_real_data_with_valid_provenance(tmp_path):
    raw = tmp_path / "creditcard.csv"
    provenance = tmp_path / "PROVENANCE.json"

    missing = _require_real_data(raw, provenance)
    assert missing.returncode != 0
    assert "real dataset is missing" in missing.stderr

    raw.write_text("Class\n0\n")
    provenance.write_text("not json")
    invalid = _require_real_data(raw, provenance)
    assert invalid.returncode != 0
    assert "valid provenance" in invalid.stderr

    provenance.write_text(json.dumps({"source": "SYNTHETIC — local test"}))
    synthetic = _require_real_data(raw, provenance)
    assert synthetic.returncode != 0
    assert "not real" in synthetic.stderr

    provenance.write_text(json.dumps({"source": "REAL — ULB creditcard"}))
    assert _require_real_data(raw, provenance).returncode == 0
```

- [ ] **Step 2: Run the guard test and verify RED**

Run:

```bash
python -m pytest -q tests/test_reproducibility.py::test_real_data_guard_accepts_only_real_data_with_valid_provenance
```

Expected: FAIL because `scripts/require_real_data.py` does not exist.

- [ ] **Step 3: Implement the minimal guard**

Create `scripts/require_real_data.py`:

```python
import json
import sys
from pathlib import Path


def main() -> None:
    raw = Path(sys.argv[1])
    provenance = Path(sys.argv[2])
    if not raw.is_file():
        raise SystemExit(f"real dataset is missing: {raw}")
    try:
        source = json.loads(provenance.read_text())["source"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(f"real dataset requires valid provenance: {provenance}") from exc
    if not isinstance(source, str) or not source.startswith("REAL"):
        raise SystemExit(f"dataset provenance is not real: {source!r}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the guard test and verify GREEN**

Run the command from Step 2.

Expected: `1 passed`.

- [ ] **Step 5: Write a failing DVC mode contract test**

Replace the current clean-clone assertion with a test that preserves the local
synthetic path and requires the real path:

```python
def test_clean_clone_verifier_supports_guarded_real_data_mode():
    script = (ROOT / "scripts" / "verify_dvc_clean_clone.sh").read_text()
    commands = [line.strip() for line in script.splitlines()]
    ordered = [
        "python -m dvc repro",
        "python -m dvc status --quiet",
    ]
    positions = [commands.index(command) for command in ordered]
    assert positions == sorted(positions)
    assert "set -euo pipefail" in commands
    assert 'DVC_REPRO_DATA_MODE:-synthetic' in script
    assert '"$repo_root/scripts/require_real_data.py"' in script
    assert 'cp "$raw_path" "$scratch/repo/data/raw/creditcard.csv"' in script
    assert 'cp "$provenance_path" "$scratch/repo/data/raw/PROVENANCE.json"' in script
    assert "python -m src.simulate_data" in commands
```

- [ ] **Step 6: Run the DVC mode test and verify RED**

Run:

```bash
python -m pytest -q tests/test_reproducibility.py::test_clean_clone_verifier_supports_guarded_real_data_mode
```

Expected: FAIL because the mode selector and guarded copy path are absent.

- [ ] **Step 7: Implement real and synthetic DVC seed modes**

In `scripts/verify_dvc_clean_clone.sh`, define `provenance_path` and
`data_mode="${DVC_REPRO_DATA_MODE:-synthetic}"`. Reject values other than
`real` and `synthetic`. For `real`, call the guard before creating the clone,
then copy the raw CSV and provenance into `scratch/repo/data/raw/`. Inside the
clone, run `python -m src.simulate_data` only when the mode is `synthetic`.

- [ ] **Step 8: Verify Task 1**

Run:

```bash
bash -n scripts/verify_dvc_clean_clone.sh
python -m pytest -q tests/test_reproducibility.py::test_real_data_guard_accepts_only_real_data_with_valid_provenance tests/test_reproducibility.py::test_clean_clone_verifier_supports_guarded_real_data_mode
```

Expected: shell syntax exits 0 and `2 passed`.

- [ ] **Step 9: Commit Task 1**

```bash
git add scripts/require_real_data.py scripts/verify_dvc_clean_clone.sh tests/test_reproducibility.py
git commit -m "Require real inputs for CI DVC reproduction"
```

---

### Task 2: Make real data mandatory in GitHub Actions

**Files:**
- Modify: `.github/workflows/ci.yml:18-27,66-67`
- Modify: `.github/workflows/monitoring.yml:35-44,58-72`
- Modify: `tests/test_reproducibility.py:174-279`

**Interfaces:**
- Consumes: `make fetch-data`, `DVC_REPRO_DATA_MODE=real`, and the existing `make verify` contract.
- Produces: workflow jobs that stop on OpenML failure and publish evidence only from real data.

- [ ] **Step 1: Change workflow expectations first**

Update the existing CI workflow test to require a `Fetch real dataset (OpenML)`
step after fast tests and before DVC reproduction, with `run: make fetch-data`
and no `continue-on-error`. Require the DVC step to run:

```text
export DVC_REPRO_EXPORT_DIR="$GITHUB_WORKSPACE/dvc-repro"
DVC_REPRO_DATA_MODE=real make verify-dvc
```

Require the Docker step to equal:

```text
docker run --rm fraud-mlops:verify make fetch-data verify
```

Update the monitoring test to assert that `continue-on-error` is absent from
the fetch step and that the summary contains `Dataset: **real ULB (OpenML
1597)**` but contains neither `synthetic fallback` nor `steps.fetch.outcome`.

- [ ] **Step 2: Run workflow tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_reproducibility.py::test_ci_workflow_runs_shared_verification_contract_and_docker_proof tests/test_reproducibility.py::test_monitoring_workflow_parses_structured_review_status_before_pages_deploy
```

Expected: both tests FAIL against the current fallback-enabled workflows.

- [ ] **Step 3: Apply the minimal workflow changes**

In `.github/workflows/ci.yml`:

- add mandatory `make fetch-data` after fast tests;
- set `DVC_REPRO_DATA_MODE=real` on the clean-clone command;
- keep complete verification after the real fetch;
- run Docker as `docker run --rm fraud-mlops:verify make fetch-data verify`.

In `.github/workflows/monitoring.yml`:

- remove the fallback comment, fetch `id`, and `continue-on-error`;
- keep `make verify` after the mandatory fetch;
- replace the summary conditional with one unconditional real-data line.

- [ ] **Step 4: Run workflow tests and verify GREEN**

Run the command from Step 2.

Expected: `2 passed`.

- [ ] **Step 5: Run the complete reproducibility test file**

Run:

```bash
python -m pytest -q tests/test_reproducibility.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add .github/workflows/ci.yml .github/workflows/monitoring.yml tests/test_reproducibility.py
git commit -m "Run CI pipelines only on real data"
```

---

### Task 3: Align data-provenance documentation

**Files:**
- Modify: `README.md`
- Modify: `Makefile:9-11`
- Modify: `params.yaml:16-17`
- Modify: `src/simulate_data.py:1-20`
- Modify: `notebooks/README.md:49-53`
- Modify: `docs/MODEL_CARD.md:58-59`
- Modify: `docs/DEMO_SCRIPT.md:136-138`
- Modify: `docs/TECHNICAL_REPORT.md:10-12,92-97,573-574,588-606`

**Interfaces:**
- Consumes: the real-data-only workflow behavior from Task 2.
- Produces: consistent instructions stating that synthetic data is local/offline test support and GitHub Actions fails when real data is unavailable.

- [ ] **Step 1: Update source labels and local comments**

Apply these exact terminology changes:

- use `LOCAL DEVELOPMENT / TESTING ONLY` in the synthetic module heading;
- use `local dev/test only` in `_SYNTHETIC_SOURCE` and `params.yaml`;
- describe `make data` in the Makefile as the local/offline fallback;
- replace the module claim that real data is unavailable in CI with the accurate
  claim that real data is not committed and the generator supports offline local
  runs and isolated tests.

- [ ] **Step 2: Update user-facing documentation**

Make these claims consistent across the listed Markdown files:

- `make verify` may still generate synthetic data locally when no raw CSV exists;
- GitHub Actions first fetches the real ULB dataset and fails on fetch failure;
- scheduled monitoring has no synthetic fallback;
- CI's DVC real-data mode reproduces lineage using the fetched ULB inputs;
- synthetic data remains useful for offline development and isolated tests.

- [ ] **Step 3: Verify stale CI-fallback wording is gone**

Run:

```bash
rg -n "DEVELOPMENT / CI ONLY|dev/CI|offline/CI|CI fallback|synthetic fallback|falling back to the synthetic|retained for CI|CI stays fast, deterministic and network-free" README.md Makefile params.yaml src/simulate_data.py notebooks/README.md docs/MODEL_CARD.md docs/DEMO_SCRIPT.md docs/TECHNICAL_REPORT.md .github/workflows
```

Expected: exit 1 with no matches.

- [ ] **Step 4: Commit Task 3**

```bash
git add README.md Makefile params.yaml src/simulate_data.py notebooks/README.md docs/MODEL_CARD.md docs/DEMO_SCRIPT.md docs/TECHNICAL_REPORT.md
git commit -m "Document synthetic data as local-only fallback"
```

---

### Task 4: Final verification and scope audit

**Files:**
- Verify only; no planned modifications.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: fresh evidence that the real-data CI contract, existing tests, and repository scope are clean.

- [ ] **Step 1: Run focused syntax and regression checks**

```bash
bash -n scripts/verify_dvc_clean_clone.sh
python -m pytest -q tests/test_reproducibility.py tests/test_fetch_data.py tests/test_simulate_data.py
```

Expected: shell syntax exits 0 and all focused tests pass.

- [ ] **Step 2: Run the full fast suite**

```bash
make test-fast
```

Expected: all non-integration tests pass with no failures.

- [ ] **Step 3: Audit the final diff and worktree**

```bash
git diff --check HEAD~3..HEAD
git status --short
git log -4 --oneline
```

Expected: no whitespace errors; only the user's pre-existing untracked
`docs/TECHNICAL_REPORT.docx` remains outside commits; commits are limited to the
plan and Tasks 1-3.

- [ ] **Step 4: Confirm the anti-fallback invariants directly**

```bash
rg -n "continue-on-error|synthetic fallback|steps\.fetch\.outcome" .github/workflows
rg -n "make fetch-data|DVC_REPRO_DATA_MODE=real" .github/workflows/ci.yml .github/workflows/monitoring.yml
```

Expected: the first command has no matches; the second shows mandatory real-data
fetches and CI DVC real mode.
