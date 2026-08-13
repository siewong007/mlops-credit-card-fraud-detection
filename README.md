# Credit Card Fraud Detection — MLOps Reorganisation

Group project for the MLOps module: reorganising a notebook-based credit-card
fraud detection workflow into a reproducible, tested, tracked and monitored
MLOps pipeline. The graded contribution is the **reorganisation** — every stage
maps to an operational requirement, not to a higher accuracy score.

- **Plan & task split:** [PLAN.md](PLAN.md)
- **Technical report (4–5k words):** [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)
- **Before/after diagrams:** [docs/workflow_diagrams.md](docs/workflow_diagrams.md)
- **Demo script:** [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md)
- **Reflection template:** [docs/REFLECTION_TEMPLATE.md](docs/REFLECTION_TEMPLATE.md)
- **Baseline notebook acknowledgement:** [notebooks/README.md](notebooks/README.md)
- **Model card:** [docs/MODEL_CARD.md](docs/MODEL_CARD.md)

## Pipeline

```
Validate raw data (Pandera) → Ingest six chronological slices →
Feature processing → Train ≥2 experiments + promote (MLflow) →
Select operating threshold (cost-based) → Evaluate default/operating views →
Explain (SHAP global + local) → Batch inference →
Drift monitoring (KS/PSI + Evidently) → Review decision
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (Git Bash + conda)

The pipeline's Makefile uses POSIX shell commands, so run it in **Git Bash**,
not Command Prompt or PowerShell. Install Git for Windows (which supplies Git
Bash) and Anaconda or Miniconda. Then create the environment and install GNU
Make from Anaconda Prompt:

```cmd
conda create -n mlops_verify python=3.13 -y
conda activate mlops_verify
pip install -r requirements.txt
conda install -c conda-forge make -y
conda init bash
```

Close the prompt, open Git Bash in the repository, and run the same POSIX
commands used by CI:

```bash
conda activate mlops_verify
SOURCE_COMMIT="$(git rev-parse HEAD)" make verify
```

Git records the source commit, while `make` drives the verification contract.
Each now reports its own absence: without `make` the suite fails
`test_make_verify_has_required_gate_order` naming the install command, and
without Git (and no `SOURCE_COMMIT` set) `src.evidence` raises a `ValueError`
explaining the same. CI provides both; the Docker image installs `make` and
receives `SOURCE_COMMIT`, so it does not need Git inside the container.

**Without Anaconda.** `make` is the only thing conda supplies that pip cannot,
so a plain `python -m venv` environment works just as well once GNU Make is
installed separately. winget installs it per-user, needing neither conda nor
administrator rights:

```cmd
winget install -e --id ezwinports.make --scope user
```

Reopen the shell afterwards so the updated PATH takes effect. Verified on
Windows 11 with GNU Make 4.4.1: `make test-fast` then passes in full.

**Data.** The dataset is not committed (144 MB, git-ignored). Two options:

- **Real data (recommended):** `make fetch-data` downloads the genuine ULB
  dataset from its open OpenML mirror — **no Kaggle account needed**. (Or drop a
  Kaggle `creditcard.csv` into `data/raw/` yourself.)
- **Synthetic (offline/CI fallback):** in a clean clone with no raw CSV,
  `make verify` generates a clearly-labelled deterministic dataset with the
  identical schema. See [notebooks/README.md](notebooks/README.md) §"data".

## Run

```bash
# Complete verification; uses an existing real CSV or generates synthetic data
SOURCE_COMMIT="$(git rev-parse HEAD)" make verify

# Authoritative real-data path
make fetch-data
SOURCE_COMMIT="$(git rev-parse HEAD)" make verify

# Fast evidence-independent tests
make test-fast
```

Inspect the tracked experiments and the registered model:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Verify the DVC lineage safely:

```bash
make verify-dvc
```

This archives the exact committed snapshot into a disposable clone and
reproduces deterministic synthetic lineage there. No DVC remote is configured
for the real dataset, and the verifier never tracks or overwrites the caller's
real CSV.

Run the same verification contract in the pinned container:

**Optional — requires Docker.** On Windows, Docker Desktop needs the WSL2
backend and hardware virtualisation enabled, and a managed device may block it
outright with `[WinError 4551] An Application Control policy has blocked this
file`. Local Docker is **not** required for reproducibility evidence: CI builds
this image and runs the identical contract on every push, so the `docker` job in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) is the authoritative
record. Skip this section if Docker is unavailable — nothing else depends on it.

```bash
docker build --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  --tag fraud-mlops:verify .
test "$(docker run --rm --entrypoint python fraud-mlops:verify --version)" \
  = "Python 3.13.9"
docker run --rm fraud-mlops:verify
```

## Automated monitoring (scheduled) & live dashboard

**Dashboard:** https://siewong007.github.io/mlops-credit-card-fraud-detection/

Because this is a *batch* system, "production" is simulated with a scheduled job
rather than a hosted API. [`.github/workflows/monitoring.yml`](.github/workflows/monitoring.yml)
runs every Monday (and on demand via *Actions → Scheduled monitoring → Run
workflow*):

1. fetches the real ULB dataset from OpenML — falling back to the synthetic
   generator if OpenML is unreachable,
2. runs the complete verification contract,
3. publishes the trigger table to the run summary and raises a workflow
   **warning annotation** if any batch meets the retrain criteria,
4. uploads the evidence as a build artefact, and
5. rebuilds and deploys the dashboard to GitHub Pages.

The dashboard always states its **data provenance**, so synthetic results can
never be mistaken for real ones. Build it locally with `make dashboard`.

## What you get (evidence)

After `SOURCE_COMMIT="$(git rev-parse HEAD)" make verify`, `reports/` contains
the generated evidence contracts:

| Artefact | What it shows | Requirement |
|----------|---------------|-------------|
| `reports/validation_report.json` | Required raw/development data validation gate | 6 |
| `reports/experiment_comparison.json` + `reports/promotion_record.json` | Candidate comparison and promoted MLflow identity | 7 |
| `reports/operating_point.json` + `reports/metrics_operating.json` + `reports/figures/threshold_tradeoff.png` + `reports/figures/pr_curve.png` | Primary calibrated operating point and evaluation | 3, 4 |
| `reports/metrics_default.json` + `reports/figures/confusion_matrix_default.png` + `reports/figures/confusion_matrix_operating.png` | Threshold 0.5 comparison against the operating view | 3, 4 |
| `reports/drift/*.json` + `reports/drift_summary.json` | Native per-batch drift and label-aware performance evidence | 2 |
| `reports/trigger_decisions.json` + `reports/trigger_log.md` | Structured human-review decision and reasons | 8 |
| `reports/run_manifest.json` | Source commit, runtime, provenance, and fingerprints | 5 |
| MLflow (`mlflow.db`) | Two tracked runs + registered `fraud-detector` | 7 |

Authoritative real-data values are added only after the technical freeze and
verified real-data evidence run.

## Tools and why

| Tool | Role | Where |
|------|------|-------|
| Git / GitHub | Version control, collaboration, CI substrate | repo |
| MLflow (SQLite backend) | Experiment tracking + Model Registry | `src/train.py` |
| Pandera | Schema/range/null/target validation | `src/validate.py` |
| scipy + native PSI | Dependency-light drift signal (KS/PSI) | `src/drift.py` |
| Evidently | Rich HTML drift reports (best-effort) | `src/drift.py` |
| XGBoost + scikit-learn | The ≥2 model experiments | `src/train.py` |
| Docker | Reproducible runtime | `Dockerfile` |
| DVC | Disposable synthetic lineage, parameters, hashes, metrics, and plots | `dvc.yaml`, `make verify-dvc` |
| Pytest + GitHub Actions | Automated tests + full-pipeline CI | `tests/`, `.github/` |
| SHAP                    | Global + local attribution for audit support                          | `src/explain.py`              |

## Repo structure

```
src/            pipeline stages + fetch_data.py (real) / simulate_data.py (synthetic)
tests/          pytest unit, integration, and evidence-consistency gates
data/           raw data + time-based batches (gitignored, regenerated)
notebooks/      baseline Kaggle notebook acknowledgement
reports/        generated JSON/figure evidence contracts
docs/           report, diagrams, demo script, reflection template
models/         promoted model bundle (gitignored, regenerated)
params.yaml     all knobs: split fractions, models, costs, drift & trigger thresholds
Makefile        stage orchestration                dvc.yaml   synthetic lineage graph
Dockerfile      pinned runtime                     requirements.txt  pinned deps
```

## Configuration & reproducibility

Every knob lives in `params.yaml` (split fractions, model list, business costs,
drift/trigger thresholds, synthetic-data settings). Reproducibility rests on
pinned dependencies, fixed seeds, the shared `make verify` contract, disposable
DVC lineage verification, the pinned Docker image, and CI.

## Academic integrity

Per briefing §18: the baseline notebook is acknowledged and AI assistance is
disclosed. Submitted results must be checked against the authoritative real-data
evidence run. The synthetic dataset is clearly labelled wherever it appears and
demonstrates the *workflow*, never real-data performance.
