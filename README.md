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

## Pipeline

```
Ingest (time-based batches) → Validate (Pandera) → Feature processing →
Train ≥2 experiments + track (MLflow) → Evaluate (imbalance-aware) →
Threshold selection (cost-based) → Register (MLflow registry) →
Batch inference → Drift monitoring (KS/PSI + Evidently) → Retraining trigger
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Data.** The dataset is not committed (144 MB, git-ignored). Two options:

- **Real data (recommended):** `make fetch-data` downloads the genuine ULB
  dataset from its open OpenML mirror — **no Kaggle account needed**. (Or drop a
  Kaggle `creditcard.csv` into `data/raw/` yourself.)
- **Synthetic (offline/CI fallback):** do nothing — `make pipeline` generates a
  clearly-labelled synthetic dataset with the identical schema so the pipeline
  runs out of the box. See [notebooks/README.md](notebooks/README.md) §"data".

## Run

```bash
make fetch-data   # download the real ULB dataset from OpenML (run once)
make pipeline     # full workflow: (data) → ingest → validate → train →
                  # threshold → evaluate → inference → drift → trigger
make dashboard    # render the monitoring dashboard into site/
make test         # unit tests (pytest)
make clean        # remove generated artefacts (models/, reports/, batches, site/)
```

Inspect the tracked experiments and the registered model:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Reproduce via DVC instead of Make (after `dvc init && dvc add data/raw/creditcard.csv`):

```bash
dvc repro
```

Or run the whole thing in a pinned container:

```bash
docker build -t fraud-mlops . && docker run --rm fraud-mlops
```

## Automated monitoring (scheduled) & live dashboard

**Dashboard:** https://siewong007.github.io/mlops-credit-card-fraud-detection/

Because this is a *batch* system, "production" is simulated with a scheduled job
rather than a hosted API. [`.github/workflows/monitoring.yml`](.github/workflows/monitoring.yml)
runs every Monday (and on demand via *Actions → Scheduled monitoring → Run
workflow*):

1. fetches the real ULB dataset from OpenML — falling back to the synthetic
   generator if OpenML is unreachable,
2. runs the full pipeline (score batches → drift → retraining trigger),
3. publishes the trigger table to the run summary and raises a workflow
   **warning annotation** if any batch meets the retrain criteria,
4. uploads the evidence as a build artefact, and
5. rebuilds and deploys the dashboard to GitHub Pages.

The dashboard always states its **data provenance**, so synthetic results can
never be mistaken for real ones. Build it locally with `make dashboard`.

## What you get (evidence)

After `make pipeline`, `reports/` contains the testing & monitoring evidence:

| Artefact | What it shows | Requirement |
|----------|---------------|-------------|
| `reports/metrics_valid.json` + `figures/confusion_matrix.png`, `figures/pr_curve.png` | Imbalance-aware evaluation of the promoted model | 3 |
| `reports/figures/threshold_tradeoff.png` + `models/threshold.json` | Cost-based threshold selection (FN≫FP) | 4 |
| `reports/drift/*.json`, `figures/psi_*.png`, `reports/drift/*.html` | Per-batch drift (KS/PSI, prediction, performance) | 2 |
| `reports/drift_summary.json` | Machine-readable monitoring summary | 2 |
| `reports/trigger_log.md` | Retraining decision per batch, with reasons | 8 |
| MLflow (`mlflow.db`) | Two tracked runs + registered `fraud-detector` | 7 |

On the real data, `prod_1`/`prod_2` raise a **warning** (the dataset naturally
drifts ~40% of features across its 2-day window, but performance stays healthy)
while the drift-injected `prod_3` trips a **retrain** (PR-AUC −89%, recall 0.55 <
floor, 59% features drifted) — the intended monitoring demonstration. See the
report §8 for the full analysis, including why we flag drift on PSI (not the
KS p-value, which over-fires at this scale).

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
| DVC | Data/artefact versioning + `dvc repro` | `dvc.yaml` |
| Pytest + GitHub Actions | Automated tests + full-pipeline CI | `tests/`, `.github/` |

## Repo structure

```
src/            pipeline stages + fetch_data.py (real) / simulate_data.py (synthetic)
tests/          pytest unit tests (13, run in CI)
data/           raw data + time-based batches (gitignored, regenerated)
notebooks/      baseline Kaggle notebook acknowledgement
reports/        committed evidence: metrics, figures, drift summaries, trigger log
docs/           report, diagrams, demo script, reflection template
models/         promoted model bundle (gitignored, regenerated)
params.yaml     all knobs: split fractions, models, costs, drift & trigger thresholds
Makefile        stage orchestration                dvc.yaml   DVC pipeline
Dockerfile      pinned runtime                     requirements.txt  pinned deps
```

## Configuration & reproducibility

Every knob lives in `params.yaml` (split fractions, model list, business costs,
drift/trigger thresholds, synthetic-data settings). Reproducibility rests on
pinned dependencies, fixed seeds, a single-command pipeline, DVC-tracked data,
a Docker image, and CI that runs the whole workflow on every push.

## Academic integrity

Per briefing §18: the baseline notebook is acknowledged, AI assistance is
disclosed, and all results were verified by running the pipeline. The synthetic
dataset is clearly labelled wherever it appears; it demonstrates the *workflow*,
and the real Kaggle CSV drops in unchanged for submission-grade numbers.
