# Credit Card Fraud Detection — MLOps Reorganisation

Group project: reorganising a notebook-based fraud detection workflow into an
MLOps-enabled pipeline. See [PLAN.md](PLAN.md) for architecture, timeline, and
task split. Baseline notebook acknowledgement: [notebooks/README.md](notebooks/README.md).

## Pipeline

Ingestion → Validation (Pandera) → Features → Training + tracking (MLflow) →
Evaluation → Threshold selection → Registration → Batch inference →
Drift monitoring (Evidently) → Retraining trigger

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Download creditcard.csv from Kaggle (mlg-ulb/creditcardfraud) into data/raw/
```

## Run

```bash
make pipeline        # full workflow: ingest → ... → trigger
make test            # unit tests
mlflow ui            # inspect experiment runs
```

Or via Docker: `docker build -t fraud-mlops . && docker run fraud-mlops`

## Repo structure

```
src/            pipeline stages (one module per stage)
tests/          pytest unit tests (run in CI on every push)
data/           raw data + time-based batches (DVC-tracked, not committed)
notebooks/      baseline Kaggle notebook + acknowledgement
reports/        drift reports, figures, trigger log
params.yaml     all knobs: split fractions, model, costs, trigger thresholds
```

## Configuration

Everything configurable lives in `params.yaml`. Reproducibility: pinned
dependencies, fixed seeds, DVC-tracked data, Docker image.
