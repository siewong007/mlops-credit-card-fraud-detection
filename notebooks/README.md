# Baseline notebook — acknowledgement

Per the briefing (§1 and §18 Academic Integrity), this project reorganises an
existing public Kaggle notebook rather than reproducing it. The baseline is
acknowledged here; the reorganisation and the MLOps components we added are
described in [`../PLAN.md`](../PLAN.md) §2 and in the technical report
[`../docs/TECHNICAL_REPORT.md`](../docs/TECHNICAL_REPORT.md).

> ⚠️ **Group action:** confirm the line below names the exact notebook your
> group selected. If you chose a different one, update the link, author, and the
> "what we kept" notes. Do not submit an acknowledgement you have not verified.

## Selected baseline

- **Notebook:** *Credit Fraud || Dealing with Imbalanced Datasets*
- **Author:** Janio Martinez Bachmann (Kaggle user `janiobachmann`)
- **URL:** https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets
- **Dataset:** ULB Credit Card Fraud Detection — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Why selected.** It is a widely-referenced, self-contained treatment of the
exact dataset that covers the core experimental steps (EDA, scaling, handling
class imbalance, a Logistic Regression baseline, and imbalance-aware evaluation
with precision/recall/ROC). That makes it a representative "experimental
notebook" of the kind the briefing asks us to operationalise.

## What we kept

- The dataset and its schema (`Time`, `V1..V28`, `Amount`, `Class`).
- The modelling intuition: scale `Amount`, treat the problem as highly
  imbalanced, and use a Logistic Regression baseline.
- Imbalance-aware evaluation (precision, recall, F1, ROC-AUC, PR-AUC, confusion
  matrix) rather than accuracy.

## What we changed / added (MLOps reorganisation)

| Notebook weakness (MLOps view)        | What we added                                             | Where |
|---------------------------------------|-----------------------------------------------------------|-------|
| Ad-hoc cells, run top-to-bottom       | Modular pipeline stages, one module per stage             | `src/` |
| No data validation                    | Pandera schema (types, ranges, nulls, target)             | `src/validate.py` |
| No experiment tracking                | MLflow params/metrics/artefacts + Model Registry          | `src/train.py` |
| Random-only split                     | Time-based batches (train/valid/prod) — no leakage        | `src/ingest.py` |
| Fixed 0.5 threshold                   | Cost-based threshold selection (FN ≫ FP)                  | `src/threshold.py` |
| No monitoring                         | Per-batch drift (KS/PSI) + performance + Evidently report | `src/drift.py` |
| No retraining logic                   | Justified, rule-based retraining trigger + audit log      | `src/retrain_trigger.py` |
| Not reproducible                      | Pinned deps, Docker, DVC stages, seeds, CI, README        | repo root |

## Note on data used for development

The real Kaggle CSV requires an account and is not committed. For development
and CI we generate a **synthetic** dataset with the identical schema
(`src/simulate_data.py`, clearly labelled). All results produced from synthetic
data are for demonstrating the *workflow*; drop the real `creditcard.csv` into
`data/raw/` for submission-grade numbers. This is acknowledged throughout, in
line with §18.
