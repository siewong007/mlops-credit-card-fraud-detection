# Project Plan — Credit Card Fraud Detection MLOps Reorganisation

Group project, 4 weeks, 4 members. Goal: reorganise a Kaggle notebook-based fraud
detection workflow into a reproducible, tested, monitored MLOps pipeline.

## 1. Workflow: before and after

**Original:** Dataset → notebook preprocessing → training → evaluation → manual saving

**Reorganised:**
Data ingestion → Data validation (Pandera) → Feature processing → Training →
Experiment tracking (MLflow) → Imbalance-aware evaluation → Threshold selection →
Model registration (MLflow registry) → Batch inference → Drift monitoring (Evidently) →
Retraining trigger → Updated model version

Both diagrams are a required deliverable — draw them early (draw.io/Mermaid) and reuse
in report and slides.

## 2. Tool stack mapped to requirements

| # | Requirement | Tool / practice | Where in repo |
|---|-------------|-----------------|---------------|
| 1 | Repeatable ingestion of new batches | Parameterised `src/ingest.py`, DVC-tracked data | `src/ingest.py`, `dvc.yaml` |
| 2 | Fraud patterns change over time | Evidently drift reports per batch | `src/drift.py` |
| 3 | Fraud is rare (0.172%) | PR-AUC, recall, precision, confusion matrix; class weights / resampling as experiment | `src/evaluate.py` |
| 4 | Asymmetric FN/FP costs | Threshold sweep + cost-based selection | `src/threshold.py` |
| 5 | Reproducibility | Git, pinned `requirements.txt`, Docker, DVC, seeded runs, README | root |
| 6 | Data quality checks before train/infer | Pandera schema (types, ranges, nulls, target values) | `src/validate.py` |
| 7 | Experiment tracking | MLflow: params, metrics, artefacts, model registry | `src/train.py` |
| 8 | Justified retraining decisions | Rule-based trigger over drift + performance signals | `src/retrain_trigger.py` |

Cross-cutting: Pytest for unit tests, GitHub Actions runs validation + tests on every
push, Docker for environment reproducibility.

## 3. Batch simulation strategy

Split by `Time` (no random-only splitting):

- Batch 0 (earliest ~50%): training
- Batch 1 (~20%): validation + threshold tuning; freeze baseline metrics here
- Batches 2–4 (remaining, equal slices): simulated production batches for inference,
  drift, and performance monitoring

Optionally inject a synthetic shift (e.g. scale `Amount` in one batch) to demonstrate
the drift warning firing. Acknowledge in the report that 2 days of data ≠ real
long-term monitoring.

## 4. Experiments (minimum two runs)

1. Baseline: Logistic Regression with class weights.
2. Improved: XGBoost or LightGBM (or same model + resampling/threshold strategy).

Both logged to MLflow with identical metric set; best model registered.

## 5. Retraining trigger (draft rule)

Retrain/review if any of:

- PR-AUC on a production batch drops >10% vs validation baseline, or
- recall at the selected threshold falls below agreed floor (e.g. 0.75), or
- drift detected (KS test / PSI) in >30% of monitored features.

Justify thresholds in report; discuss risks: false drift alarms, delayed fraud labels,
retraining on bad data, cost of frequent retraining.

## 6. Four-week timeline

| Week | Milestone | Detail |
|------|-----------|--------|
| 1 | Foundations | Select + acknowledge baseline Kaggle notebook; repo up; data via DVC; Pandera schema; ingest/split by Time; before/after diagrams drafted |
| 2 | Training + tracking | Feature processing, both experiment runs in MLflow, imbalance-aware evaluation, threshold analysis, model registered; unit tests + CI green |
| 3 | Monitoring | Batch inference, Evidently drift reports across batches, performance-over-batches analysis, retraining trigger implemented and demonstrated; Docker image |
| 4 | Deliverables | Technical report (4,000–5,000 words), polish README + reproducibility check on clean machine, 8–10 min demo, individual reflections (500–800 words each) |

Buffer rule: code freeze end of week 3; week 4 is writing and rehearsal only.

## 7. Task split (4 members)

| Member | Workstream | Owns |
|--------|-----------|------|
| A | Data engineering | Ingestion, batch splitting, Pandera validation, DVC |
| B | Modelling | Training, MLflow tracking, evaluation, threshold analysis |
| C | Monitoring | Batch inference, Evidently drift, retraining trigger |
| D | Platform + docs | CI (GitHub Actions), Docker, Pytest coordination, README, report assembly |

Everyone: reviews PRs, writes their report sections + individual reflection. Report
section ownership: A→§2–3, B→§6–7, C→§8, D→§5+§9; intro/conclusion shared.

## 8. Deliverables checklist

- [ ] Repo: source, README, dependency file, reproducibility instructions
- [ ] Technical report 4,000–5,000 words (structure §15 of briefing)
- [ ] Before/after workflow diagrams
- [ ] Evidence: validation output, evaluation results, drift reports, trigger logic
- [ ] 8–10 min presentation/demo
- [ ] Individual reflections, 500–800 words each
- [ ] Baseline notebook acknowledged + modifications explained (academic integrity)

## 9. Rubric alignment

Reorganisation story + architecture (20%) → diagrams and §1–2 here; tool use (20%) →
justify every tool in report, not just use it; testing/monitoring/retraining (20%) →
weeks 2–3 outputs; implementation quality (15%) → CI, tests, code review; repro + docs
(15%) → README verified on a clean machine; presentation + reflection (10%) → week 4.

Key warning from the briefing: accuracy is not the goal — the graded evidence is the
reorganisation itself. Every design choice should trace back to one of the 8
requirements.
