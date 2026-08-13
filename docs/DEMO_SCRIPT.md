# Presentation / demo script (8–10 minutes)

Deliverable per briefing §14. A running order for a group demo that shows the
*reorganisation*, not just a trained model. Suggested split across 3–4 speakers.

The distinguishing move in this demo is **showing a guard refuse bad input**.
Every group can show a pipeline printing green; a gate that catches a deliberate
break is what demonstrates the pipeline produces evidence rather than output.

## Before the demo

```bash
make fetch-data                                  # real ULB dataset, run once
SOURCE_COMMIT="$(git rev-parse HEAD)" make verify
```

Pre-run this and leave the artefacts in place. Live-run only the stage you are
breaking (§"Money moments") — a full `make verify` takes ~36 s on synthetic data
and roughly 2 minutes on the real 285k-row dataset. **Time it on the actual demo
machine.**

### Pre-flight checklist

- [ ] `python --version` on the demo machine reports **3.13.9**.
      `tests/test_evidence_consistency.py` asserts this exactly, so any other
      patch release fails `make verify` at the very end, after the whole
      pipeline has run. If you cannot match it, demo `make test-fast` instead.
- [ ] Docker image pre-built if you plan to show it — it needs the build arg:
      `docker build --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" -t fraud-mlops:verify .`
      Without it the container dies at the manifest stage.
- [ ] `cp data/raw/creditcard.csv /tmp/creditcard.backup.csv` before any
      break-it demo. **Restoring is not optional** — the raw dataset is
      gitignored and has no DVC remote, so a corrupted copy is unrecoverable
      short of re-running `make fetch-data`.
- [ ] Tabs open: MLflow UI, `reports/`, `site/index.html`, the published
      dashboard, and a terminal in the repo root.
- [ ] Numbers on your slides re-derived from *this* run — see §"Honesty" below.

## Running order

| Time | Speaker | Section | What to show / say |
|------|---------|---------|--------------------|
| 0:00–1:00 | A | **Problem & objective** | The business needs to *operate* a fraud model, not train one once. State the learning objective: reorganise a notebook workflow with MLOps tools. Show the before/after diagrams (`docs/workflow_diagrams.md`). |
| 1:00–2:00 | A | **Original workflow & its gaps** | Walk the 5-step notebook and name the weaknesses (no validation, no tracking, random split, fixed threshold, no monitoring, no retraining, not reproducible, no tests). Report §3. |
| 2:00–3:15 | B | **The gate refuses bad data** *(live)* | Money moment 1 below. Corrupt one row, run `python -m src.validate`. It writes `reports/validation_report.json` **and then** fails. Open the report: `overall_status: failed`, and the named checks that failed. The point: it records *why* it refused — the evidence survives the failure. |
| 3:15–4:30 | B | **Pipeline & chronological split** | Narrate `make verify` (pre-run, or run live if you have the time budget). Point at the six-way split — `train` / `model_valid` / `calibration` / `prod_1..3` — and the `<-- drift injected` line on `prod_3`. Stress: split is by **time**, never random. Report §5.1, §8.1. |
| 4:30–5:30 | C | **Tracked experiments & promotion** | MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Two runs, LogReg vs XGBoost. Then `reports/experiment_comparison.json` and `reports/promotion_record.json` — the promoted model, its MLflow run id, registry version, and the data/parameter fingerprints it was trained against. Promotion is by model-validation PR-AUC and is *recorded*, not asserted. |
| 5:30–6:30 | C | **Operating point vs default 0.5** | `reports/figures/threshold_tradeoff.png` and the pair `metrics_operating.json` / `metrics_default.json`. Explain the FN≫FP cost rationale, then the headline trade (see below): threshold 0.98 cuts false positives **1,205 → 66** and cost **1,405 → 666**, for recall 0.917 → 0.750. The comparison is the argument: 0.5 is not a decision, it is a default. |
| 6:30–7:30 | D | **Monitoring & the model-identity contract** *(live)* | `reports/figures/psi_prod_3.png`, then money moment 2: swap the model id in a prediction file and watch `python -m src.drift` refuse it. Monitoring will not score evidence that did not come from the promoted model. Then the PSI-not-KS point (report §8.3). |
| 7:30–8:15 | D | **Review decision, not auto-retrain** | `reports/trigger_decisions.json` and `reports/trigger_log.md`. Show the machine-readable `reason_codes` (`PR_AUC_DROP`, `RECALL_BELOW_FLOOR`, `FEATURE_DRIFT_WARNING`, `LABELS_PENDING`, …) and the per-batch statuses. Say plainly: a `retrain` status is a **recommendation for human review**; nothing retrains or replaces a model automatically. |
| 8:15–9:00 | D | **Reproducibility** | `reports/run_manifest.json` — source commit, Python version, exact dependency pins, data and parameter fingerprints. Then `make verify-dvc`: it archives the commit into a disposable clone and reproduces the whole graph there. Mention the `--quiet` detail (see FAQ) — it shows you understand the tool, not just the command. |
| 9:00–9:45 | D | **Automated monitoring (live)** | The published dashboard: https://siewong007.github.io/mlops-credit-card-fraud-detection/ — rebuilt by a scheduled job that re-scores batches, re-evaluates the trigger and annotates the run when review is due. Strongest closing beat: the evidence publishes itself. Also the safest fallback — it needs no local machine. |
| 9:45–10:00 | all | **Wrap-up** | One sentence each: which requirement your stage satisfied. Close on "the graded change is the reorganisation, and every stage traces to a requirement." |

## Money moments (verified working)

### 1 — The validation gate

```bash
cp data/raw/creditcard.csv /tmp/creditcard.backup.csv    # DO NOT SKIP
python -c "import pandas as pd; d=pd.read_csv('data/raw/creditcard.csv'); d.loc[5,'Amount']=-1; d.to_csv('data/raw/creditcard.csv',index=False)"
python -m src.validate                                    # fails
python -c "import json; r=json.load(open('reports/validation_report.json')); print(r['overall_status'], [c['name'] for c in r['datasets']['raw']['checks'] if not c['passed']])"
cp /tmp/creditcard.backup.csv data/raw/creditcard.csv     # restore
```

Prints `failed ['pandera_schema', 'non_negative_time_and_amount']`. The gate
names what it rejected and where.

### 2 — The prediction contract

```bash
python -c "import pandas as pd; p=pd.read_csv('data/batches/preds_prod_1.csv'); p['promoted_model_id']='fraud-detector:v99'; p.to_csv('data/batches/preds_prod_1.csv',index=False)"
python -m src.drift        # ValueError: prediction promoted model does not match
python -m src.batch_inference   # regenerate clean predictions
```

Dropping a row instead gives `prediction row count does not match the current
batch row count`. Either works; the model-id swap tells the better story.

### Do not use

Editing `proba` in a prediction file **passes undetected** — the contract does
not re-derive `pred` from `proba` and the threshold. Known gap; do not invite it
in Q&A, and do not use it as a demo beat.

## The numbers (authoritative real-data run)

From commit `36b76a0` on the genuine ULB dataset — 284,807 rows, 492 frauds
(0.1727%), SHA-256 `1700322b…`. These are the figures committed in `reports/`.

**Promotion** — `fraud-detector:v1`, logistic regression, on the `model_valid`
slice at threshold 0.5:

| Model | PR-AUC | ROC-AUC | Recall |
|---|---:|---:|---:|
| Logistic Regression | **0.865** | 0.976 | 0.912 |
| XGBoost | 0.835 | 0.983 | 0.846 |

If asked why the lower ROC-AUC model won: PR-AUC is the metric that matters at
0.17% prevalence, and the selection metric is recorded in `promotion_record.json`.

**Operating point** — the strongest slide in the deck:

| | Default 0.5 | Operating 0.98 |
|---|---:|---:|
| Precision | 0.018 | **0.214** |
| Recall | 0.917 | 0.750 |
| False positives | 1,205 | **66** |
| Estimated cost | 1,405 | **666** |

18× fewer false alarms, 53% lower modelled cost, for 17 points of recall.

**Monitoring:**

| Batch | Status | PR-AUC | Recall | Drift |
|---|---|---:|---:|---:|
| prod_1 | warning | 0.810 | 0.818 | 37.9% |
| prod_2 | warning | 0.835 | 0.849 | 41.4% |
| prod_3 | **retrain** | 0.088 | 0.545 | 58.6% |

`prod_3` trips all three criteria: PR-AUC drop 85.4% > 10%, recall 0.545 < 0.75,
drift 58.6% > 50%. Prediction PSI separates them most cleanly — 0.00 and 0.01 on
the healthy batches against 2.69 on the drifted one.

**Own the limitation before you are asked.** The calibration slice holds only
**24 frauds**, so the operating point rests on 18 true positives and 6 false
negatives. The method is sound; the specific 0.98 is less precise than it looks.
Saying so first is stronger than being caught by it (report §8.5).

If you re-run before the demo, re-derive these from your own `reports/` — and if
you demo on synthetic data instead, say so every time a number appears on screen.

## Backup / FAQ

- **"Is this real data?"** The genuine ULB dataset (284,807 transactions) via
  OpenML with `make fetch-data`, no Kaggle account needed. A synthetic generator
  with an identical schema supports local/offline development and is labelled as
  such everywhere it appears. CI fails rather than falling back when real data
  is unavailable. Report §2.4.
- **"Why PSI, not KS, for the drift flag?"** At these batch sizes the KS test
  flags essentially every feature as significant for trivial differences. PSI is
  magnitude-based and stable across sample sizes, so it is the flag; KS is
  reported alongside for information. Report §8.3, §8.5.
- **"Why `dvc status --quiet` and not just `dvc status`?"** Plain `dvc status`
  prints what has drifted and then exits **0**, so a reproducibility gate built
  on it can never fail. `--quiet` exits 1 when the pipeline is not up to date.
  That is the form the verifier uses.
- **"Why is raw data not a DVC stage output?"** DVC deletes a stage's declared
  outputs before running it. A stage that generated `data/raw/creditcard.csv`
  would erase a real dataset on every `dvc repro` — unrecoverably, since the file
  is gitignored with no remote. Raw data is an input; its hash is recorded as a
  dependency, which pins lineage just as precisely.
- **"Does it retrain automatically?"** No, by design. The trigger emits a
  reviewed recommendation with reason codes. Automatic retraining on drift
  without a human in the loop is how you promote a model trained on corrupted
  data. Report §8.4.
- **"Why did Logistic Regression beat XGBoost?"** Only just, and neither is
  tuned — model choice is not the point. Promotion picks whichever wins on
  model-validation PR-AUC; both runs are tracked in MLflow.
- **"How would this run in production?"** Shared MLflow server plus object
  storage, inference as a service, label-delay-aware performance monitoring,
  SHAP for analyst review. Report §9, §10.
