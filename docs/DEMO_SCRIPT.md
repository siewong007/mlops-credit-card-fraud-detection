# Presentation / demo script (8–10 minutes)

Deliverable per briefing §14. A running order for a group demo that shows the
*reorganisation*, not just a trained model. Suggested split across 3–4 speakers.
Rehearse once end-to-end; the live pipeline run takes ~1 minute, so pre-run it
and have the terminal + `reports/` open as a fallback.

Before the demo, run once so artefacts exist:
```bash
make pipeline
```

| Time | Speaker | Section | What to show / say |
|------|---------|---------|--------------------|
| 0:00–1:00 | A | **Problem & objective** | The business needs to *operate* a fraud model, not just train one once. State the learning objective: reorganise a notebook workflow with MLOps tools. Show the **before/after diagrams** (`docs/workflow_diagrams.md`). |
| 1:00–2:00 | A | **Original workflow & its gaps** | Walk the 5-step notebook and name the 8 weaknesses (no validation, no tracking, random split, fixed threshold, no monitoring, no retraining, not reproducible, no tests). |
| 2:00–3:30 | B | **Reorganised pipeline, live** | Run `make pipeline`. While it runs, narrate the stages. Point out ingest's **time-based** split and the `<-- drift injected` line on `prod_3`. |
| 3:30–5:00 | B | **Experiment tracking** | Open MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Show the two runs (LogReg vs XGBoost), their PR-AUC/recall, and the **registered** `fraud-detector` model. Emphasise automatic promotion by PR-AUC. |
| 5:00–6:00 | C | **Validation & threshold** | Show the Pandera schema and a deliberately broken row failing. Show `reports/figures/threshold_tradeoff.png` and explain the FN≫FP cost choice → threshold 0.95. |
| 6:00–7:30 | C | **Monitoring & drift** | Show `reports/figures/psi_prod_3.png` (only injected features drift) and open an Evidently HTML report from `reports/drift/`. Contrast stable `prod_1`/`prod_2` with drifted `prod_3`. |
| 7:30–8:30 | D | **Retraining trigger** | Show `reports/trigger_log.md`: prod_1/2 OK, prod_3 → RETRAIN with all three reasons. Explain the rule and its justification, and why recall-at-threshold matters (prod_3's ranking looks fine but the operating point collapsed). |
| 8:30–9:30 | D | **Reproducibility** | Show pinned `requirements.txt`, `Dockerfile`, `dvc.yaml`, and the green **CI run** executing the whole pipeline on every push. `make clean && make pipeline` reproduces everything. |
| 9:30–10:00 | all | **Wrap-up** | One sentence each: which requirement your stage satisfied. Close on "the graded change is the reorganisation, and every stage traces to a requirement." |

## Backup / FAQ answers

- *"Is this real data?"* No — a clearly-labelled synthetic dataset with the
  identical schema so the pipeline runs in CI. The real Kaggle CSV drops into
  `data/raw/` unchanged. All caveats are in the report §2.4.
- *"Why did Logistic Regression beat XGBoost?"* The synthetic signal is near
  linearly separable. On real PCA data a boosted model usually leads; the
  promotion logic picks whichever wins by PR-AUC.
- *"Why PSI **and** KS?"* PSI measures distribution shift magnitude; KS tests
  significance. Using both, plus a warning band, reduces false alarms.
- *"How would this run in production?"* Shared MLflow server + object storage,
  inference as a service, label-delay-aware performance monitoring, SHAP for
  analyst review. See report §9 and §10.
