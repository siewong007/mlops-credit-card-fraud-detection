# Model Card — `fraud-detector`

> Governance document for the promoted credit-card fraud detection model.
> Every figure below is traceable to a generated evidence file in `reports/`
> from a single pipeline run. Regenerate the evidence before amending.

## 1. Model details

| Field | Value |
| --- | --- |
| Registered name | `fraud-detector` |
| Version | `9` (`fraud-detector:v9`) |
| Algorithm | Logistic Regression (scikit-learn 1.7.2) |
| MLflow run ID | `6d70936c8b7240188f549852a514c86a` |
| Model URI | `models:/m-335609cc22ed428ebc4a07097f62ae81` |
| Selection reason | Highest model-validation PR-AUC |
| Source commit | "211bf755211641b2d82ff0b8df1d6cfee1cfdb98" |
| Owner | CY |
| Card last updated | 2026-08-07 |

Evidence: `reports/promotion_record.json`, `reports/experiment_comparison.json`.

## 2. Intended use

**In scope.** Batch scoring of card transactions to rank cases for human
analyst review. The model produces a score; the operating threshold converts
that score into a review queue.

**Out of scope.**

- Automated declines or account actions without human review.
- Real-time / per-transaction inference. The pipeline is batch-only.
- Any transaction population outside the ULB two-day distribution. The model
  has not been validated on other issuers, geographies, or time periods.
- Any use requiring a calibrated probability (see §7).

## 3. Training data

| Property | Value |
| --- | --- |
| Source | ULB Credit Card Fraud Detection, downloaded from Kaggle (`mlg-ulb/creditcardfraud`) |
| Rows | 284,807 |
| Fraud cases | 492 (0.1727%) |
| Features | `Time`, `V1`–`V28` (PCA-transformed), `Amount` |
| Target | `Class` (1 = fraud) |
| Data fingerprint | `76274b69…551a89` (SHA-256) |
| Split | Time-based, no random shuffling |

Splits: earliest slice trains; `model_valid` selects between candidates;
`calibration` fixes the operating threshold; `prod_1`–`prod_3` simulate
production batches for monitoring.

Fraud is unevenly distributed across time slices — the model-validation slice
contains 91 fraud cases while the calibration slice contains only 24. This is
material to the reliability of the operating point (§7).

Provenance is recorded in `data/raw/PROVENANCE.json` and echoed into
`reports/run_manifest.json`. When no real dataset is present the pipeline
generates a clearly-labelled synthetic dataset for dev/CI; synthetic runs are
never reported as real.

## 4. Candidate comparison

Both candidates evaluated on the `model_valid` slice at threshold 0.5.
Selection metric: `model_validation.pr_auc`.

| | Logistic Regression (**promoted**) | XGBoost |
| --- | --- | --- |
| PR-AUC | **0.8652** | 0.8313 |
| ROC-AUC | 0.9765 | 0.9810 |
| Recall | 0.9121 | 0.8352 |
| Precision | 0.0707 | 0.7525 |
| F1 | 0.1312 | 0.7917 |
| FP | 1,091 | 25 |
| FN | 8 | 15 |
| Estimated cost | 1,891 | **1,525** |

The two metrics disagree. PR-AUC is threshold-independent and appropriate for
extreme imbalance, which justifies its use for selection. However, at the
default threshold XGBoost is the cheaper model under the project's own cost
assumptions. The promotion rule therefore optimises ranking quality rather than
the stated business objective. This is a deliberate design choice — threshold
tuning happens downstream of promotion, so a threshold-fixed cost comparison
would prejudge that stage — but it should be revisited if cost becomes the
primary operating criterion.

Evidence: `reports/experiment_comparison.json`.

## 5. Operating point

Selected on the calibration slice by minimising expected cost.

| Assumption | Value |
| --- | --- |
| Cost of a false negative | 100 |
| Cost of a false positive | 1 |
| **Selected threshold** | **0.98** |

| Metric | Threshold 0.5 | **Operating (0.98)** |
| --- | --- | --- |
| Recall | 0.9167 | **0.7500** |
| Precision | 0.0179 | **0.2143** |
| F1 | 0.0352 | **0.3333** |
| PR-AUC | 0.6033 | 0.6033 |
| ROC-AUC | 0.9492 | 0.9492 |
| False positive rate | 0.0423 | **0.0023** |
| TP / FP / FN / TN | 22 / 1,205 / 2 / 27,251 | **18 / 66 / 6 / 28,390** |
| Estimated cost | 1,405 | **666** |

Moving from 0.5 to 0.98 trades six additional missed frauds for 1,139 fewer
false alarms, cutting estimated cost by 53%. Accuracy is deliberately not
reported: at a 0.084% fraud rate a model predicting "legitimate" always would
score above 99.9%.

The 100:1 cost ratio is an **illustrative assumption**, not a real bank's cost
model. The selected threshold is only as defensible as that ratio.

Evidence: `reports/operating_point.json`, `reports/metrics_operating.json`,
`reports/metrics_default.json`, `reports/figures/threshold_tradeoff.png`,
`reports/figures/pr_curve.png`, `reports/figures/confusion_matrix_*.png`.

## 6. Explainability

Global attribution computed with SHAP `LinearExplainer` over a seeded
2,000-row sample of the calibration slice.

| Rank | Feature | Mean \|SHAP\| |
| --- | --- | --- |
| 1 | V6 | 1.386 |
| 2 | V4 | 1.005 |
| 3 | V24 | 0.947 |
| 4 | V14 | 0.862 |
| 5 | V22 | 0.758 |
| 6 | **Amount** | 0.687 |
| 7 | V3 | 0.679 |

**Implementation validation.** For a linear model, SHAP value =
coefficient x (feature - background mean), so mean|SHAP| must equal
|coef| x E|X - mu|. Recomputing that product directly from the model
coefficients reproduces all 15 top mean|SHAP| values to a ratio of **1.0000**.
This is an exact identity, not an approximation, and it is what confirms the
stage is computing SHAP correctly.

**Where the Gaussian assumption breaks.** A weaker check compares mean|SHAP|
against |coef| x sigma. For an approximately normal feature E|X - mu| = 0.798
sigma, so that ratio should sit near 0.798; deviation from it measures
non-normality rather than implementation error. Eleven of the top 15 features
fall in a 0.70–0.85 band. Four sit well below, and all four are heavily skewed:

| Feature | mean\|SHAP\| / (\|coef\| x sigma) | Skew |
| --- | ---: | ---: |
| V20 | 0.444 | −5.2 |
| V8 | 0.480 | −5.3 |
| **Amount** | **0.490** | **+9.2** |
| V2 | 0.626 | −4.1 |

`Amount` is the interpretable case. It is strongly right-skewed, so its standard
deviation is inflated by a small number of large transactions while most
transactions sit near the median. Consequently `Amount` ranks 2nd by
|coef| x sigma but only 6th by SHAP. Coefficient-based importance overstates it;
SHAP reflects its influence on the transactions actually observed. This is the
clearest case where SHAP adds information beyond reading the coefficients.

Ratios are computed against sigma over the full calibration slice; the SHAP
values themselves come from the seeded 2,000-row sample.

**Interpretability limit.** V1–V28 are PCA components published for
confidentiality. SHAP therefore yields statistically valid attribution but not
analyst-readable reason codes — "V6 was low" is not an explanation a fraud
investigator can act on. Only `Amount` and `Time` carry business meaning. A
production deployment on non-anonymised features would not have this constraint.

Evidence: `reports/explainability.json`, `reports/figures/shap_summary.png`,
`reports/figures/shap_local_flagged.png`.

## 7. Limitations

**Probabilities are not calibrated.** The highest-scoring transaction in the
calibration sample (source row 20413) scored p = 0.999945 and is **actually
legitimate** — a maximum-confidence false positive. The cost-based threshold is
therefore selected over a monotone score, not a trustworthy probability.
Ranking is reliable; absolute confidence is not. Probability calibration
(isotonic or Platt scaling on a held-out slice) is identified as future work.

**The operating point rests on 24 fraud cases.** The calibration slice contains
28,480 transactions of which only 24 are fraud. Every calibration figure in §5 —
recall 0.75, precision 0.214, PR-AUC 0.603 — is estimated from those 24
positives, so a single case moves recall by roughly four percentage points.
Confidence intervals around the operating point are wide.

**Calibration PR-AUC is well below model-validation PR-AUC** (0.603 vs 0.865)
despite both being held-out slices. Performance degrades across time even before
the simulated production batches begin, which suggests genuine temporal
non-stationarity rather than a monitoring artefact.

**Two days of data.** The public dataset covers roughly 48 hours. Time-based
batching simulates operational arrival but cannot represent seasonal effects,
long-term fraud evolution, or realistic label delay.

**Labels are assumed immediate.** All monitoring batches report
`label_status: available`. In production, fraud labels arrive weeks later via
chargebacks, so the performance arm of the retraining trigger would be
unavailable in real time and drift signals would have to carry the decision.

**Class imbalance in the promoted model.** At threshold 0.5 the promoted model
produces 1,205 false positives against 22 true positives on calibration data.
It is usable only in combination with the tuned threshold in §5.

## 8. Monitoring and retraining

Three simulated production batches (~28,480 transactions each) are scored and
compared against the calibration baseline.

| Batch | PR-AUC | Recall | Precision | Amount PSI | Prediction PSI | Drifted features | Cost | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prod_1 | 0.810 | 0.818 | 0.329 | 0.004 | 0.002 | 37.9% | 655 | Warning |
| prod_2 | 0.835 | 0.849 | 0.592 | 0.004 | 0.007 | 41.4% | 831 | Warning |
| prod_3 | **0.088** | **0.545** | **0.080** | **0.276** | **2.685** | **58.6%** | 1,138 | **Retrain** |

**Trigger rule.** Review or retrain if any of:

- PR-AUC falls more than 10% below the calibration baseline (0.603), or
- recall at the operating threshold falls below 0.75, or
- more than 50% of monitored features show drift (25–50% raises a warning).

**Observed behaviour.** prod_1 and prod_2 raise feature-drift warnings without
triggering retraining. prod_3 fires all three arms simultaneously — PR-AUC down
85.4%, recall 0.545 below the 0.75 floor, and 58.6% of features drifted — with a
prediction PSI of 2.685 indicating the score distribution has shifted almost
entirely. The trigger discriminates correctly between normal variation and
genuine degradation.

**Caveat on the baseline.** prod_1 and prod_2 both score *above* the calibration
baseline of 0.603, so the PR-AUC arm cannot fire on them regardless of their
behaviour. The calibration slice appears to be an unusually difficult period.
Using a baseline drawn from a harder-than-typical slice makes the performance
trigger conservative; a rolling baseline across several batches would be more
robust.

Evidence: `reports/drift_summary.json`, `reports/drift/prod_*.json`,
`reports/trigger_decisions.json`, `reports/trigger_log.md`.

## 9. Review cadence

| Item | Commitment |
| --- | --- |
| Automated monitoring | Weekly, via `.github/workflows/monitoring.yml` |
| Model card review | On every model promotion, or on any `retrain` trigger |
| Threshold review | Whenever the FN:FP cost assumption changes |
| Owner | CY |

## 10. Reproducing this card

```bash
make fetch-data                                   # or place creditcard.csv in data/raw/
SOURCE_COMMIT="$(git rev-parse HEAD)" make verify
```

All figures above are read directly from the resulting `reports/` files. If any
number in this card cannot be traced to a file in that directory, treat the card
as stale and regenerate before relying on it.
