# Retraining review decision log

Promoted model: **fraud-detector:v1**  
Calibration PR-AUC baseline: **0.6033**  
Operating threshold: **0.98**

A `retrain` result is a recommendation for human review; it does not start training or replace a model.

| Batch | Labels | PR-AUC | Recall | Drifted | Status | Reason codes | Reasons |
|---|---|---:|---:|---:|---|---|---|
| prod_1 | available | 0.810 | 0.818 | 37.9% | warning | FEATURE_DRIFT_WARNING | 37.9% feature drift is in the warning band |
| prod_2 | available | 0.835 | 0.849 | 41.4% | warning | FEATURE_DRIFT_WARNING | 41.4% feature drift is in the warning band |
| prod_3 | available | 0.088 | 0.545 | 58.6% | retrain | PR_AUC_DROP, RECALL_BELOW_FLOOR, FEATURE_DRIFT_RETRAIN | PR-AUC drop 85.4% exceeds 10%; recall 0.545 is below 0.75; 58.6% feature drift exceeds 50% |

Overall status: **retrain**.

Thresholds: PR-AUC drop > 10.0%; recall < 0.75; warning drift 25.0–50.0%; retrain drift > 50.0%.
