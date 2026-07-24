# Retraining trigger log

Promoted model: **logistic_regression** | validation baseline PR-AUC: **0.8181**

Rule: retrain if PR-AUC drop > 10% OR recall < 0.75 OR >50% features drifted.

| Batch | PR-AUC | recall | % drifted | decision | reason |
|-------|--------|--------|-----------|----------|--------|
| prod_1 | 0.810 | 0.818 | 38% | ⚠️ warning | 38% features drifted (warning band) |
| prod_2 | 0.835 | 0.849 | 41% | ⚠️ warning | 41% features drifted (warning band) |
| prod_3 | 0.088 | 0.545 | 59% | 🚨 retrain | PR-AUC drop 89% > 10%; recall 0.55 < floor 0.75; 59% features drifted > 50% |
