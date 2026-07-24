# Retraining trigger log

Promoted model: **logistic_regression** | validation baseline PR-AUC: **0.9678**

Rule: retrain if PR-AUC drop > 10% OR recall < 0.75 OR >30% features drifted.

| Batch | PR-AUC | recall | % drifted | decision | reason |
|-------|--------|--------|-----------|----------|--------|
| prod_1 | 0.995 | 0.950 | 7% | ✅ ok | within all thresholds |
| prod_2 | 0.942 | 0.867 | 3% | ✅ ok | within all thresholds |
| prod_3 | 0.001 | 0.000 | 31% | 🚨 retrain | PR-AUC drop 100% > 10%; recall 0.00 < floor 0.75; 31% features drifted > 30% |
