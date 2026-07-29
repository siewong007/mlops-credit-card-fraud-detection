# 95-plus core contract handoff

## Frozen filenames

- `train.csv`
- `model_valid.csv`
- `calibration.csv`
- `prod_1.csv`
- `prod_2.csv`
- `prod_3.csv`
- `validation_report.json`
- `experiment_comparison.json`
- `promotion_record.json`
- `operating_point.json`
- `metrics_default.json`
- `metrics_operating.json`
- `drift_summary.json`
- `trigger_decisions.json`
- `trigger_log.md`

## Verified core regression gate

Recorded 2026-07-29: `pytest -q -ra` passed 40 tests with zero skips;
`python -m compileall -q src tests`, strict JSON serialization, and
`git diff --check` all exited successfully.
