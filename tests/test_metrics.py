from src.evaluate import compute_metrics
from src.threshold import select_threshold
from src.retrain_trigger import decide

PARAMS = {"trigger": {"pr_auc_drop_pct": 10, "recall_floor": 0.75, "drifted_features_pct": 30}}


def test_perfect_classifier():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    assert m["recall"] == 1.0 and m["precision"] == 1.0 and m["pr_auc"] == 1.0


def test_threshold_prefers_recall_when_fn_costly():
    y = [0, 0, 0, 1]
    proba = [0.1, 0.2, 0.4, 0.45]
    t, _ = select_threshold(y, proba, c_fn=100, c_fp=1)
    assert t <= 0.45  # low threshold catches the fraud case


def test_trigger_decisions():
    ok = {"pr_auc": 0.80, "recall": 0.9, "pct_drifted_features": 5}
    bad_perf = {"pr_auc": 0.60, "recall": 0.9, "pct_drifted_features": 5}
    low_recall = {"pr_auc": 0.80, "recall": 0.5, "pct_drifted_features": 5}
    drifted = {"pr_auc": 0.80, "recall": 0.9, "pct_drifted_features": 40}
    warn = {"pr_auc": 0.80, "recall": 0.9, "pct_drifted_features": 20}
    assert decide(0.82, ok, PARAMS) == "ok"
    assert decide(0.82, bad_perf, PARAMS) == "retrain"
    assert decide(0.82, low_recall, PARAMS) == "retrain"
    assert decide(0.82, drifted, PARAMS) == "retrain"
    assert decide(0.82, warn, PARAMS) == "warning"
