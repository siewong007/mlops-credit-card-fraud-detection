"""Retraining trigger (Owner: C). Requirement 8, SS13.

Rule (thresholds in params.yaml, justified in report SS8):
retrain/review if PR-AUC drops >10% vs validation baseline, OR recall < floor,
OR >30% of monitored features drifted.
"""
from src.config import load_params


def decide(baseline_pr_auc: float, batch: dict, params: dict) -> str:
    """batch: {"pr_auc": float, "recall": float, "pct_drifted_features": float}
    Returns one of: "ok", "warning", "retrain".
    """
    t = params["trigger"]
    drop = 100 * (baseline_pr_auc - batch["pr_auc"]) / baseline_pr_auc
    if drop > t["pr_auc_drop_pct"] or batch["recall"] < t["recall_floor"]:
        return "retrain"
    if batch["pct_drifted_features"] > t["drifted_features_pct"]:
        return "retrain"
    if batch["pct_drifted_features"] > t["drifted_features_pct"] / 2:
        return "warning"
    return "ok"


def main() -> None:
    params = load_params()
    # TODO(C): read drift/performance summary JSONs, print decision per batch,
    # write decisions to reports/trigger_log.md as evidence.
    raise SystemExit(f"TODO: implement using rule {params['trigger']}")


if __name__ == "__main__":
    main()
