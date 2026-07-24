"""Retraining trigger (Owner: C). Requirement 8, SS13.

Rule (thresholds in params.yaml, justified in the report SS8):
retrain/review if PR-AUC drops >10% vs validation baseline, OR recall < floor,
OR >30% of monitored features drifted; a milder feature-drift level raises a
warning. ``main`` evaluates the rule against every batch's monitoring summary
and writes ``reports/trigger_log.md`` as the audit trail.
"""
import json

from src.config import MODELS_DIR, REPORTS_DIR, load_params


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


def _reasons(baseline_pr_auc: float, batch: dict, params: dict) -> str:
    t = params["trigger"]
    drop = 100 * (baseline_pr_auc - batch["pr_auc"]) / baseline_pr_auc
    out = []
    if drop > t["pr_auc_drop_pct"]:
        out.append(f"PR-AUC drop {drop:.0f}% > {t['pr_auc_drop_pct']}%")
    if batch["recall"] < t["recall_floor"]:
        out.append(f"recall {batch['recall']:.2f} < floor {t['recall_floor']}")
    if batch["pct_drifted_features"] > t["drifted_features_pct"]:
        out.append(f"{batch['pct_drifted_features']:.0f}% features drifted > {t['drifted_features_pct']}%")
    elif batch["pct_drifted_features"] > t["drifted_features_pct"] / 2:
        out.append(f"{batch['pct_drifted_features']:.0f}% features drifted (warning band)")
    return "; ".join(out) or "within all thresholds"


_ICON = {"ok": "✅ ok", "warning": "⚠️ warning", "retrain": "🚨 retrain"}


def main() -> None:
    params = load_params()
    baseline = json.loads((MODELS_DIR / "baseline.json").read_text())
    baseline_pr_auc = baseline["valid_at_0.5"]["pr_auc"]
    summaries = json.loads((REPORTS_DIR / "drift_summary.json").read_text())

    lines = [
        "# Retraining trigger log",
        "",
        f"Promoted model: **{baseline['model']}** | validation baseline PR-AUC: "
        f"**{baseline_pr_auc:.4f}**",
        "",
        f"Rule: retrain if PR-AUC drop > {params['trigger']['pr_auc_drop_pct']}% OR "
        f"recall < {params['trigger']['recall_floor']} OR "
        f">{params['trigger']['drifted_features_pct']}% features drifted.",
        "",
        "| Batch | PR-AUC | recall | % drifted | decision | reason |",
        "|-------|--------|--------|-----------|----------|--------|",
    ]
    for b in summaries:
        d = decide(baseline_pr_auc, b, params)
        lines.append(
            f"| {b['batch']} | {b['pr_auc']:.3f} | {b['recall']:.3f} | "
            f"{b['pct_drifted_features']:.0f}% | {_ICON[d]} | {_reasons(baseline_pr_auc, b, params)} |"
        )
        print(f"{b['batch']}: {d.upper()} — {_reasons(baseline_pr_auc, b, params)}")

    (REPORTS_DIR / "trigger_log.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORTS_DIR/'trigger_log.md'}")


if __name__ == "__main__":
    main()
