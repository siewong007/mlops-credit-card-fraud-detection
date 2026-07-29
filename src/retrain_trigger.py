"""Produce review recommendations from monitoring evidence without training models."""
import json

from src.config import REPORTS_DIR, load_params
from src.evidence import write_json


_SEVERITY = {"ok": 0, "warning": 1, "retrain": 2}


def evaluate_batch(baseline_pr_auc: float, batch: dict, params: dict) -> dict:
    """Return a stable, machine-readable review decision for one monitored batch."""
    thresholds = params["trigger"]
    codes = []
    reasons = []
    if batch["label_status"] == "available":
        if batch["pr_auc"] is None or batch["recall"] is None:
            codes.append("PERFORMANCE_METRICS_UNAVAILABLE")
            reasons.append(
                "available labels contain one class; performance rules were not evaluated"
            )
        else:
            drop = 100 * (baseline_pr_auc - batch["pr_auc"]) / baseline_pr_auc
            if drop > thresholds["pr_auc_drop_pct"]:
                codes.append("PR_AUC_DROP")
                reasons.append(
                    f"PR-AUC drop {drop:.1f}% exceeds "
                    f"{thresholds['pr_auc_drop_pct']}%"
                )
            if batch["recall"] < thresholds["recall_floor"]:
                codes.append("RECALL_BELOW_FLOOR")
                reasons.append(
                    f"recall {batch['recall']:.3f} is below "
                    f"{thresholds['recall_floor']}"
                )
    else:
        codes.append("LABELS_PENDING")
        reasons.append("labels pending; performance rules were not evaluated")

    drift = batch["pct_drifted_features"]
    retrain_drift = thresholds["drifted_features_pct"]
    warning_drift = retrain_drift / 2
    if drift > retrain_drift:
        codes.append("FEATURE_DRIFT_RETRAIN")
        reasons.append(f"{drift:.1f}% feature drift exceeds {retrain_drift}%")
    elif drift >= warning_drift:
        codes.append("FEATURE_DRIFT_WARNING")
        reasons.append(f"{drift:.1f}% feature drift is in the warning band")

    if any(
        code in codes
        for code in ("PR_AUC_DROP", "RECALL_BELOW_FLOOR", "FEATURE_DRIFT_RETRAIN")
    ):
        status = "retrain"
    elif "FEATURE_DRIFT_WARNING" in codes:
        status = "warning"
    else:
        status = "ok"
    if not codes:
        codes = ["WITHIN_THRESHOLDS"]
        reasons = ["all available signals are within thresholds"]
    return {
        "batch": batch["batch"],
        "status": status,
        "label_status": batch["label_status"],
        "reason_codes": codes,
        "reasons": reasons,
        "observed": {
            "pr_auc": batch["pr_auc"],
            "recall": batch["recall"],
            "pct_drifted_features": drift,
            "promoted_model_id": batch["promoted_model_id"],
            "operating_threshold": batch["operating_threshold"],
            "batch_data_fingerprint": batch["batch_data_fingerprint"],
        },
    }


def build_decision_report(
    batches: list[dict], operating_point: dict, params: dict
) -> dict:
    """Build the complete review report using the calibrated operating baseline."""
    for batch in batches:
        if batch["promoted_model_id"] != operating_point["promoted_model_id"]:
            raise ValueError(
                f"drift summary model does not match operating point: "
                f"{batch['batch']}"
            )
        if float(batch["operating_threshold"]) != float(
            operating_point["threshold"]
        ):
            raise ValueError(
                f"drift summary threshold does not match operating point: "
                f"{batch['batch']}"
            )
    baseline_pr_auc = operating_point["calibration_metrics"]["pr_auc"]
    decisions = [
        evaluate_batch(baseline_pr_auc, batch, params) for batch in batches
    ]
    overall = max(
        (decision["status"] for decision in decisions), key=_SEVERITY.__getitem__
    )
    trigger = params["trigger"]
    return {
        "baseline": {
            "pr_auc": baseline_pr_auc,
            "operating_threshold": operating_point["threshold"],
            "promoted_model_id": operating_point["promoted_model_id"],
        },
        "thresholds": {
            "pr_auc_drop_pct": float(trigger["pr_auc_drop_pct"]),
            "recall_floor": float(trigger["recall_floor"]),
            "warning_drift_pct": float(trigger["drifted_features_pct"] / 2),
            "retrain_drift_pct": float(trigger["drifted_features_pct"]),
        },
        "decisions": decisions,
        "overall_status": overall,
    }


def github_annotation(report: dict) -> str | None:
    """Return a GitHub Actions annotation, or no output for healthy batches."""
    status = report.get("overall_status")
    if status == "ok":
        return None
    if status == "warning":
        return (
            "::notice title=Monitoring warning::Signals entered the warning band; "
            "continue monitoring."
        )
    if status == "retrain":
        return (
            "::warning title=Retraining review recommended::At least one batch "
            "met the review criteria; human approval is required before retraining."
        )
    raise ValueError(f"unknown overall status: {status!r}")


def _display(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_markdown(report: dict) -> str:
    """Render the strict decision payload as an audit-friendly Markdown log."""
    baseline = report["baseline"]
    thresholds = report["thresholds"]
    lines = [
        "# Retraining review decision log",
        "",
        f"Promoted model: **{baseline['promoted_model_id']}**  ",
        f"Calibration PR-AUC baseline: **{baseline['pr_auc']:.4f}**  ",
        f"Operating threshold: **{baseline['operating_threshold']:.2f}**",
        "",
        "A `retrain` result is a recommendation for human review; it does not "
        "start training or replace a model.",
        "",
        "| Batch | Labels | PR-AUC | Recall | Drifted | Status | Reason codes | Reasons |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for decision in report["decisions"]:
        observed = decision["observed"]
        lines.append(
            f"| {decision['batch']} | {decision['label_status']} | "
            f"{_display(observed['pr_auc'])} | {_display(observed['recall'])} | "
            f"{observed['pct_drifted_features']:.1f}% | {decision['status']} | "
            f"{', '.join(decision['reason_codes'])} | "
            f"{'; '.join(decision['reasons'])} |"
        )
    lines.extend(
        [
            "",
            f"Overall status: **{report['overall_status']}**.",
            "",
            "Thresholds: PR-AUC drop > "
            f"{thresholds['pr_auc_drop_pct']:.1f}%; recall < "
            f"{thresholds['recall_floor']:.2f}; warning drift "
            f"{thresholds['warning_drift_pct']:.1f}–"
            f"{thresholds['retrain_drift_pct']:.1f}%; retrain drift > "
            f"{thresholds['retrain_drift_pct']:.1f}%.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_decision_stage() -> dict:
    """Write the review recommendation JSON and its derived Markdown log."""
    params = load_params()
    operating_point = json.loads((REPORTS_DIR / "operating_point.json").read_text())
    batches = json.loads((REPORTS_DIR / "drift_summary.json").read_text())
    report = build_decision_report(batches, operating_point, params)
    write_json(REPORTS_DIR / "trigger_decisions.json", report)
    (REPORTS_DIR / "trigger_log.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


def main(argv=None) -> None:
    """Run the stage, or print a workflow annotation from an existing report."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--github-annotation", type=Path)
    args = parser.parse_args(argv)
    if args.github_annotation:
        report = json.loads(args.github_annotation.read_text())
        annotation = github_annotation(report)
        if annotation:
            print(annotation)
        return
    report = run_decision_stage()
    print(f"overall monitoring status: {report['overall_status']}")


if __name__ == "__main__":
    main()
