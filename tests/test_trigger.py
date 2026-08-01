import json

import pytest

from src.retrain_trigger import (
    build_decision_report,
    evaluate_batch,
    github_annotation,
    main,
    render_markdown,
)


PARAMS = {
    "trigger": {
        "pr_auc_drop_pct": 10,
        "recall_floor": 0.75,
        "drifted_features_pct": 50,
    }
}


def _batch(drift=0, pr_auc=0.9, recall=0.9, labels="available"):
    return {
        "batch": "prod_1",
        "label_status": labels,
        "pct_drifted_features": drift,
        "pr_auc": pr_auc if labels == "available" else None,
        "recall": recall if labels == "available" else None,
        "promoted_model_id": "fraud-detector:v3",
        "operating_threshold": 0.42,
        "batch_data_fingerprint": "a" * 64,
    }


def test_feature_drift_boundaries_are_calibrated():
    assert evaluate_batch(0.9, _batch(drift=24.9), PARAMS)["status"] == "ok"
    at_warning_start = evaluate_batch(0.9, _batch(drift=25), PARAMS)
    assert at_warning_start["status"] == "warning"
    assert at_warning_start["reason_codes"] == ["FEATURE_DRIFT_WARNING"]
    at_retrain_boundary = evaluate_batch(0.9, _batch(drift=50), PARAMS)
    assert at_retrain_boundary["status"] == "warning"
    assert at_retrain_boundary["reason_codes"] == ["FEATURE_DRIFT_WARNING"]
    result = evaluate_batch(0.9, _batch(drift=50.1), PARAMS)
    assert result["status"] == "retrain"
    assert result["reason_codes"] == ["FEATURE_DRIFT_RETRAIN"]


def test_performance_thresholds_are_strict_at_literal_boundaries():
    at_pr_auc_boundary = evaluate_batch(0.9, _batch(pr_auc=0.81), PARAMS)
    assert at_pr_auc_boundary["status"] == "ok"
    assert at_pr_auc_boundary["reason_codes"] == ["WITHIN_THRESHOLDS"]
    beyond_pr_auc_boundary = evaluate_batch(0.9, _batch(pr_auc=0.809), PARAMS)
    assert beyond_pr_auc_boundary["status"] == "retrain"
    assert beyond_pr_auc_boundary["reason_codes"] == ["PR_AUC_DROP"]
    below_recall_boundary = evaluate_batch(0.9, _batch(recall=0.749), PARAMS)
    assert below_recall_boundary["status"] == "retrain"
    assert below_recall_boundary["reason_codes"] == ["RECALL_BELOW_FLOOR"]
    at_recall_boundary = evaluate_batch(0.9, _batch(recall=0.75), PARAMS)
    assert at_recall_boundary["status"] == "ok"
    assert at_recall_boundary["reason_codes"] == ["WITHIN_THRESHOLDS"]


def test_pending_labels_ignore_performance_rules_and_retain_drift_rules():
    result = evaluate_batch(0.9, _batch(drift=10, labels="pending"), PARAMS)
    assert result["status"] == "ok"
    assert result["reason_codes"] == ["LABELS_PENDING"]
    warning = evaluate_batch(0.9, _batch(drift=25, labels="pending"), PARAMS)
    assert warning["status"] == "warning"
    assert warning["reason_codes"] == ["LABELS_PENDING", "FEATURE_DRIFT_WARNING"]
    retrain = evaluate_batch(0.9, _batch(drift=50.1, labels="pending"), PARAMS)
    assert retrain["status"] == "retrain"
    assert retrain["reason_codes"] == ["LABELS_PENDING", "FEATURE_DRIFT_RETRAIN"]


def test_available_one_class_metrics_skip_both_performance_rules_and_keep_drift():
    batch = _batch(drift=25, pr_auc=None, recall=0.0, labels="available")

    result = evaluate_batch(0.9, batch, PARAMS)

    assert result["label_status"] == "available"
    assert result["status"] == "warning"
    assert result["reason_codes"] == [
        "PERFORMANCE_METRICS_UNAVAILABLE",
        "FEATURE_DRIFT_WARNING",
    ]
    assert result["observed"]["pr_auc"] is None
    assert result["observed"]["recall"] == 0.0
    json.dumps(result, allow_nan=False)


def test_report_uses_operating_point_baseline_and_highest_severity():
    batches = [
        {**_batch(drift=0), "batch": "prod_1"},
        {**_batch(drift=30), "batch": "prod_2"},
        {**_batch(drift=60), "batch": "prod_3"},
    ]
    report = build_decision_report(
        batches,
        {
            "threshold": 0.42,
            "promoted_model_id": "fraud-detector:v3",
            "calibration_metrics": {"pr_auc": 0.9},
        },
        PARAMS,
    )
    assert report["baseline"]["pr_auc"] == 0.9
    assert report["overall_status"] == "retrain"
    assert report["decisions"][0]["observed"]["promoted_model_id"] == (
        "fraud-detector:v3"
    )
    assert report["decisions"][0]["observed"]["operating_threshold"] == 0.42
    assert report["decisions"][0]["observed"]["batch_data_fingerprint"] == "a" * 64
    assert github_annotation({"overall_status": "ok"}) is None
    assert github_annotation({"overall_status": "warning"}).startswith("::notice")
    assert github_annotation(report).startswith("::warning")
    assert "human approval" in github_annotation(report)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("promoted_model_id", "fraud-detector:v999", "model"),
        ("operating_threshold", 0.5, "threshold"),
    ],
)
def test_report_rejects_drift_evidence_from_another_operating_point(
    field, value, message
):
    batch = _batch()
    batch[field] = value

    with pytest.raises(ValueError, match=message):
        build_decision_report(
            [batch],
            {
                "threshold": 0.42,
                "promoted_model_id": "fraud-detector:v3",
                "calibration_metrics": {"pr_auc": 0.9},
            },
            PARAMS,
        )


def test_markdown_renders_nullable_performance_values_without_crashing():
    report = build_decision_report(
        [_batch(labels="pending")],
        {
            "threshold": 0.42,
            "promoted_model_id": "fraud-detector:v3",
            "calibration_metrics": {"pr_auc": 0.9},
        },
        PARAMS,
    )
    markdown = render_markdown(report)
    assert "| prod_1 | pending | n/a | n/a |" in markdown
    assert "does not start training or replace a model" in markdown


def test_annotation_cli_emits_only_expected_output_and_rejects_bad_status(tmp_path, capsys):
    ok_path = tmp_path / "ok.json"
    ok_path.write_text(json.dumps({"overall_status": "ok"}), encoding="utf-8")
    main(["--github-annotation", str(ok_path)])
    assert capsys.readouterr().out == ""

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text(json.dumps({"overall_status": "unexpected"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown overall status"):
        main(["--github-annotation", str(malformed_path)])
