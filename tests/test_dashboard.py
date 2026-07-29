import json

import pytest

from src.build_dashboard import build


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evidence(tmp_path):
    reports = tmp_path / "reports"
    site = tmp_path / "site"
    _write(
        reports / "promotion_record.json",
        {
            "promoted_model_id": "fraud-detector:v3",
            "model_name": "logistic_regression",
            "mlflow_run_id": "run-123",
            "registered_model_version": "3",
            "data_fingerprint": "a" * 64,
        },
    )
    _write(
        reports / "operating_point.json",
        {
            "threshold": 0.42,
            "cost_assumptions": {"false_negative": 100, "false_positive": 1},
            "promoted_model_id": "fraud-detector:v3",
        },
    )
    _write(
        reports / "metrics_operating.json",
        {
            "pr_auc": 0.84,
            "roc_auc": 0.96,
            "recall": 0.81,
            "precision": 0.40,
            "threshold": 0.42,
            "evidence_role": "primary_operating_point",
        },
    )
    _write(
        reports / "metrics_default.json",
        {
            "pr_auc": 0.84,
            "roc_auc": 0.96,
            "recall": 0.55,
            "precision": 0.70,
            "threshold": 0.5,
            "evidence_role": "comparison_default_threshold",
        },
    )
    _write(
        reports / "drift_summary.json",
        [
            {
                "batch": "prod_1",
                "label_status": "pending",
                "n_rows": 100,
                "pr_auc": None,
                "recall": None,
                "precision": None,
                "pct_drifted_features": 10,
                "amount_psi": 0.03,
                "prediction_psi": 0.05,
            }
        ],
    )
    _write(
        reports / "trigger_decisions.json",
        {
            "overall_status": "ok",
            "decisions": [
                {
                    "batch": "prod_1",
                    "status": "ok",
                    "label_status": "pending",
                    "reason_codes": ["LABELS_PENDING"],
                    "reasons": ["labels pending"],
                }
            ],
        },
    )
    _write(
        reports / "run_manifest.json",
        {"data": {"source": "REAL — test", "fingerprint_sha256": "a" * 64}},
    )
    (reports / "figures").mkdir()
    (reports / "figures" / "pr_curve.png").write_bytes(b"local figure")
    (reports / "drift").mkdir()
    (reports / "drift" / "prod_1.html").write_text(
        "<p>local drift report</p>", encoding="utf-8"
    )
    return reports, site


def test_build_renders_joined_operating_evidence_before_default_comparison(tmp_path):
    """Catches the renderer using legacy/default evidence instead of supplied records."""
    reports, site = _evidence(tmp_path)

    out = build(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert out["status"] == "ok"
    assert html.index("Operating threshold") < html.index("Default threshold comparison")
    for expected in (
        "fraud-detector:v3",
        "run-123",
        "Version 3",
        "REAL — test",
        "0.42",
        "false negative 100",
        "LABELS_PENDING",
        "labels pending",
        "pending",
        "n/a",
        "requires human approval",
    ):
        assert expected in html
    assert (site / "figures" / "pr_curve.png").read_bytes() == b"local figure"
    assert (site / "drift" / "prod_1.html").read_text(encoding="utf-8") == "<p>local drift report</p>"


def test_build_rejects_drift_and_decision_batch_mismatch(tmp_path):
    """Catches silently rendering unrelated drift and trigger evidence together."""
    reports, site = _evidence(tmp_path)
    _write(
        reports / "trigger_decisions.json",
        {"overall_status": "ok", "decisions": [{"batch": "prod_2", "status": "ok"}]},
    )

    with pytest.raises(ValueError, match="batch"):
        build(reports, site)


def test_build_renders_null_evidence_values_as_na(tmp_path):
    """Catches nullable evidence leaking as an exception or literal None."""
    reports, site = _evidence(tmp_path)
    summaries = json.loads((reports / "drift_summary.json").read_text(encoding="utf-8"))
    summaries[0]["n_rows"] = None
    _write(reports / "drift_summary.json", summaries)
    _write(
        reports / "run_manifest.json",
        {"data": {"source": None, "fingerprint_sha256": None}},
    )

    build(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    assert "None" not in html
    assert "<code>prod_1</code></td><td>n/a</td>" in html
    assert html.count("n/a") >= 5


def test_build_is_self_contained(tmp_path):
    """Catches a dashboard that depends on a remote asset or executable script."""
    reports, site = _evidence(tmp_path)

    build(reports, site)

    html = (site / "index.html").read_text(encoding="utf-8")
    for forbidden in ("http://", "https://", "<script"):
        assert forbidden not in html
