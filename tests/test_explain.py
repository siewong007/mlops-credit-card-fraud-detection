import json
from src.config import REPORTS_DIR


def test_explainability_evidence_shape():
    report = json.loads((REPORTS_DIR / "explainability.json").read_text())
    assert report["evidence_role"] == "model_explainability"
    assert report["explainer"] == "LinearExplainer"
    assert len(report["top_features"]) == 15
    ranked = [f["mean_abs_shap"] for f in report["top_features"]]
    assert ranked == sorted(ranked, reverse=True)
    assert all(v >= 0 for v in ranked)


def test_explained_transaction_is_auditable():
    et = json.loads((REPORTS_DIR / "explainability.json").read_text())["explained_transaction"]
    assert et["actual_class"] in (0, 1)
    assert 0.0 <= et["predicted_probability"] <= 1.0
    assert et["source_row_index"] >= 0


def test_explainability_traces_to_operating_point():
    report = json.loads((REPORTS_DIR / "explainability.json").read_text())
    op = json.loads((REPORTS_DIR / "operating_point.json").read_text())
    assert report["promoted_model_id"] == op["promoted_model_id"]
    assert report["calibration_data_fingerprint"] == op["calibration_data_fingerprint"]