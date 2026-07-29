import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_make_verify_has_required_gate_order():
    """Removing a pipeline dependency must break the verification command order."""
    result = subprocess.run(
        ["make", "-n", "verify"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout
    ordered = [
        "src.validate",
        "src.ingest",
        "src.train",
        "src.threshold",
        "src.evaluate",
        "src.batch_inference",
        "src.drift",
        "src.retrain_trigger",
        "src.evidence",
        "src.build_dashboard",
        "pytest",
    ]
    positions = [output.index(token) for token in ordered]
    assert positions == sorted(positions)


def test_no_test_has_a_skip_marker():
    """A skipped test would hide a verification failure instead of reporting it."""
    source = "\n".join(
        path.read_text() for path in sorted((ROOT / "tests").glob("test_*.py"))
    )
    skip_marker = "pytest.mark." + "skip"
    skip_call = "pytest." + "skip("
    assert skip_marker not in source
    assert skip_call not in source


def test_dvc_graph_tracks_current_artifacts_and_no_retired_paths():
    """The reproducible graph must bind the artifacts each stage really reads."""
    graph = yaml.safe_load((ROOT / "dvc.yaml").read_text())
    stages = graph["stages"]
    assert set(stages) == {
        "data", "validate", "ingest", "train", "threshold", "evaluate",
        "inference", "drift", "trigger",
    }
    assert "data/raw/creditcard.csv" in stages["train"]["deps"]
    assert "data/raw/PROVENANCE.json" in stages["train"]["deps"]
    assert "reports/promotion_record.json" in stages["threshold"]["deps"]
    assert {
        "models/model.joblib", "models/scaler.joblib", "models/threshold.json",
        "data/batches/calibration.csv",
    }.issubset(stages["evaluate"]["deps"])
    assert {
        "models/model.joblib", "models/scaler.joblib", "models/threshold.json",
        "data/batches/calibration.csv",
    }.issubset(stages["drift"]["deps"])
    assert "reports/operating_point.json" in stages["trigger"]["deps"]
    rendered = (ROOT / "dvc.yaml").read_text()
    assert "data/batches/valid.csv" not in rendered
    assert "baseline.json" not in rendered


def test_dvc_metadata_has_no_configured_remote():
    """The project must own DVC metadata without pointing raw data at a remote."""
    assert (ROOT / ".dvc" / "config").is_file()
    result = subprocess.run(
        ["dvc", "remote", "list"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    assert result.stdout == ""
