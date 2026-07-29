import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_versions_and_docker_contract_are_exact():
    """A floating image or dependency can change verification without a commit."""
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    direct_dependencies = [
        line.strip()
        for line in requirements
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert all("==" in line for line in direct_dependencies)
    assert "dvc==3.67.1" in direct_dependencies
    assert "pyarrow==21.0.0" in direct_dependencies

    dockerfile = (ROOT / "Dockerfile").read_text()
    assert (
        "FROM python:3.13.9-slim@sha256:"
        "326df678c20c78d465db501563f3492d17c42a4afe33a1f2bf5406a1d56b0e86"
    ) in dockerfile
    assert "ARG SOURCE_COMMIT" in dockerfile
    assert "SOURCE_COMMIT=${SOURCE_COMMIT}" in dockerfile
    assert "apt-get install -y --no-install-recommends make" in dockerfile
    assert "pip install --no-cache-dir -r requirements.txt" in dockerfile
    for setting in (
        "MPLBACKEND=Agg",
        "DISABLE_PANDERA_IMPORT_WARNING=True",
        "PYTHONUNBUFFERED=1",
    ):
        assert setting in dockerfile
    assert 'CMD ["make", "verify"]' in dockerfile

    ignored = (ROOT / ".dockerignore").read_text().splitlines()
    for path in (
        ".git",
        ".superpowers",
        "data/raw/*",
        "data/batches/*",
        "models/",
        "site/",
        "reports/",
        "mlruns/",
        "mlartifacts/",
        "mlflow.db",
        "submission-private/",
        "dist/",
    ):
        assert path in ignored


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
    assert "params.yaml" in stages["train"]["deps"]
    assert "params" not in stages["train"]
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
    assert "src/evidence.py" in stages["inference"]["deps"]
    drift_outputs = {
        path: options
        for output in stages["drift"]["outs"]
        for path, options in output.items()
    }
    assert {
        "reports/drift/prod_1.json",
        "reports/drift/prod_2.json",
        "reports/drift/prod_3.json",
    } == set(drift_outputs)
    assert all(spec["cache"] is False for spec in drift_outputs.values())
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
