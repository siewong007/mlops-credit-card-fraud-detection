import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _workflow(name):
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())


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
        "validate", "ingest", "train", "threshold", "evaluate", "explain",
        "inference", "drift", "trigger",
    }
    # DVC erases a stage's outputs before running it, so declaring the raw
    # dataset as an output would destroy a real, unrecoverable ULB CSV on repro.
    produced = {
        path
        for stage in stages.values()
        for output in stage.get("outs", [])
        for path in ([output] if isinstance(output, str) else output)
    }
    assert not any(path.startswith("data/raw/") for path in produced)
    assert "data/raw/creditcard.csv" in stages["validate"]["deps"]
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
    }.issubset(stages["explain"]["deps"])
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


def test_clean_clone_verifier_fails_when_repro_does_not_converge():
    """Plain `dvc status` exits 0 on drift, so the gate needs the --quiet form."""
    script = (ROOT / "scripts" / "verify_dvc_clean_clone.sh").read_text()
    commands = [line.strip() for line in script.splitlines()]
    ordered = [
        "python -m src.simulate_data",  # seeds the graph's raw input
        "python -m dvc repro",
        "python -m dvc status --quiet",  # plain `dvc status` never exits non-zero
    ]
    positions = [commands.index(command) for command in ordered]
    assert positions == sorted(positions)
    assert "set -euo pipefail" in commands


def test_dvc_metadata_has_no_configured_remote():
    """The project must own DVC metadata without pointing raw data at a remote."""
    assert (ROOT / ".dvc" / "config").is_file()
    result = subprocess.run(
        ["dvc", "remote", "list"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    assert result.stdout == ""


def test_ci_workflow_runs_shared_verification_contract_and_docker_proof():
    jobs = _workflow("ci.yml")["jobs"]
    assert set(jobs) == {"verify", "docker"}

    verify_steps = jobs["verify"]["steps"]
    setup = next(step for step in verify_steps if step.get("uses") == "actions/setup-python@v5")
    assert setup["with"]["python-version"] == "3.13.9"

    named = {step["name"]: step for step in verify_steps if "name" in step}
    ordered = [
        "Fast unit tests",
        "DVC clean-clone reproduction",
        "Complete verification",
        "Evidence consistency",
        "Upload successful evidence",
    ]
    names = [step.get("name") for step in verify_steps]
    assert [names.index(name) for name in ordered] == sorted(
        names.index(name) for name in ordered
    )
    assert named["Fast unit tests"]["run"] == "make test-fast"
    assert named["DVC clean-clone reproduction"]["run"].splitlines() == [
        'export DVC_REPRO_EXPORT_DIR="$GITHUB_WORKSPACE/dvc-repro"',
        "make verify-dvc",
    ]
    assert named["Complete verification"]["run"] == (
        'SOURCE_COMMIT="$GITHUB_SHA" make verify'
    )
    assert named["Evidence consistency"]["run"] == (
        "python -m pytest -q tests/test_evidence_consistency.py"
    )

    upload = named["Upload successful evidence"]
    assert "if" not in upload
    uploaded = set(upload["with"]["path"].splitlines())
    assert {
        "reports/*.json",
        "reports/trigger_log.md",
        "reports/figures/**",
        "reports/drift/**",
        "site/**",
        "dvc.yaml",
        "dvc.lock",
        "params.yaml",
        "requirements.txt",
        "dvc-repro/**",
    } == uploaded
    # An absolute path raises the artifact's least common ancestor above the
    # workspace, burying the evidence under /home/runner/work.
    assert not any(path.startswith(("/", "$", "${{")) for path in uploaded)

    docker_named = {
        step["name"]: step for step in jobs["docker"]["steps"] if "name" in step
    }
    assert 'SOURCE_COMMIT="$GITHUB_SHA"' in docker_named[
        "Build immutable verification image"
    ]["run"]
    assert '= "Python 3.13.9"' in docker_named["Assert exact Python runtime"]["run"]
    assert docker_named["Run shared verification contract"]["run"] == (
        "docker run --rm fraud-mlops:verify"
    )


def test_monitoring_workflow_parses_structured_review_status_before_pages_deploy():
    workflow = _workflow("monitoring.yml")
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"schedule", "workflow_dispatch"}

    jobs = workflow["jobs"]
    steps = jobs["monitor"]["steps"]
    setup = next(step for step in steps if step.get("uses") == "actions/setup-python@v5")
    assert setup["with"]["python-version"] == "3.13.9"

    named = {step["name"]: step for step in steps if "name" in step}
    assert {
        "Fetch real dataset (OpenML)",
        "Run verified monitoring simulation",
        "Parse structured review decision",
        "Publish decision to run summary",
        "Upload successful evidence",
    }.issubset(named)
    assert named["Fetch real dataset (OpenML)"]["continue-on-error"] is True
    assert named["Run verified monitoring simulation"]["run"] == (
        'SOURCE_COMMIT="$GITHUB_SHA" make verify'
    )

    decision = named["Parse structured review decision"]
    assert decision["id"] == "decision"
    assert "python -m src.retrain_trigger" in decision["run"]
    assert "--github-annotation reports/trigger_decisions.json" in decision["run"]
    assert 'json.load(open("reports/trigger_decisions.json"))["overall_status"]' in (
        decision["run"]
    )
    assert "ok|warning|retrain" in decision["run"]
    assert 'echo "overall_status=$status" >> "$GITHUB_OUTPUT"' in decision["run"]

    summary = named["Publish decision to run summary"]["run"]
    assert "${{ steps.decision.outputs.overall_status }}" in summary
    assert "cat reports/trigger_log.md" in summary

    upload = named["Upload successful evidence"]
    assert "if" not in upload
    assert "reports/*.json" in upload["with"]["path"].splitlines()
    assert "site/**" in upload["with"]["path"].splitlines()
    assert any(step.get("uses") == "actions/upload-pages-artifact@v3" for step in steps)
    assert jobs["deploy"]["needs"] == "monitor"
