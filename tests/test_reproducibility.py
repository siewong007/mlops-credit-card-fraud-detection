import subprocess
from pathlib import Path


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
