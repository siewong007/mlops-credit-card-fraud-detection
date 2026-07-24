import json

import pytest

from src.build_dashboard import SITE_DIR, build
from src.config import MODELS_DIR, REPORTS_DIR

# The dashboard renders artefacts produced by the pipeline. In a fresh checkout
# (e.g. the unit-test step in CI, which runs before the pipeline) they do not
# exist yet, so this integration check skips rather than failing spuriously.
pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "baseline.json").exists()
    or not (REPORTS_DIR / "drift_summary.json").exists(),
    reason="pipeline artefacts not present — run `make pipeline` first",
)


def test_build_writes_site_with_decisions_for_every_batch():
    out = build()
    html = (SITE_DIR / "index.html").read_text()
    summaries = json.loads((REPORTS_DIR / "drift_summary.json").read_text())

    assert (SITE_DIR / "index.html").exists()
    assert out["status"] in {"ok", "warning", "retrain"}
    # every monitored batch appears with a decision badge
    assert len(out["decisions"]) == len(summaries)
    for batch in summaries:
        assert batch["batch"] in html
    assert "Retraining" in html or "retrain" in html


def test_build_states_data_provenance():
    """The published page must always say which dataset produced the figures,
    so synthetic results can never be mistaken for real ones."""
    build()
    html = (SITE_DIR / "index.html").read_text()
    assert "Data provenance" in html
    assert ("REAL" in html) or ("SYNTHETIC" in html) or ("unknown" in html)


def test_build_is_self_contained():
    """No external assets: the page must render offline and under Pages' CSP."""
    build()
    html = (SITE_DIR / "index.html").read_text()
    for bad in ("http://", "https://cdn", "<script"):
        assert bad not in html, f"unexpected external/script reference: {bad}"
