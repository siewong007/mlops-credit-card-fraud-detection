"""Shared config + path management.

Single source of truth for every knob (params.yaml) and for the on-disk
layout used to pass artefacts between pipeline stages. Downstream stages load
the *promoted* model bundle from ``models/`` (our lightweight stand-in for a
model registry) and write evidence to ``reports/``.
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


# --- on-disk layout (created on demand) -------------------------------------
MODELS_DIR = ROOT / "models"          # promoted model bundle (gitignored, binary)
REPORTS_DIR = ROOT / "reports"        # committed evidence: json, md, figures
FIGURES_DIR = REPORTS_DIR / "figures"
DRIFT_DIR = REPORTS_DIR / "drift"     # per-batch drift summaries + Evidently html


def ensure_dirs() -> None:
    for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR, DRIFT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def batch_dir(params: dict) -> Path:
    return ROOT / params["data"]["batch_dir"]


def provenance_path(params: dict) -> Path:
    """Data lineage record written by fetch_data/simulate_data and read by the
    dashboard, so published monitoring output always states which dataset
    produced it (real vs synthetic) rather than leaving it implicit."""
    return (ROOT / params["data"]["raw_path"]).parent / "PROVENANCE.json"


def write_provenance(
    params: dict,
    source: str,
    rows: int,
    fraud: int,
    *,
    include_generated_at: bool = True,
) -> None:
    payload = {
        "source": source,
        "rows": rows,
        "fraud": fraud,
        "fraud_rate": round(fraud / rows, 6) if rows else 0.0,
    }
    if include_generated_at:
        from datetime import datetime, timezone

        payload["generated_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
    from src.evidence import write_json

    write_json(provenance_path(params), payload)


def read_provenance(params: dict) -> dict:
    import json

    p = provenance_path(params)
    return json.loads(p.read_text()) if p.exists() else {"source": "unknown"}
