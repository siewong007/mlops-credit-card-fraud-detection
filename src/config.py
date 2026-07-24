"""Shared config loading."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)
