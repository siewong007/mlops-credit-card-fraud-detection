"""Load/save helpers for artefacts passed between pipeline stages.

The promoted model bundle in ``models/`` is the contract between the training
stage and everything downstream (evaluate / threshold / batch inference). This
is a lightweight local stand-in for a model registry; the same objects are also
logged to MLflow in ``src/train.py``.
"""
import json

import joblib

from src.config import MODELS_DIR, REPORTS_DIR
from src.evidence import write_json

_MODEL_PATH = MODELS_DIR / "model.joblib"
_THRESHOLD_PATH = MODELS_DIR / "threshold.json"


def load_model():
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"{_MODEL_PATH} missing — run `make train` first")
    return joblib.load(_MODEL_PATH)


def save_operating_point(payload: dict) -> None:
    write_json(_THRESHOLD_PATH, payload)
    write_json(REPORTS_DIR / "operating_point.json", payload)


def load_threshold() -> dict:
    if not _THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"{_THRESHOLD_PATH} missing — run `make threshold` first")
    return json.loads(_THRESHOLD_PATH.read_text())
