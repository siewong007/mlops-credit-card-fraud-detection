"""Load/save helpers for artefacts passed between pipeline stages.

The promoted model bundle in ``models/`` is the contract between the training
stage and everything downstream (evaluate / threshold / batch inference). This
is a lightweight local stand-in for a model registry; the same objects are also
logged to MLflow in ``src/train.py``.
"""
import json

import joblib

from src.config import MODELS_DIR

_MODEL_PATH = MODELS_DIR / "model.joblib"
_BASELINE_PATH = MODELS_DIR / "baseline.json"
_THRESHOLD_PATH = MODELS_DIR / "threshold.json"


def load_model():
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"{_MODEL_PATH} missing — run `make train` first")
    return joblib.load(_MODEL_PATH)


def load_baseline() -> dict:
    return json.loads(_BASELINE_PATH.read_text())


def save_threshold(payload: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _THRESHOLD_PATH.write_text(json.dumps(payload, indent=2))


def load_threshold() -> dict:
    if not _THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"{_THRESHOLD_PATH} missing — run `make threshold` first")
    return json.loads(_THRESHOLD_PATH.read_text())
