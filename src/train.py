"""Training + MLflow experiment tracking (Owner: B). Requirements 3, 7.

Runs the configured experiments (>= 2: a Logistic Regression baseline and a
gradient-boosted improvement), logging params, imbalance-aware metrics and the
model artefact to MLflow for every run. The run with the best validation PR-AUC
is *promoted*: its model + fraud-rate are written to ``models/`` (our stand-in
for a model registry) and, where the tracking backend supports it, registered
in the MLflow Model Registry.
"""
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import features
from src.config import MODELS_DIR, ROOT, batch_dir, ensure_dirs, load_params
from src.evaluate import compute_metrics

EXPERIMENT = "fraud-detection"
# SQLite backend works on mlflow 2.x and 3.x and, unlike the file store,
# supports the Model Registry. Override with MLFLOW_TRACKING_URI if desired.
DEFAULT_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
REGISTERED_MODEL = "fraud-detector"
_MODEL_PATH = MODELS_DIR / "model.joblib"
_BASELINE_PATH = MODELS_DIR / "baseline.json"


def build_model(name: str, params: dict, pos_weight: float):
    """Return an estimator for ``name``. Falls back to RandomForest if xgboost
    is unavailable, so the pipeline still runs anywhere (logged as a note)."""
    rs = params["train"]["random_state"]
    if name == "logistic_regression":
        return LogisticRegression(
            class_weight=params["train"]["class_weight"], max_iter=1000, random_state=rs
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.9,
                scale_pos_weight=pos_weight,  # counter class imbalance
                eval_metric="aucpr",
                random_state=rs,
                n_jobs=-1,
            )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier

            print("[note] xgboost unavailable -> RandomForest fallback")
            return RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=rs
            )
    raise ValueError(f"unknown model {name}")


def main() -> None:
    ensure_dirs()
    params = load_params()
    bdir = batch_dir(params)
    train = pd.read_csv(bdir / "train.csv")
    valid = pd.read_csv(bdir / "valid.csv")

    scaler = features.fit_scaler(train)
    features.save_scaler(scaler)
    X_tr, y_tr = features.xy(features.transform(train, scaler))
    X_va, y_va = features.xy(features.transform(valid, scaler))
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT)
    results = []
    for name in params["train"]["models"]:
        with mlflow.start_run(run_name=name):
            model = build_model(name, params, pos_weight)
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_va)[:, 1]
            metrics = compute_metrics(y_va, proba, threshold=0.5)
            mlflow.log_param("model", name)
            mlflow.log_params({k: params["train"][k] for k in ("class_weight", "random_state")})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
            print(f"{name:>20}: PR-AUC={metrics['pr_auc']:.4f} recall={metrics['recall']:.3f}")
            results.append((name, metrics, model))

    # Promote the best run by validation PR-AUC.
    best_name, best_metrics, best_model = max(results, key=lambda r: r[1]["pr_auc"])
    joblib.dump(best_model, _MODEL_PATH)
    _BASELINE_PATH.write_text(
        json.dumps(
            {"model": best_name, "pos_weight": pos_weight, "valid_at_0.5": best_metrics}, indent=2
        )
    )
    print(f"promoted: {best_name} (valid PR-AUC {best_metrics['pr_auc']:.4f}) -> {_MODEL_PATH}")

    # Best-effort registry (file-store backends may not support it).
    try:
        with mlflow.start_run(run_name=f"promote-{best_name}"):
            info = mlflow.sklearn.log_model(
                best_model, name="model", registered_model_name=REGISTERED_MODEL,
                serialization_format="cloudpickle",
            )
        print(f"registered '{REGISTERED_MODEL}' <- {info.model_uri}")
    except Exception as e:  # noqa: BLE001 - registry is a nice-to-have, never fatal
        print(f"[note] MLflow registry skipped ({type(e).__name__}); promoted bundle in models/")


if __name__ == "__main__":
    main()
