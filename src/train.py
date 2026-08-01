"""Training + traceable MLflow candidate promotion."""
import json
import os
from pathlib import Path
import tempfile

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression

from src import features
from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    ROOT,
    batch_dir,
    ensure_dirs,
    load_params,
    read_provenance,
)
from src.evidence import sha256_file, write_json
from src.evaluate import compute_metrics

EXPERIMENT = "fraud-detection"
# SQLite backend works on mlflow 2.x and 3.x and, unlike the file store,
# supports the Model Registry. Override with MLFLOW_TRACKING_URI if desired.
DEFAULT_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
REGISTERED_MODEL = "fraud-detector"
_MODEL_PATH = MODELS_DIR / "model.joblib"


def build_model(name: str, params: dict, pos_weight: float):
    """Return an estimator for ``name``. Falls back to RandomForest if xgboost
    is unavailable, so the pipeline still runs anywhere (logged as a note)."""
    rs = params["train"]["random_state"]
    n_jobs = params["train"]["n_jobs"]
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
                n_jobs=n_jobs,
            )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier

            print("[note] xgboost unavailable -> RandomForest fallback")
            return RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                n_jobs=n_jobs,
                random_state=rs,
            )
    raise ValueError(f"unknown model {name}")


def normalise_mlflow_param(value) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return "None" if value is None else value
    return json.dumps(value, sort_keys=True, default=str)


def log_candidate(
    name,
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    metadata,
    scaler_path,
) -> dict:
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_valid)[:, 1]
        metrics = compute_metrics(
            y_valid,
            probability,
            threshold=0.5,
            cost_false_negative=metadata["cost_false_negative"],
            cost_false_positive=metadata["cost_false_positive"],
        )
        params = {
            f"estimator.{key}": normalise_mlflow_param(value)
            for key, value in model.get_params(deep=False).items()
        }
        mlflow.log_params(params)
        mlflow.log_param("model_name", name)
        mlflow.log_param("imbalance", metadata["imbalance"])
        mlflow.log_param("n_jobs", metadata["n_jobs"])
        mlflow.log_param("cost_false_negative", metadata["cost_false_negative"])
        mlflow.log_param("cost_false_positive", metadata["cost_false_positive"])
        mlflow.set_tags(
            {
                "source_provenance": metadata["source"],
                "data_fingerprint": metadata["data_fingerprint"],
                "parameter_fingerprint": metadata["parameter_fingerprint"],
            }
        )
        mlflow.log_metrics(
            {
                f"model_validation.{key}": value
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and np.isfinite(value)
            }
        )
        mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")
        model_info = mlflow.sklearn.log_model(
            model, name="model", serialization_format="cloudpickle"
        )
        return {
            "model_name": name,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "model_validation_metrics": metrics,
            "model": model,
        }


def choose_promoted_candidate(candidates: list[dict]) -> dict:
    return max(
        candidates,
        key=lambda candidate: candidate["model_validation_metrics"]["pr_auc"],
    )


def register_winner(
    candidate: dict,
    registered_model_name: str = REGISTERED_MODEL,
) -> str:
    registered = mlflow.register_model(candidate["model_uri"], registered_model_name)
    client = MlflowClient()
    version = str(registered.version)
    client.set_registered_model_alias(registered_model_name, "champion", version)
    return version


def main() -> None:
    ensure_dirs()
    params = load_params()
    bdir = batch_dir(params)
    train = pd.read_csv(bdir / "train.csv")
    model_valid = pd.read_csv(bdir / "model_valid.csv")

    scaler = features.fit_scaler(train)
    X_tr, y_tr = features.xy(features.transform(train, scaler))
    X_model_valid, y_model_valid = features.xy(features.transform(model_valid, scaler))
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    raw_path = ROOT / params["data"]["raw_path"]
    provenance = read_provenance(params)
    metadata = {
        "source": json.dumps(provenance, sort_keys=True),
        "data_fingerprint": sha256_file(raw_path),
        "parameter_fingerprint": sha256_file(ROOT / "params.yaml"),
        "imbalance": params["train"]["class_weight"],
        "n_jobs": params["train"]["n_jobs"],
        "cost_false_negative": params["threshold"]["cost_false_negative"],
        "cost_false_positive": params["threshold"]["cost_false_positive"],
    }

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT)
    with tempfile.TemporaryDirectory() as staging_dir:
        staged_scaler = Path(staging_dir) / "scaler.joblib"
        joblib.dump(scaler, staged_scaler)
        candidates = [
            log_candidate(
                name,
                build_model(name, params, pos_weight),
                X_tr,
                y_tr,
                X_model_valid,
                y_model_valid,
                metadata,
                staged_scaler,
            )
            for name in params["train"]["models"]
        ]
        winner = choose_promoted_candidate(candidates)
        version = register_winner(winner)

    joblib.dump(winner["model"], _MODEL_PATH)
    features.save_scaler(scaler)
    comparison = {
        "selection_metric": "model_validation.pr_auc",
        "candidates": [
            {
                **{key: value for key, value in candidate.items() if key != "model"},
                "promoted": candidate["run_id"] == winner["run_id"],
            }
            for candidate in candidates
        ],
        "promoted_run_id": winner["run_id"],
    }
    promotion = {
        "promoted_model_id": f"{REGISTERED_MODEL}:v{version}",
        "model_name": winner["model_name"],
        "mlflow_run_id": winner["run_id"],
        "registered_model_name": REGISTERED_MODEL,
        "registered_model_version": version,
        "model_validation_metrics": winner["model_validation_metrics"],
        "source_provenance": provenance,
        "data_fingerprint": metadata["data_fingerprint"],
        "parameter_fingerprint": metadata["parameter_fingerprint"],
        "selection_reason": "highest model-validation PR-AUC",
    }
    write_json(REPORTS_DIR / "experiment_comparison.json", comparison)
    write_json(REPORTS_DIR / "promotion_record.json", promotion)
    print(
        f"promoted: {winner['model_name']} "
        f"(model-validation PR-AUC {winner['model_validation_metrics']['pr_auc']:.4f}) "
        f"-> {_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
