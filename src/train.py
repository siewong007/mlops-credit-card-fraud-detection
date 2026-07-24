"""Training + MLflow experiment tracking (Owner: B). Requirements 3, 7.

Minimum two runs (briefing SS9.2): run once with model=logistic_regression,
once with model=xgboost (edit params.yaml or pass overrides).
"""
import mlflow
from sklearn.linear_model import LogisticRegression
from src import features
from src.config import ROOT, load_params
from src.evaluate import compute_metrics
import pandas as pd

EXPERIMENT = "fraud-detection"


def build_model(params: dict):
    if params["train"]["model"] == "logistic_regression":
        return LogisticRegression(
            class_weight=params["train"]["class_weight"],
            max_iter=1000,
            random_state=params["train"]["random_state"],
        )
    if params["train"]["model"] == "xgboost":
        from xgboost import XGBClassifier
        # TODO(B): set scale_pos_weight from train fraud rate
        return XGBClassifier(random_state=params["train"]["random_state"])
    raise ValueError(f"unknown model {params['train']['model']}")


def main() -> None:
    params = load_params()
    batch_dir = ROOT / params["data"]["batch_dir"]
    train = pd.read_csv(batch_dir / "train.csv")
    valid = pd.read_csv(batch_dir / "valid.csv")

    scaler = features.fit_scaler(train)
    X_tr, y_tr = features.xy(features.transform(train, scaler))
    X_va, y_va = features.xy(features.transform(valid, scaler))

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=params["train"]["model"]):
        mlflow.log_params(params["train"])
        model = build_model(params).fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        metrics = compute_metrics(y_va, proba, threshold=0.5)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        # TODO(B): register best model in MLflow Model Registry
        print(metrics)


if __name__ == "__main__":
    main()
