from pathlib import Path

import joblib
import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.train import (
    choose_promoted_candidate,
    log_candidate,
    register_winner,
)


def test_choose_promoted_candidate_uses_pr_auc():
    """Changing selection to any metric other than PR-AUC must fail here."""
    candidates = [
        {
            "model_name": "first",
            "model_validation_metrics": {"pr_auc": 0.71, "recall": 0.99},
        },
        {
            "model_name": "second",
            "model_validation_metrics": {"pr_auc": 0.82, "recall": 0.01},
        },
    ]

    assert choose_promoted_candidate(candidates)["model_name"] == "second"


def test_candidate_run_is_registered_without_a_second_run(tmp_path):
    """Registration must use the logged candidate model and preserve its run ID."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-fraud")
    X = np.array([[0.0], [0.2], [0.8], [1.0], [1.2], [1.4]])
    y = np.array([0, 0, 0, 1, 1, 1])
    scaler_path = tmp_path / "scaler.joblib"
    joblib.dump(StandardScaler().fit(X), scaler_path)
    model = LogisticRegression(random_state=42, class_weight="balanced")

    result = log_candidate(
        "logistic_regression",
        model,
        X,
        y,
        X,
        y,
        {
            "source": "unit-test",
            "data_fingerprint": "a" * 64,
            "parameter_fingerprint": "b" * 64,
            "imbalance": "balanced",
            "cost_false_negative": 100,
            "cost_false_positive": 1,
        },
        scaler_path,
    )
    version = register_winner(result, registered_model_name="test-fraud-detector")

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(result["run_id"])
    registered = client.get_model_version("test-fraud-detector", version)
    assert result["run_id"] == registered.run_id
    assert "estimator.C" in run.data.params
    assert run.data.params["imbalance"] == "balanced"
    assert run.data.params["cost_false_negative"] == "100"
    assert run.data.tags["source_provenance"] == "unit-test"
    assert run.data.tags["data_fingerprint"] == "a" * 64
    assert run.data.tags["parameter_fingerprint"] == "b" * 64
    assert all(
        np.isfinite(value)
        for value in result["model_validation_metrics"].values()
        if isinstance(value, (int, float))
    )
    assert any(file.path == "preprocessing/scaler.joblib" for file in client.list_artifacts(result["run_id"], "preprocessing"))
    assert client.get_model_version_by_alias(
        "test-fraud-detector", "champion"
    ).version == int(version)


def test_registry_failure_propagates(monkeypatch):
    """A registry outage must abort promotion rather than being silently ignored."""
    def fail_registration(model_uri, registered_model_name):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(mlflow, "register_model", fail_registration)

    with pytest.raises(RuntimeError, match="registry unavailable"):
        register_winner(
            {"model_uri": "runs:/missing/model"},
            registered_model_name="test-fraud-detector",
        )
