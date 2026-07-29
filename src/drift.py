"""Drift + performance monitoring (Owner: C). Requirement 2, SS12.

Each production batch is compared against the training reference across the
drift areas in briefing SS12:

* feature drift      -- KS test + Population Stability Index per feature
* amount drift       -- PSI on Amount (a specific, interpretable feature)
* prediction drift   -- PSI between reference vs batch predicted probabilities
* target drift       -- fraud-rate shift vs the training reference
* performance drift  -- PR-AUC / recall on the batch (labels available here)

Drift is computed natively (scipy/numpy) so the monitoring signal that drives
the retraining trigger has no heavy dependency. A rich Evidently HTML report is
additionally emitted per batch when Evidently is importable (best-effort).
The per-batch summary JSON is the contract consumed by ``src.retrain_trigger``.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src import features
from src.artifacts import load_model
from src.config import DRIFT_DIR, FIGURES_DIR, REPORTS_DIR, batch_dir, ensure_dirs, load_params
from src.evidence import sha256_file, write_json
from src.evaluate import compute_metrics


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using quantile bins of the reference."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature -> no meaningful PSI
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    eps = 1e-6
    ref_pct, cur_pct = np.clip(ref_pct, eps, None), np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, params: dict) -> dict:
    """Per-feature PSI + KS. A feature is flagged 'drifted' on **PSI magnitude**
    (PSI > threshold). The KS p-value is reported alongside for information but
    is deliberately NOT used as the binary flag: at production batch sizes
    (tens of thousands of rows) KS becomes significant for negligibly small
    differences, flagging every feature. PSI is magnitude-based and stable
    across sample sizes, so it is the robust drift signal."""
    psi_th = params["drift"]["psi_threshold"]
    detail, drifted = {}, 0
    for f in features.FEATURES:
        p = psi(reference[f].to_numpy(), current[f].to_numpy())
        ks_p = float(ks_2samp(reference[f], current[f]).pvalue)
        is_drift = bool(p > psi_th)
        drifted += is_drift
        detail[f] = {"psi": round(p, 4), "ks_pvalue": round(ks_p, 4), "drifted": is_drift}
    return {
        "per_feature": detail,
        "n_features": len(features.FEATURES),
        "n_drifted": drifted,
        "pct_drifted_features": round(100 * drifted / len(features.FEATURES), 1),
    }


def _plot_psi(detail: dict, batch: str, psi_th: float, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(detail)
    vals = [detail[n]["psi"] for n in names]
    colors = ["tab:red" if v > psi_th else "tab:blue" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(names, vals, color=colors)
    ax.axhline(psi_th, color="black", ls=":", lw=1, label=f"PSI threshold {psi_th}")
    ax.set(title=f"Feature PSI vs training reference — {batch}", ylabel="PSI")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize_batch(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_proba: np.ndarray,
    predictions: pd.DataFrame,
    *,
    batch_name: str,
    threshold: float,
    promoted_model_id: str,
    batch_data_fingerprint: str,
    params: dict,
) -> dict:
    """Return native drift and, where available, label-based batch evidence."""
    probability = _validate_prediction_contract(
        current,
        predictions,
        threshold=threshold,
        promoted_model_id=promoted_model_id,
        batch_data_fingerprint=batch_data_fingerprint,
    )
    drift = feature_drift(reference, current, params)
    labels_available = features.TARGET in predictions
    performance = None
    if labels_available:
        costs = params["threshold"]
        performance = compute_metrics(
            predictions[features.TARGET],
            probability,
            threshold,
            cost_false_negative=costs["cost_false_negative"],
            cost_false_positive=costs["cost_false_positive"],
        )
    target_rate = float(predictions[features.TARGET].mean()) if labels_available else None
    baseline_target_rate = float(reference[features.TARGET].mean())
    return {
        "batch": batch_name,
        "n_rows": len(current),
        "label_status": "available" if labels_available else "pending",
        "promoted_model_id": promoted_model_id,
        "operating_threshold": float(threshold),
        "batch_data_fingerprint": batch_data_fingerprint,
        "pct_drifted_features": drift["pct_drifted_features"],
        "n_drifted": drift["n_drifted"],
        "amount_psi": drift["per_feature"]["Amount"]["psi"],
        "prediction_psi": round(psi(reference_proba, probability), 4),
        "target_rate": target_rate,
        "baseline_target_rate": baseline_target_rate if labels_available else None,
        "target_rate_delta": target_rate - baseline_target_rate if labels_available else None,
        "pr_auc": performance["pr_auc"] if performance else None,
        "precision": performance["precision"] if performance else None,
        "recall": performance["recall"] if performance else None,
        "estimated_business_cost": (
            performance["estimated_business_cost"] if performance else None
        ),
        "per_feature": drift["per_feature"],
    }


def _validate_prediction_contract(
    current: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    threshold: float,
    promoted_model_id: str,
    batch_data_fingerprint: str,
) -> np.ndarray:
    if len(predictions) != len(current):
        raise ValueError(
            "prediction row count does not match the current batch row count"
        )
    required = {
        "row_position",
        "Time",
        "Amount",
        "proba",
        "pred",
        "operating_threshold",
        "promoted_model_id",
        "batch_data_fingerprint",
    }
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"prediction evidence is missing columns: {sorted(missing)}")
    if not np.array_equal(
        predictions["row_position"].to_numpy(), np.arange(len(current))
    ):
        raise ValueError("prediction row alignment does not match the current batch")
    aligned_columns = ["Time", "Amount"]
    current_labeled = features.TARGET in current
    predictions_labeled = features.TARGET in predictions
    if current_labeled != predictions_labeled:
        raise ValueError("prediction row alignment does not match current Class labels")
    if current_labeled:
        aligned_columns.append(features.TARGET)
    if any(
        not np.array_equal(
            current[column].to_numpy(), predictions[column].to_numpy()
        )
        for column in aligned_columns
    ):
        raise ValueError("prediction row alignment does not match the current batch")
    if not bool(
        len(predictions)
        and predictions["operating_threshold"].eq(float(threshold)).all()
    ):
        raise ValueError("prediction operating threshold does not match")
    if not bool(
        len(predictions)
        and predictions["promoted_model_id"].eq(promoted_model_id).all()
    ):
        raise ValueError("prediction promoted model does not match")
    if not bool(
        len(predictions)
        and predictions["batch_data_fingerprint"].eq(
            batch_data_fingerprint
        ).all()
    ):
        raise ValueError("prediction batch data fingerprint does not match")
    probability = pd.to_numeric(
        predictions["proba"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError("prediction evidence must contain finite probabilities")
    prediction = pd.to_numeric(
        predictions["pred"], errors="coerce"
    ).to_numpy(dtype=float)
    if not set(prediction).issubset({0.0, 1.0}):
        raise ValueError("prediction evidence must contain binary predictions")
    return probability


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, path) -> dict:
    """Emit optional Evidently HTML without weakening native monitoring."""
    target = Path(path)
    try:
        target.unlink(missing_ok=True)
        from evidently import Report
        from evidently.presets import DataDriftPreset

        columns = features.FEATURES.copy()
        if features.TARGET in reference and features.TARGET in current:
            columns.append(features.TARGET)
        snapshot = Report([DataDriftPreset()]).run(
            reference_data=reference[columns], current_data=current[columns]
        )
        snapshot.save_html(str(target))
        return {"status": "generated", "error_type": None, "message": None}
    except Exception as error:  # noqa: BLE001 - optional, never fatal
        target.unlink(missing_ok=True)
        return {
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
        }


def main() -> None:
    ensure_dirs()
    params = load_params()
    bdir = batch_dir(params)
    psi_th = params["drift"]["psi_threshold"]

    from src.artifacts import load_threshold

    operating_point = load_threshold()
    threshold = operating_point["threshold"]
    promoted_model_id = operating_point["promoted_model_id"]
    train = pd.read_csv(bdir / "train.csv")
    calibration = pd.read_csv(bdir / "calibration.csv")
    model, scaler = load_model(), features.load_scaler()
    ref_proba = model.predict_proba(
        features.transform(calibration, scaler)[features.FEATURES]
    )[:, 1]

    summaries = []
    for i in range(1, params["data"]["n_prod_batches"] + 1):
        name = f"prod_{i}"
        source_path = bdir / f"{name}.csv"
        batch_data_fingerprint = sha256_file(source_path)
        cur = pd.read_csv(source_path)
        if sha256_file(source_path) != batch_data_fingerprint:
            raise ValueError(f"{source_path} changed while being read")
        preds = pd.read_csv(bdir / f"preds_{name}.csv")

        full_summary = summarize_batch(
            train,
            cur,
            ref_proba,
            preds,
            batch_name=name,
            threshold=threshold,
            promoted_model_id=promoted_model_id,
            batch_data_fingerprint=batch_data_fingerprint,
            params=params,
        )
        full_summary["evidently"] = evidently_report(train, cur, DRIFT_DIR / f"{name}.html")
        write_json(DRIFT_DIR / f"{name}.json", full_summary)
        _plot_psi(full_summary["per_feature"], name, psi_th, FIGURES_DIR / f"psi_{name}.png")
        summaries.append({key: value for key, value in full_summary.items() if key != "per_feature"})
        print(
            f"{name}: {full_summary['pct_drifted_features']:.0f}% features drifted, "
            f"amount_psi={full_summary['amount_psi']:.2f}, "
            f"pred_psi={full_summary['prediction_psi']:.2f}, "
            f"labels={full_summary['label_status']}"
        )

    write_json(REPORTS_DIR / "drift_summary.json", summaries)
    print(f"saved per-batch drift to {DRIFT_DIR} and reports/drift_summary.json")


if __name__ == "__main__":
    main()
