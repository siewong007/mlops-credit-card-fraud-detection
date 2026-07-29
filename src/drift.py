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
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src import features
from src.artifacts import load_model
from src.config import DRIFT_DIR, FIGURES_DIR, REPORTS_DIR, batch_dir, ensure_dirs, load_params
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


def _evidently_html(reference: pd.DataFrame, current: pd.DataFrame, path) -> bool:
    """Best-effort rich HTML report. Evidently's API changes across versions,
    so any failure is swallowed — the native JSON is the real evidence."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        cols = features.FEATURES + [features.TARGET]
        snapshot = Report([DataDriftPreset()]).run(
            reference_data=reference[cols], current_data=current[cols]
        )
        snapshot.save_html(str(path))
        return True
    except Exception:  # noqa: BLE001 - optional, never fatal
        return False


def main() -> None:
    ensure_dirs()
    params = load_params()
    bdir = batch_dir(params)
    psi_th = params["drift"]["psi_threshold"]

    from src.artifacts import load_threshold

    threshold = load_threshold()["threshold"]
    train = pd.read_csv(bdir / "train.csv")
    valid = pd.read_csv(bdir / "valid.csv")
    model, scaler = load_model(), features.load_scaler()
    ref_proba = model.predict_proba(features.transform(valid, scaler)[features.FEATURES])[:, 1]
    base_fraud_rate = float(train[features.TARGET].mean())

    summaries, evidently_ok = [], False
    for i in range(1, params["data"]["n_prod_batches"] + 1):
        name = f"prod_{i}"
        cur = pd.read_csv(bdir / f"{name}.csv")
        preds = pd.read_csv(bdir / f"preds_{name}.csv")

        fd = feature_drift(train, cur, params)
        perf = compute_metrics(
            preds[features.TARGET],
            preds["proba"],
            threshold=threshold,
            cost_false_negative=params["threshold"]["cost_false_negative"],
            cost_false_positive=params["threshold"]["cost_false_positive"],
        )
        summary = {
            "batch": name,
            "n_rows": len(cur),
            "pct_drifted_features": fd["pct_drifted_features"],
            "n_drifted": fd["n_drifted"],
            "amount_psi": fd["per_feature"]["Amount"]["psi"],
            "prediction_psi": round(psi(ref_proba, preds["proba"].to_numpy()), 4),
            "fraud_rate": round(float(cur[features.TARGET].mean()), 5),
            "baseline_fraud_rate": round(base_fraud_rate, 5),
            "pr_auc": perf["pr_auc"],
            "recall": perf["recall"],
            "precision": perf["precision"],
        }
        (DRIFT_DIR / f"{name}.json").write_text(json.dumps({**summary, "per_feature": fd["per_feature"]}, indent=2))
        _plot_psi(fd["per_feature"], name, psi_th, FIGURES_DIR / f"psi_{name}.png")
        evidently_ok |= _evidently_html(train, cur, DRIFT_DIR / f"{name}.html")
        summaries.append(summary)
        print(f"{name}: {fd['pct_drifted_features']:.0f}% features drifted, "
              f"amount_psi={summary['amount_psi']:.2f}, pred_psi={summary['prediction_psi']:.2f}, "
              f"PR-AUC={perf['pr_auc']:.3f}, recall={perf['recall']:.3f}")

    (REPORTS_DIR / "drift_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"saved per-batch drift to {DRIFT_DIR} and reports/drift_summary.json"
          + ("" if evidently_ok else "  (Evidently HTML skipped — native drift only)"))


if __name__ == "__main__":
    main()
