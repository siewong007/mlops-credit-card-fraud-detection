"""Threshold selection with business costs (Owner: B). Requirement 4.

FN (missed fraud) is far more costly than FP (false alarm); sweep thresholds
and pick the cost-minimising one on the calibration batch. The chosen threshold
becomes the operating point for batch inference and the recall baseline for the
retraining trigger. A trade-off figure (cost + precision/recall vs threshold)
is saved as evidence.
"""
import numpy as np
from sklearn.metrics import confusion_matrix

_GRID = np.linspace(0.01, 0.99, 99)


def expected_cost(y_true, proba, threshold: float, c_fn: float, c_fp: float) -> float:
    pred = (np.asarray(proba) >= threshold).astype(int)
    _, fp, fn, _ = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return c_fn * fn + c_fp * fp


def select_threshold(y_true, proba, c_fn: float, c_fp: float) -> tuple[float, float]:
    costs = [expected_cost(y_true, proba, t, c_fn, c_fp) for t in _GRID]
    i = int(np.argmin(costs))
    return float(_GRID[i]), float(costs[i])


def _plot_tradeoff(y_true, proba, c_fn, c_fp, chosen, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_score, recall_score

    costs = [expected_cost(y_true, proba, t, c_fn, c_fp) for t in _GRID]
    prec = [precision_score(y_true, (proba >= t).astype(int), zero_division=0) for t in _GRID]
    rec = [recall_score(y_true, (proba >= t).astype(int), zero_division=0) for t in _GRID]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.8))
    ax1.plot(_GRID, prec, color="tab:blue", label="precision")
    ax1.plot(_GRID, rec, color="tab:green", label="recall")
    ax1.set(xlabel="threshold", ylabel="precision / recall", ylim=(0, 1.02))
    ax2 = ax1.twinx()
    ax2.plot(_GRID, costs, color="tab:red", ls="--", label="expected cost")
    ax2.set_ylabel("expected cost", color="tab:red")
    ax1.axvline(chosen, color="black", ls=":", lw=1)
    ax1.set_title(f"Calibration threshold trade-off (chosen={chosen:.2f}, c_fn={c_fn}:c_fp={c_fp})")
    ax1.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_operating_point(
    y_true,
    proba,
    *,
    cost_false_negative: float,
    cost_false_positive: float,
    promotion_record: dict,
    calibration_data_fingerprint: str,
) -> dict:
    from src.evaluate import compute_metrics

    threshold, cost = select_threshold(
        y_true, proba, cost_false_negative, cost_false_positive
    )
    metrics = compute_metrics(
        y_true,
        proba,
        threshold,
        cost_false_negative=cost_false_negative,
        cost_false_positive=cost_false_positive,
    )
    return {
        "threshold": threshold,
        "cost_assumptions": {
            "false_negative": float(cost_false_negative),
            "false_positive": float(cost_false_positive),
        },
        "estimated_business_cost": cost,
        "calibration_metrics": metrics,
        "promoted_model_id": promotion_record["promoted_model_id"],
        "calibration_data_fingerprint": calibration_data_fingerprint,
    }


def main() -> None:
    from src import features
    from src.artifacts import load_model, save_operating_point
    from src.config import FIGURES_DIR, REPORTS_DIR, batch_dir, ensure_dirs, load_params
    from src.validate import read_stable_csv
    import json

    ensure_dirs()
    params = load_params()
    c_fn = params["threshold"]["cost_false_negative"]
    c_fp = params["threshold"]["cost_false_positive"]

    calibration_path = batch_dir(params) / "calibration.csv"
    calibration, calibration_data_fingerprint = read_stable_csv(calibration_path)
    model, scaler = load_model(), features.load_scaler()
    X, y = features.xy(features.transform(calibration, scaler))
    proba = model.predict_proba(X)[:, 1]

    operating_point = build_operating_point(
        y,
        proba,
        cost_false_negative=c_fn,
        cost_false_positive=c_fp,
        promotion_record=json.loads((REPORTS_DIR / "promotion_record.json").read_text()),
        calibration_data_fingerprint=calibration_data_fingerprint,
    )
    save_operating_point(operating_point)
    _plot_tradeoff(
        y, proba, c_fn, c_fp, operating_point["threshold"], FIGURES_DIR / "threshold_tradeoff.png"
    )

    print(f"selected calibration threshold={operating_point['threshold']:.2f} "
          f"(expected cost={operating_point['estimated_business_cost']:.0f}) "
          f"-> recall={operating_point['calibration_metrics']['recall']:.3f} "
          f"precision={operating_point['calibration_metrics']['precision']:.3f}")
    print(f"saved models/threshold.json + reports/operating_point.json + {FIGURES_DIR/'threshold_tradeoff.png'}")


if __name__ == "__main__":
    main()
