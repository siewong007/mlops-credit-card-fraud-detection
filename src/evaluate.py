"""Imbalance-aware evaluation (Owner: B). Requirements 3 (imbalance), SS11.

Never rely on accuracy: fraud rate is ~0.172%. ``compute_metrics`` is the
shared metric definition used by training, threshold selection, batch inference
and drift. ``main`` produces default-threshold comparison and calibrated
operating-point evidence for the promoted model on calibration data.
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(fn, y_true, proba) -> float | None:
    # ROC/PR-AUC are undefined when only one class is present in a batch.
    return float(fn(y_true, proba)) if len(np.unique(y_true)) > 1 else None


def compute_metrics(
    y_true,
    proba,
    threshold: float,
    *,
    cost_false_negative: float,
    cost_false_positive: float,
) -> dict:
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": _safe_auc(average_precision_score, y_true, proba),
        "roc_auc": _safe_auc(roc_auc_score, y_true, proba),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": float(threshold),
        "estimated_business_cost": float(cost_false_negative * fn + cost_false_positive * fp),
    }


def evaluate_probabilities(y_true, proba, operating_point: dict) -> tuple[dict, dict]:
    costs = operating_point["cost_assumptions"]
    common = {
        "cost_false_negative": costs["false_negative"],
        "cost_false_positive": costs["false_positive"],
    }
    default = compute_metrics(y_true, proba, 0.5, **common)
    operating = compute_metrics(y_true, proba, operating_point["threshold"], **common)
    evidence = {
        "promoted_model_id": operating_point["promoted_model_id"],
        "calibration_data_fingerprint": operating_point["calibration_data_fingerprint"],
    }
    default.update({"evidence_role": "comparison_default_threshold", **evidence})
    operating.update({"evidence_role": "primary_operating_point", **evidence})
    return default, operating


def _plot_confusion(y_true, pred, path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["legit", "fraud"],
           yticklabels=["legit", "fraud"], xlabel="predicted", ylabel="actual",
           title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pr_curve(y_true, proba, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prec, rec, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(rec, prec, label=f"AP = {ap:.3f}")
    ax.set(xlabel="recall", ylabel="precision", title="Precision-Recall curve (calibration)",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    import pandas as pd

    from src import features
    from src.artifacts import load_model, load_threshold
    from src.config import FIGURES_DIR, REPORTS_DIR, batch_dir, ensure_dirs, load_params
    from src.evidence import write_json

    ensure_dirs()
    params = load_params()
    calibration = pd.read_csv(batch_dir(params) / "calibration.csv")
    model, scaler = load_model(), features.load_scaler()
    X, y = features.xy(features.transform(calibration, scaler))
    proba = model.predict_proba(X)[:, 1]
    operating_point = load_threshold()
    default_metrics, operating_metrics = evaluate_probabilities(y, proba, operating_point)

    print(f"Calibration operating metrics (threshold={operating_point['threshold']:.2f}):")
    for k in ("pr_auc", "roc_auc", "precision", "recall", "f1",
              "false_positive_rate", "false_negative_rate"):
        print(f"  {k:>20}: {operating_metrics[k]:.4f}" if operating_metrics[k] is not None else f"  {k:>20}: undefined")
    print("  confusion tp/fp/fn/tn: "
          f"{operating_metrics['tp']}/{operating_metrics['fp']}/"
          f"{operating_metrics['fn']}/{operating_metrics['tn']}")

    write_json(REPORTS_DIR / "metrics_default.json", default_metrics)
    write_json(REPORTS_DIR / "metrics_operating.json", operating_metrics)
    _plot_confusion(
        y,
        (proba >= 0.5).astype(int),
        FIGURES_DIR / "confusion_matrix_default.png",
        "Confusion matrix — default threshold 0.5 (calibration)",
    )
    _plot_confusion(
        y,
        (proba >= operating_point["threshold"]).astype(int),
        FIGURES_DIR / "confusion_matrix_operating.png",
        f"Confusion matrix — operating threshold {operating_point['threshold']:.2f} (calibration)",
    )
    _plot_pr_curve(y, proba, FIGURES_DIR / "pr_curve.png")
    print(f"saved operating and default metrics + calibration figures to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
