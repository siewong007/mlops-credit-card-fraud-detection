"""Imbalance-aware evaluation (Owner: B). Requirements 3 (imbalance), SS11.

Never rely on accuracy: fraud rate is ~0.172%. ``compute_metrics`` is the
shared metric definition used by training, threshold selection, batch inference
and drift. ``main`` produces the human-readable evidence: a metric table plus
confusion-matrix and precision-recall-curve figures for the promoted model on
the validation batch.
"""
import json

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


def _safe_auc(fn, y_true, proba) -> float:
    # ROC/PR-AUC are undefined when only one class is present in a batch.
    return float(fn(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan")


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
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
    }


def _plot_confusion(y_true, pred, path) -> None:
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
           title="Confusion matrix (valid)")
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
    ax.set(xlabel="recall", ylabel="precision", title="Precision-Recall curve (valid)",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    import pandas as pd

    from src import features
    from src.artifacts import load_model
    from src.config import FIGURES_DIR, REPORTS_DIR, batch_dir, ensure_dirs, load_params

    ensure_dirs()
    params = load_params()
    valid = pd.read_csv(batch_dir(params) / "valid.csv")
    model, scaler = load_model(), features.load_scaler()
    X, y = features.xy(features.transform(valid, scaler))
    proba = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y, proba, threshold=0.5)

    print("Validation metrics (threshold=0.5):")
    for k in ("pr_auc", "roc_auc", "precision", "recall", "f1",
              "false_positive_rate", "false_negative_rate"):
        print(f"  {k:>20}: {metrics[k]:.4f}")
    print(f"  confusion tp/fp/fn/tn: {metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']}")

    (REPORTS_DIR / "metrics_valid.json").write_text(json.dumps(metrics, indent=2))
    _plot_confusion(y, (proba >= 0.5).astype(int), FIGURES_DIR / "confusion_matrix.png")
    _plot_pr_curve(y, proba, FIGURES_DIR / "pr_curve.png")
    print(f"saved metrics_valid.json + figures to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
