"""Imbalance-aware evaluation (Owner: B). Requirements 3 (imbalance), SS11.

Never rely on accuracy: fraud rate is ~0.172%.
"""
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import numpy as np


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "threshold": threshold,
    }


def main() -> None:
    # TODO(B): load registered model + valid batch, print full metric table,
    # save confusion matrix plot to reports/figures/
    raise SystemExit("TODO: implement standalone evaluation")


if __name__ == "__main__":
    main()
