"""Threshold selection with business costs (Owner: B). Requirement 4.

FN (missed fraud) is far more costly than FP (false alarm); sweep thresholds
and pick the cost-minimising one. Report the trade-off curve.
"""
import numpy as np
from sklearn.metrics import confusion_matrix


def expected_cost(y_true, proba, threshold: float, c_fn: float, c_fp: float) -> float:
    pred = (np.asarray(proba) >= threshold).astype(int)
    _, fp, fn, _ = confusion_matrix(y_true, pred).ravel()
    return c_fn * fn + c_fp * fp


def select_threshold(y_true, proba, c_fn: float, c_fp: float) -> tuple[float, float]:
    grid = np.linspace(0.01, 0.99, 99)
    costs = [expected_cost(y_true, proba, t, c_fn, c_fp) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def main() -> None:
    # TODO(B): load valid predictions, run sweep with params.yaml costs,
    # log chosen threshold to MLflow, save cost curve to reports/figures/
    raise SystemExit("TODO: implement threshold analysis")


if __name__ == "__main__":
    main()
