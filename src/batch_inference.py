"""Batch inference over simulated production batches (Owner: C). Requirement 1.

Scores prod_*.csv with the registered model at the selected threshold and
stores predictions for monitoring.
"""


def main() -> None:
    # TODO(C): load model from MLflow registry, validate each prod batch first
    # (src.validate), score, save predictions to data/batches/preds_prod_*.csv
    raise SystemExit("TODO: implement batch inference")


if __name__ == "__main__":
    main()
