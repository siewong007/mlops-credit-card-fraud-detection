"""Batch inference over simulated production batches (Owner: C). Requirement 1.

For each prod_*.csv: validate it (schema/ranges — requirement 6, fail fast on
bad data), score it with the promoted model at the selected operating
threshold, and persist predictions (proba + label, plus the true Class where
available) for the drift/performance monitoring stage.
"""
import pandas as pd

from src import features
from src.artifacts import load_model, load_threshold
from src.config import batch_dir, load_params
from src.validate import schema


def score_batch(df: pd.DataFrame, model, scaler, threshold: float) -> pd.DataFrame:
    proba = model.predict_proba(features.transform(df, scaler)[features.FEATURES])[:, 1]
    out = df[["Time", "Amount"]].copy()
    out["proba"] = proba
    out["pred"] = (proba >= threshold).astype(int)
    if features.TARGET in df:
        out[features.TARGET] = df[features.TARGET].values
    return out


def main() -> None:
    params = load_params()
    bdir = batch_dir(params)
    model, scaler = load_model(), features.load_scaler()
    threshold = load_threshold()["threshold"]

    for i in range(1, params["data"]["n_prod_batches"] + 1):
        name = f"prod_{i}"
        df = pd.read_csv(bdir / f"{name}.csv")
        schema.validate(df)  # requirement 6: reject malformed batches before scoring
        preds = score_batch(df, model, scaler, threshold)
        preds.to_csv(bdir / f"preds_{name}.csv", index=False)
        flagged = int(preds["pred"].sum())
        print(f"{name}: {len(df):,} rows scored, {flagged} flagged as fraud "
              f"(threshold={threshold:.2f})")


if __name__ == "__main__":
    main()
