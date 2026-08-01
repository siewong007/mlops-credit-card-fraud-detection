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
from src.validate import read_stable_csv, require_valid_dataframe


def score_batch(
    df: pd.DataFrame,
    model,
    scaler,
    *,
    threshold: float,
    promoted_model_id: str,
    batch_data_fingerprint: str,
) -> pd.DataFrame:
    contract = "labeled" if features.TARGET in df else "inference"
    require_valid_dataframe(
        df,
        contract=contract,
        require_target_distribution=False,
    )
    proba = model.predict_proba(features.transform(df, scaler)[features.FEATURES])[:, 1]
    out = df[["Time", "Amount"]].copy().reset_index(drop=True)
    out.insert(0, "row_position", range(len(out)))
    out["proba"] = proba
    out["pred"] = (proba >= threshold).astype(int)
    out["operating_threshold"] = float(threshold)
    out["promoted_model_id"] = promoted_model_id
    out["batch_data_fingerprint"] = batch_data_fingerprint
    if features.TARGET in df:
        out[features.TARGET] = df[features.TARGET].values
    return out


def main() -> None:
    params = load_params()
    bdir = batch_dir(params)
    model, scaler = load_model(), features.load_scaler()
    operating_point = load_threshold()
    threshold = operating_point["threshold"]
    promoted_model_id = operating_point["promoted_model_id"]

    for i in range(1, params["data"]["n_prod_batches"] + 1):
        name = f"prod_{i}"
        source_path = bdir / f"{name}.csv"
        df, batch_data_fingerprint = read_stable_csv(source_path)
        preds = score_batch(
            df,
            model,
            scaler,
            threshold=threshold,
            promoted_model_id=promoted_model_id,
            batch_data_fingerprint=batch_data_fingerprint,
        )
        preds.to_csv(bdir / f"preds_{name}.csv", index=False)
        flagged = int(preds["pred"].sum())
        print(f"{name}: {len(df):,} rows scored, {flagged} flagged as fraud "
              f"(threshold={threshold:.2f})")


if __name__ == "__main__":
    main()
