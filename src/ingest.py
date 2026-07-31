"""Ingestion: load raw data and produce time-based batches (Owner: A).

Addresses requirement 1 (repeatable ingestion) and briefing SS10
(no random-only splitting).
"""
import pandas as pd

from src.config import ROOT, batch_dir, load_params

_SIGNAL_COMPONENTS = [4, 10, 11, 12, 14, 17]  # kept in sync with simulate_data

DEVELOPMENT_SPLIT_NAMES = ("train", "model_valid", "calibration")


def split_names(n_prod_batches: int) -> tuple[str, ...]:
    """Return the ordered slice names for ``n_prod_batches`` production batches."""
    return (
        *DEVELOPMENT_SPLIT_NAMES,
        *(f"prod_{index}" for index in range(1, n_prod_batches + 1)),
    )


# The default configuration; `split_names` is the source of truth for any other
# `n_prod_batches`, which params.yaml documents as configurable.
SPLIT_NAMES = split_names(3)


def split_by_time(df: pd.DataFrame, params: dict) -> dict[str, pd.DataFrame]:
    """Return stable time-ordered, disjoint slices that preserve every row."""
    ordered = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n_rows = len(ordered)
    data_params = params["data"]
    train_end = int(n_rows * data_params["train_frac"])
    model_valid_end = train_end + int(n_rows * data_params["model_valid_frac"])
    calibration_end = model_valid_end + int(n_rows * data_params["calibration_frac"])
    parts = {
        "train": ordered.iloc[:train_end],
        "model_valid": ordered.iloc[train_end:model_valid_end],
        "calibration": ordered.iloc[model_valid_end:calibration_end],
    }
    production = ordered.iloc[calibration_end:]
    n_batches = data_params["n_prod_batches"]
    if n_batches < 1:
        raise ValueError(f"n_prod_batches must be at least 1, got {n_batches}")
    for index in range(n_batches):
        start = index * len(production) // n_batches
        end = (index + 1) * len(production) // n_batches
        parts[f"prod_{index + 1}"] = production.iloc[start:end]
    expected = split_names(n_batches)
    if tuple(parts) != expected:
        raise ValueError(f"split produced {tuple(parts)}, expected {expected}")
    if sum(map(len, parts.values())) != n_rows:
        raise ValueError("time split did not preserve the complete dataset")
    return {name: frame.copy().reset_index(drop=True) for name, frame in parts.items()}


def inject_drift(batch: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Simulate a shifted production batch (clearly-labelled synthetic drift).

    Two effects, so the batch exercises every drift area in briefing SS12:

    * covariate/amount drift: scale Amount and shift the signal components for
      *all* rows -> PSI/KS flag many features as drifted.
    * concept drift: push the *fraud* rows an extra step so they overlap the
      legitimate cloud -> the model (trained on the old separation) can no
      longer rank or catch them, collapsing PR-AUC and recall.

    Used only to demonstrate the monitoring + retraining trigger; acknowledged
    in the report as not equivalent to real long-term production drift.
    """
    p = params["data"]
    shift = p["drift_feature_shift"]
    batch = batch.copy()
    fraud = batch["Class"] == 1
    batch["Amount"] = (batch["Amount"] * p["drift_amount_scale"]).clip(0, 25_691).round(2)
    for c in _SIGNAL_COMPONENTS:
        col = f"V{c}"
        batch[col] = batch[col] + shift            # covariate drift (all rows)
        batch.loc[fraud, col] = batch.loc[fraud, col] + shift  # concept drift (fraud only)
    return batch


def main() -> None:
    from src.validate import read_stable_csv, require_validation_gate

    params = load_params()
    raw_path = ROOT / params["data"]["raw_path"]
    report = require_validation_gate(raw_path, params)
    df, _ = read_stable_csv(
        raw_path,
        expected_sha256=report["raw_data_fingerprint"],
    )
    parts = split_by_time(df, params)

    inject = params["data"].get("inject_drift", False)
    drift_batch = params["data"].get("drift_batch")
    out_dir = batch_dir(params)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, part in parts.items():
        if inject and name == drift_batch:
            part = inject_drift(part, params)
            tag = "  <-- drift injected"
        else:
            tag = ""
        part.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows, fraud rate {part['Class'].mean():.5f}{tag}")


if __name__ == "__main__":
    main()
