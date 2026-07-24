"""Ingestion: load raw data and produce time-based batches (Owner: A).

Addresses requirement 1 (repeatable ingestion) and briefing SS10
(no random-only splitting).
"""
import pandas as pd

from src.config import ROOT, batch_dir, load_params

_SIGNAL_COMPONENTS = [4, 10, 11, 12, 14, 17]  # kept in sync with simulate_data


def split_by_time(df: pd.DataFrame, params: dict) -> dict[str, pd.DataFrame]:
    """Sort by Time; return train / valid / prod_1..N slices (disjoint, ordered)."""
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    p = params["data"]
    t_end = int(n * p["train_frac"])
    v_end = t_end + int(n * p["valid_frac"])
    out = {"train": df.iloc[:t_end], "valid": df.iloc[t_end:v_end]}
    prod = df.iloc[v_end:]
    k = p["n_prod_batches"]
    for i in range(k):
        out[f"prod_{i + 1}"] = prod.iloc[i * len(prod) // k : (i + 1) * len(prod) // k]
    return out


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
    params = load_params()
    df = pd.read_csv(ROOT / params["data"]["raw_path"])
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
