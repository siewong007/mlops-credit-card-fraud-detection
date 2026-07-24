"""Ingestion: load raw data and produce time-based batches (Owner: A).

Addresses requirement 1 (repeatable ingestion) and briefing SS10
(no random-only splitting).
"""
import pandas as pd
from src.config import ROOT, load_params


def split_by_time(df: pd.DataFrame, params: dict) -> dict[str, pd.DataFrame]:
    """Sort by Time; return train / valid / prod_1..N slices."""
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


def main() -> None:
    params = load_params()
    df = pd.read_csv(ROOT / params["data"]["raw_path"])
    for name, part in split_by_time(df, params).items():
        part.to_csv(ROOT / params["data"]["batch_dir"] / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows, fraud rate {part['Class'].mean():.5f}")


if __name__ == "__main__":
    main()
