"""Fetch the real ULB Credit Card Fraud dataset — no Kaggle account needed.

The canonical dataset lives on Kaggle (mlg-ulb/creditcardfraud) behind a login.
The identical data is mirrored on OpenML as dataset 1597 ("creditcard"), which
is openly downloadable. Note: OpenML flags ``Time`` as a row-identifier, so the
usual ``fetch_openml`` drops it; we download the raw parquet instead, which
keeps all 31 columns (Time, V1..V28, Amount, Class) in the canonical schema.

Usage:  python -m src.fetch_data   (or `make fetch-data`)
"""
import io
import urllib.request

import pandas as pd

from src.config import ROOT, load_params, write_provenance

PARQUET_URL = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def fetch() -> pd.DataFrame:
    with urllib.request.urlopen(PARQUET_URL, timeout=180) as r:  # noqa: S310 - fixed https URL
        raw = r.read()
    df = pd.read_parquet(io.BytesIO(raw))[COLS].copy()
    df["Class"] = df["Class"].astype(int)
    for c in COLS:
        if c != "Class":
            df[c] = df[c].astype(float)
    return df


def main() -> None:
    params = load_params()
    out = ROOT / params["data"]["raw_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df = fetch()
    df.to_csv(out, index=False)
    write_provenance(
        params, "REAL — ULB creditcard via OpenML dataset 1597",
        len(df), int(df["Class"].sum()),
    )
    print(
        f"[REAL] wrote {out} — {len(df):,} rows, {int(df['Class'].sum())} fraud "
        f"({df['Class'].mean():.6f}). Source: OpenML dataset 1597 (ULB creditcard)."
    )


if __name__ == "__main__":
    main()
