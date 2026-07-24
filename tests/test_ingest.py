import pandas as pd
from src.ingest import split_by_time

PARAMS = {"data": {"train_frac": 0.5, "valid_frac": 0.2, "n_prod_batches": 3}}


def _df(n=100):
    return pd.DataFrame({"Time": range(n), "Class": [0] * n})


def test_split_is_time_ordered_and_disjoint():
    parts = split_by_time(_df(), PARAMS)
    assert set(parts) == {"train", "valid", "prod_1", "prod_2", "prod_3"}
    assert sum(len(p) for p in parts.values()) == 100
    assert parts["train"]["Time"].max() < parts["valid"]["Time"].min()
    assert parts["valid"]["Time"].max() < parts["prod_1"]["Time"].min()
