import numpy as np
import pandas as pd
import pytest

from src import ingest
from src.evidence import sha256_file
from src.ingest import SPLIT_NAMES, inject_drift, split_by_time
from src.validate import DataValidationError

PARAMS = {
    "data": {
        "train_frac": 0.5,
        "model_valid_frac": 0.1,
        "calibration_frac": 0.1,
        "n_prod_batches": 3,
    }
}

EXPECTED_SPLIT_NAMES = (
    "train",
    "model_valid",
    "calibration",
    "prod_1",
    "prod_2",
    "prod_3",
)


def _df(n=100):
    return pd.DataFrame(
        {
            "row_id": np.arange(n),
            "Time": np.repeat(np.arange((n + 1) // 2), 2)[:n],
            "Class": 0,
        }
    )


def test_split_has_six_chronological_disjoint_slices():
    parts = split_by_time(_df(100).sample(frac=1, random_state=4), PARAMS)
    assert SPLIT_NAMES == EXPECTED_SPLIT_NAMES
    assert tuple(parts) == SPLIT_NAMES
    assert [len(parts[name]) for name in SPLIT_NAMES] == [50, 10, 10, 10, 10, 10]
    joined = pd.concat(parts.values(), ignore_index=True)
    assert len(joined) == 100
    assert joined["row_id"].is_unique
    assert joined["Time"].is_monotonic_increasing


def test_split_preserves_remainder_rows():
    parts = split_by_time(_df(103), PARAMS)
    assert sum(map(len, parts.values())) == 103
    production_sizes = [len(parts[f"prod_{i}"]) for i in range(1, 4)]
    assert max(production_sizes) - min(production_sizes) <= 1


def test_inject_drift_shifts_amount_and_signal_features():
    rng = np.random.RandomState(0)
    n = 200
    data = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    data["Amount"] = np.abs(rng.normal(50, 10, n))
    data["Class"] = [1] * 5 + [0] * (n - 5)
    df = pd.DataFrame(data)
    params = {"data": {"drift_amount_scale": 3.0, "drift_feature_shift": 2.5}}

    out = inject_drift(df, params)
    # Amount is scaled up; a signal feature (V14) shifts; a non-signal one (V1) does not.
    assert out["Amount"].mean() > df["Amount"].mean() * 2
    assert out["V14"].mean() > df["V14"].mean() + 1.0
    assert np.allclose(out["V1"], df["V1"])
    # Fraud rows shift twice as far as legit on a signal feature (concept drift).
    fraud, legit = out["Class"] == 1, out["Class"] == 0
    assert out.loc[fraud, "V14"].mean() > out.loc[legit, "V14"].mean() + 1.0


def test_main_gates_the_exact_raw_path_and_split_params(tmp_path, monkeypatch):
    class GateObserved(Exception):
        pass

    params = {
        "data": {
            **PARAMS["data"],
            "raw_path": "data/raw/source.csv",
            "batch_dir": "data/batches",
        }
    }
    expected_path = tmp_path / "data/raw/source.csv"
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    monkeypatch.setattr(ingest, "load_params", lambda: params)

    def observe_gate(raw_path, supplied_params):
        assert raw_path == expected_path
        assert supplied_params is params
        raise GateObserved

    monkeypatch.setattr("src.validate.require_validation_gate", observe_gate)

    with pytest.raises(GateObserved):
        ingest.main()


def test_main_writes_no_batches_when_raw_changes_during_read(
    tmp_path, monkeypatch
):
    raw_path = tmp_path / "data/raw/source.csv"
    raw_path.parent.mkdir(parents=True)
    _df(100).to_csv(raw_path, index=False)
    expected_digest = sha256_file(raw_path)
    output_dir = tmp_path / "data/batches"
    params = {
        "data": {
            **PARAMS["data"],
            "raw_path": "data/raw/source.csv",
            "batch_dir": "data/batches",
        }
    }
    real_read_csv = pd.read_csv

    def read_then_mutate(path, *args, **kwargs):
        frame = real_read_csv(path, *args, **kwargs)
        if path == raw_path:
            mutated = frame.copy()
            mutated.loc[0, "row_id"] = 999
            mutated.to_csv(raw_path, index=False)
        return frame

    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    monkeypatch.setattr(ingest, "load_params", lambda: params)
    monkeypatch.setattr(ingest, "batch_dir", lambda supplied: output_dir)
    monkeypatch.setattr(
        "src.validate.require_validation_gate",
        lambda raw, supplied: {"raw_data_fingerprint": expected_digest},
    )
    monkeypatch.setattr("src.validate.pd.read_csv", read_then_mutate)

    with pytest.raises(DataValidationError, match="changed while being read"):
        ingest.main()

    assert not output_dir.exists()
