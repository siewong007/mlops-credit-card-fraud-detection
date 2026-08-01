import json

import pytest

from src.simulate_data import _SYNTHETIC_SOURCE, refuse_to_clobber_real_data


def _params(tmp_path):
    # An absolute raw_path resolves independently of the repository ROOT.
    return {"data": {"raw_path": str(tmp_path / "raw" / "creditcard.csv")}}


def _write_raw(tmp_path, provenance_source):
    """Place a raw CSV, optionally alongside a provenance record."""
    raw = tmp_path / "raw" / "creditcard.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("Time,Amount,Class\n0,1.0,0\n", encoding="utf-8")
    if provenance_source is not None:
        (raw.parent / "PROVENANCE.json").write_text(
            json.dumps({"source": provenance_source, "rows": 1, "fraud": 0}),
            encoding="utf-8",
        )
    return raw


def test_generation_is_allowed_when_no_raw_data_exists(tmp_path):
    refuse_to_clobber_real_data(_params(tmp_path))


def test_generation_may_replace_its_own_synthetic_output(tmp_path):
    """`dvc repro` must still be able to regenerate a synthetic dataset."""
    _write_raw(tmp_path, _SYNTHETIC_SOURCE)

    refuse_to_clobber_real_data(_params(tmp_path))


@pytest.mark.parametrize(
    "provenance_source",
    [
        "REAL — ULB creditcard via OpenML dataset 1597",
        None,  # a Kaggle CSV dropped in by hand carries no provenance
    ],
)
def test_generation_refuses_to_overwrite_unrecoverable_real_data(
    tmp_path, provenance_source
):
    """Catches `dvc repro` silently destroying the caller's real dataset."""
    raw = _write_raw(tmp_path, provenance_source)
    before = raw.read_bytes()

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        refuse_to_clobber_real_data(_params(tmp_path))
    assert raw.read_bytes() == before
