import json

import pytest

from src import config
from src.evidence import sha256_file, write_json


def test_sha256_file_is_stable(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"fraud-mlops\n")
    assert sha256_file(source) == (
        "bd69cbe6d0148a4dbec170d60a316d5dbad2da4bf8b0584f1dcb095e3d2bdd33"
    )


def test_write_json_is_sorted_and_rejects_nan(tmp_path):
    target = tmp_path / "nested" / "evidence.json"
    assert write_json(target, {"z": 2, "a": 1}) == target
    assert target.read_text() == '{\n  "a": 1,\n  "z": 2\n}\n'
    assert json.loads(target.read_text()) == {"a": 1, "z": 2}
    with pytest.raises(ValueError):
        write_json(target, {"metric": float("nan")})


def test_provenance_writer_rejects_non_strict_json(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    params = {"data": {"raw_path": "data/raw/creditcard.csv"}}

    with pytest.raises(ValueError):
        config.write_provenance(params, float("nan"), rows=1, fraud=0)

    assert not config.provenance_path(params).exists()
