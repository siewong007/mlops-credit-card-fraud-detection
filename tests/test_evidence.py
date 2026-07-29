import importlib.metadata
import json
import platform

import pytest

from src import config
from src.evidence import (
    _source_commit,
    build_run_manifest,
    installed_direct_dependencies,
    load_exact_requirements,
    sha256_file,
    write_run_manifest,
    write_json,
)


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


def test_synthetic_provenance_omits_volatile_generation_timestamp(tmp_path, monkeypatch):
    """Synthetic lineage must be byte-stable for the same generated data."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    params = {"data": {"raw_path": "data/raw/creditcard.csv"}}

    config.write_provenance(
        params,
        "SYNTHETIC — unit test",
        rows=10,
        fraud=1,
        include_generated_at=False,
    )
    first = config.provenance_path(params).read_text()
    config.write_provenance(
        params,
        "SYNTHETIC — unit test",
        rows=10,
        fraud=1,
        include_generated_at=False,
    )

    assert config.provenance_path(params).read_text() == first
    assert "generated_at" not in json.loads(first)


def test_exact_requirement_parser_rejects_ranges_and_unpinned_lines(tmp_path):
    """Catches accepting a direct dependency without one exact version."""
    exact = tmp_path / "exact.txt"
    exact.write_text("pandas==2.3.3\n# comment\npyyaml==6.0.3\n")
    assert load_exact_requirements(exact) == {
        "pandas": "2.3.3",
        "pyyaml": "6.0.3",
    }

    for invalid in ("dvc>=3.50,<4\n", "pyarrow\n"):
        requirements = tmp_path / "invalid.txt"
        requirements.write_text(invalid)
        with pytest.raises(ValueError, match="exact == pin"):
            load_exact_requirements(requirements)


def test_installed_direct_dependencies_reports_actionable_mismatches(
    tmp_path, monkeypatch
):
    """Catches silently recording a version other than the declared direct pin."""
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("pandas==2.3.3\n")
    monkeypatch.setattr("src.evidence.metadata.version", lambda _: "2.3.2")

    with pytest.raises(RuntimeError, match="pandas") as error:
        installed_direct_dependencies(requirements)

    assert "required" in str(error.value)
    assert "installed" in str(error.value)


def test_installed_direct_dependencies_reports_missing_distribution(
    tmp_path, monkeypatch
):
    """Catches an uninstalled direct dependency escaping as an opaque metadata error."""
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("pandas==2.3.3\n")

    def missing(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError("pandas")

    monkeypatch.setattr("src.evidence.metadata.version", missing)

    with pytest.raises(RuntimeError, match="pandas") as error:
        installed_direct_dependencies(requirements)

    assert "required" in str(error.value)
    assert "not installed" in str(error.value)


def test_manifest_contains_only_invariant_reproducibility_fields(
    tmp_path, monkeypatch
):
    """Catches volatile or incomplete evidence in the reproducibility manifest."""
    raw = tmp_path / "creditcard.csv"
    raw.write_text("Time,V1,Amount,Class\n0,0,1,0\n")
    params = tmp_path / "params.yaml"
    params.write_text("data: {}\n")
    provenance = {
        "source": "REAL — unit test",
        "rows": 1,
        "fraud": 0,
        "fraud_rate": 0.0,
    }
    monkeypatch.setattr("src.evidence.RAW_PATH", raw)
    monkeypatch.setattr("src.evidence.PARAMS_PATH", params)
    monkeypatch.setattr("src.evidence.load_params", lambda: {"data": {}})
    monkeypatch.setattr("src.evidence.read_provenance", lambda _: provenance)
    monkeypatch.setattr(
        "src.evidence.installed_direct_dependencies",
        lambda _: {"pandas": "2.3.3"},
    )
    monkeypatch.setenv("SOURCE_COMMIT", "b" * 40)

    manifest = build_run_manifest(source_commit="a" * 40)

    assert manifest == {
        "schema_version": 1,
        "source_commit": "a" * 40,
        "python_version": platform.python_version(),
        "dependencies": {"pandas": "2.3.3"},
        "data": {
            "source": "REAL — unit test",
            "rows": 1,
            "fraud": 0,
            "fraud_rate": 0.0,
            "fingerprint_sha256": sha256_file(raw),
        },
        "parameters": {
            "path": "params.yaml",
            "fingerprint_sha256": sha256_file(params),
        },
    }


def test_explicit_empty_source_commit_does_not_defer_to_environment(monkeypatch):
    """Catches an explicit invalid source SHA being silently replaced by the env value."""
    monkeypatch.setenv("SOURCE_COMMIT", "b" * 40)

    with pytest.raises(ValueError, match="full 40-character SHA"):
        _source_commit("")


def test_write_run_manifest_writes_strict_json_to_requested_path(tmp_path, monkeypatch):
    """Catches the writer returning a manifest without persisting it as strict JSON."""
    target = tmp_path / "reports" / "run_manifest.json"
    payload = {"schema_version": 1, "data": {"fraud_rate": 0.0}}
    monkeypatch.setattr("src.evidence.build_run_manifest", lambda: payload)

    assert write_run_manifest(target) == target
    assert json.loads(target.read_text()) == payload
