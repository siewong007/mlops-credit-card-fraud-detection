"""Stable, portable evidence helpers shared by pipeline stages."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
import re
import subprocess
from importlib import metadata

import pandas as pd

from src.config import REPORTS_DIR, ROOT, load_params, read_provenance


PARAMS_PATH = ROOT / "params.yaml"
RAW_PATH = ROOT / load_params()["data"]["raw_path"]
_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict | list) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.write_text(text, encoding="utf-8")
    return target


def load_exact_requirements(path: Path) -> dict[str, str]:
    """Load direct dependencies, rejecting every line that is not ``name==version``."""
    requirements = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN.fullmatch(line)
        if not match:
            raise ValueError(f"requirement is not an exact == pin: {line}")
        requirements[match.group(1)] = match.group(2)
    return requirements


def installed_direct_dependencies(path: Path) -> dict[str, str]:
    """Return installed direct pins, failing if any version differs from the file."""
    pins = load_exact_requirements(path)
    installed = {}
    for name in pins:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = "not installed"
    mismatches = {
        name: {"required": pins[name], "installed": installed[name]}
        for name in pins
        if installed[name] != pins[name]
    }
    if mismatches:
        raise RuntimeError(f"direct dependency mismatch: {mismatches}")
    return installed


def _source_commit(explicit: str | None) -> str:
    if explicit is not None:
        commit = explicit
    else:
        # `ARG SOURCE_COMMIT` with no --build-arg bakes in an empty string, which
        # is unset rather than invalid; an explicit "" is still an error.
        commit = os.environ.get("SOURCE_COMMIT", "").strip() or _head_commit()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source commit must be a full 40-character SHA")
    return commit


def _head_commit() -> str:
    """Read HEAD, explaining the fix when no repository is reachable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "cannot determine the source commit: no git repository is reachable "
            "(the Docker image excludes .git). Set the SOURCE_COMMIT environment "
            "variable, or build with --build-arg SOURCE_COMMIT=$(git rev-parse HEAD)."
        ) from error
    return result.stdout.strip()


def _data_evidence(params: dict) -> dict:
    """Describe the raw dataset, counting it when no sidecar record exists.

    ``fetch_data`` and ``simulate_data`` write PROVENANCE.json, but a Kaggle CSV
    dropped into ``data/raw/`` by hand (a documented setup path) carries none,
    and ``read_provenance`` then reports only ``source: unknown``. Row and fraud
    counts are objective properties of the file, so derive them rather than
    failing the run at its last stage; only ``source`` is genuinely unknown.
    """
    provenance = read_provenance(params)
    counted = {"rows", "fraud", "fraud_rate"}.issubset(provenance)
    if counted:
        rows = int(provenance["rows"])
        fraud = int(provenance["fraud"])
        fraud_rate = float(provenance["fraud_rate"])
    else:
        target = pd.read_csv(RAW_PATH, usecols=["Class"])["Class"]
        rows = len(target)
        fraud = int((target == 1).sum())
        fraud_rate = round(fraud / rows, 6) if rows else 0.0
    return {
        "source": provenance.get("source", "unknown"),
        "rows": rows,
        "fraud": fraud,
        "fraud_rate": fraud_rate,
        "fingerprint_sha256": sha256_file(RAW_PATH),
    }


def build_run_manifest(source_commit: str | None = None) -> dict:
    """Build deterministic evidence that identifies the exact pipeline inputs."""
    params = load_params()
    return {
        "schema_version": 1,
        "source_commit": _source_commit(source_commit),
        "python_version": platform.python_version(),
        "dependencies": installed_direct_dependencies(ROOT / "requirements.txt"),
        "data": _data_evidence(params),
        "parameters": {
            "path": "params.yaml",
            "fingerprint_sha256": sha256_file(PARAMS_PATH),
        },
    }


def write_run_manifest(path: Path = REPORTS_DIR / "run_manifest.json") -> Path:
    """Write the deterministic run manifest using the strict JSON serializer."""
    return write_json(path, build_run_manifest())


if __name__ == "__main__":
    output = write_run_manifest()
    print(f"wrote {output}")
