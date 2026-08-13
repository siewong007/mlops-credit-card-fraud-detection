"""Fail unless a raw dataset and its provenance are explicitly real."""

import hashlib
import json
import sys
from pathlib import Path

REAL_SOURCE = "REAL — ULB creditcard via OpenML dataset 1597"
REAL_ROWS = 284_807
REAL_FRAUD = 492
REAL_CSV_SHA256 = "1700322b377ac8340ab6b75f22b974944fbcaf5b4f0daf3d5e8f620d96313026"


def require_real_data(
    raw: Path,
    provenance: Path,
    *,
    expected_sha256: str = REAL_CSV_SHA256,
) -> None:
    if not raw.is_file():
        raise SystemExit(f"real dataset is missing: {raw}")
    try:
        payload = json.loads(provenance.read_text())
        source = payload["source"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"real dataset requires valid provenance: {provenance}"
        ) from exc
    if source != REAL_SOURCE:
        raise SystemExit(
            f"dataset provenance is not OpenML dataset 1597: {source!r}"
        )
    if payload.get("rows") != REAL_ROWS or payload.get("fraud") != REAL_FRAUD:
        raise SystemExit(
            "dataset provenance counts do not match ULB dataset 1597: "
            f"rows={payload.get('rows')!r}, fraud={payload.get('fraud')!r}"
        )
    with raw.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "dataset fingerprint does not match ULB dataset 1597: "
            f"{actual_sha256}"
        )


def main() -> None:
    require_real_data(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
