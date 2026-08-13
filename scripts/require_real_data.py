"""Fail unless a raw dataset and its provenance are explicitly real."""

import json
import sys
from pathlib import Path


def main() -> None:
    raw = Path(sys.argv[1])
    provenance = Path(sys.argv[2])
    if not raw.is_file():
        raise SystemExit(f"real dataset is missing: {raw}")
    try:
        source = json.loads(provenance.read_text())["source"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"real dataset requires valid provenance: {provenance}"
        ) from exc
    if not isinstance(source, str) or not source.startswith("REAL"):
        raise SystemExit(f"dataset provenance is not real: {source!r}")


if __name__ == "__main__":
    main()
