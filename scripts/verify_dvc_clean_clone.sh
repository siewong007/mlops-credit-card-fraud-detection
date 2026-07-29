#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
raw_path="$repo_root/data/raw/creditcard.csv"
scratch="$(mktemp -d)"

cleanup() {
  rm -rf "$scratch"
}
trap cleanup EXIT

checksum() {
  python - "$1" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.exists():
    print("missing")
else:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    print(digest.hexdigest())
PY
}

before="$(checksum "$raw_path")"
source_commit="$(git -C "$repo_root" rev-parse HEAD)"
mkdir -p "$scratch/repo"
git -C "$repo_root" archive "$source_commit" | tar -x -C "$scratch/repo"

(
  cd "$scratch/repo"
  git init --quiet
  git config user.name "DVC verification"
  git config user.email "dvc-verification@example.invalid"
  git add --all
  git commit --quiet -m "isolated verification snapshot"
  export SOURCE_COMMIT="$source_commit"
  python -m dvc repro
  python -m dvc status
  python -m src.evidence
  python -m src.build_dashboard
  if [[ -f tests/test_evidence_consistency.py ]]; then
    python -m pytest -q tests/test_evidence_consistency.py
  else
    python -m pytest -q tests/test_evidence.py
  fi
)

after="$(checksum "$raw_path")"
if [[ "$before" != "$after" ]]; then
  echo "caller raw data changed during DVC verification" >&2
  exit 1
fi

if [[ -n "${DVC_REPRO_EXPORT_DIR:-}" ]]; then
  case "$DVC_REPRO_EXPORT_DIR" in
    /*) ;;
    *)
      echo "DVC_REPRO_EXPORT_DIR must be an absolute path" >&2
      exit 2
      ;;
  esac
  mkdir -p "$DVC_REPRO_EXPORT_DIR"
  cp "$scratch/repo/dvc.lock" "$DVC_REPRO_EXPORT_DIR/dvc.lock"
  cp -R "$scratch/repo/reports" "$DVC_REPRO_EXPORT_DIR/reports"
fi

echo "DVC clean-clone verification passed; caller raw checksum: $after"
