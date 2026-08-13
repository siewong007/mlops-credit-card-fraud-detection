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
  # Seed the graph's raw input. The snapshot carries no CSV (data/raw is
  # gitignored), so this only ever writes inside the disposable clone.
  python -m src.simulate_data
  python -m dvc repro
  # Plain `dvc status` always exits 0, so it reports drift without ever failing
  # the gate; it runs first only to put the readable diff in the log. `--quiet`
  # is the form that exits 1 when the pipeline is not up to date after repro.
  python -m dvc status
  python -m dvc status --quiet
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
  # The export must be absolute so it lands outside the disposable clone. A
  # Windows absolute path carries a drive letter instead of a leading slash,
  # so C:/out is every bit as absolute as /home/runner/out and was previously
  # rejected here. Backslash form stays unsupported on purpose: bash treats \
  # as an escape, so C:\out cannot be normalised without quoting traps, and
  # the message below names the forms that do work.
  case "$DVC_REPRO_EXPORT_DIR" in
    /*|[A-Za-z]:/*) ;;
    *)
      echo "DVC_REPRO_EXPORT_DIR must be an absolute path using forward slashes," >&2
      echo "e.g. /home/you/out, /c/Users/you/out, or C:/Users/you/out" >&2
      exit 2
      ;;
  esac
  mkdir -p "$DVC_REPRO_EXPORT_DIR"
  cp "$scratch/repo/dvc.lock" "$DVC_REPRO_EXPORT_DIR/dvc.lock"
  cp -R "$scratch/repo/reports" "$DVC_REPRO_EXPORT_DIR/reports"
fi

echo "DVC clean-clone verification passed; caller raw checksum: $after"
