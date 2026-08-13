"""Fetch the real ULB Credit Card Fraud dataset — no Kaggle account needed.

The canonical dataset lives on Kaggle (mlg-ulb/creditcardfraud) behind a login.
The identical data is mirrored on OpenML as dataset 1597 ("creditcard"), which
is openly downloadable. Note: OpenML flags ``Time`` as a row-identifier, so the
usual ``fetch_openml`` drops it; we download the raw parquet instead, which
keeps all 31 columns (Time, V1..V28, Amount, Class) in the canonical schema.

The source file is ~70 MB. A single ``urlopen().read()`` fails outright if the
connection drops part-way (``http.client.IncompleteRead``), and on a throttled
or unstable link that is common enough to block reproduction entirely. The
download is therefore streamed to a ``.part`` file and resumed with HTTP Range
requests, retrying with exponential backoff. The partial file survives a failed
run, so re-running this module continues from where it stopped.

Usage:  python -m src.fetch_data   (or `make fetch-data`)
"""
import time
import urllib.error
import urllib.request
from http.client import IncompleteRead
from pathlib import Path

import pandas as pd

from src.config import ROOT, load_params, write_provenance

PARQUET_URL = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

CHUNK_BYTES = 1 << 20  # 1 MiB
MAX_ATTEMPTS = 8
TIMEOUT_SECONDS = 180

# "The transfer broke" errors, which are retryable. A 404 or 403 is an
# HTTPError (a URLError subclass) and is re-raised immediately below.
_RETRYABLE = (IncompleteRead, urllib.error.URLError, TimeoutError, OSError)


def _expected_total(response) -> int | None:
    """Total resource size, from Content-Range (206) or Content-Length (200)."""
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1].strip()
        if total.isdigit():
            return int(total)
    length = response.headers.get("Content-Length")
    if length and length.isdigit():
        return int(length)
    return None


def download(url: str = PARQUET_URL, dest: Path | None = None) -> Path:
    """Stream ``url`` to ``dest``, resuming a partial download if one exists."""
    dest = Path(dest) if dest is not None else ROOT / "data" / "raw" / "dataset_1597.pq"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    total: int | None = None
    last_error: Exception | None = None
    completed = False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = part.stat().st_size if part.exists() else 0
        if total is not None and have >= total:
            completed = True
            break

        request = urllib.request.Request(url)
        if have:
            request.add_header("Range", f"bytes={have}-")

        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed https URL
                # A server that ignores Range replies 200 with the whole body;
                # appending that would corrupt the file, so restart at byte 0.
                if have and response.status != 206:
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"

                if total is None:
                    total = _expected_total(response)
                    if total is not None and response.status == 206:
                        pass  # Content-Range already carries the full size

                with open(part, mode) as handle:
                    while True:
                        chunk = response.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        handle.write(chunk)
                        have += len(chunk)
                        if total:
                            print(
                                f"\r  {have / 1e6:,.1f} / {total / 1e6:,.1f} MB "
                                f"({100 * have / total:.1f}%)",
                                end="",
                                flush=True,
                            )
            print()
            if total is None or have >= total:
                completed = True
                break
            raise IncompleteRead(b"", total - have)

        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404, 410):
                raise  # the URL is wrong or gone; retrying cannot help
            last_error = exc
        except _RETRYABLE as exc:
            last_error = exc

        done = part.stat().st_size if part.exists() else 0
        if attempt == MAX_ATTEMPTS:
            break
        backoff = min(2 ** (attempt - 1), 30)
        print(
            f"\n  attempt {attempt}/{MAX_ATTEMPTS} interrupted after "
            f"{done / 1e6:,.1f} MB ({type(last_error).__name__}); "
            f"resuming in {backoff}s"
        )
        time.sleep(backoff)

    done = part.stat().st_size if part.exists() else 0
    completed = completed or (total is not None and done >= total)
    if not completed:
        progress = f"{done:,}/{total:,} bytes" if total is not None else f"{done:,} bytes"
        raise RuntimeError(
            f"download incomplete after {MAX_ATTEMPTS} attempts: {progress}. "
            f"The partial file is kept at {part} — re-run "
            f"`python -m src.fetch_data` to resume, or use the manual Kaggle "
            f"path documented in the README."
        ) from last_error
    if not done:
        raise RuntimeError(f"download produced no data: {last_error}") from last_error

    part.replace(dest)
    return dest


def fetch() -> pd.DataFrame:
    """Download (resumably) and load the dataset in the canonical schema."""
    path = download()
    try:
        df = pd.read_parquet(path)[COLS].copy()
    except Exception:
        path.unlink(missing_ok=True)  # never leave a corrupt file to poison reruns
        raise
    df["Class"] = df["Class"].astype(int)
    for c in COLS:
        if c != "Class":
            df[c] = df[c].astype(float)
    path.unlink(missing_ok=True)
    return df


def main() -> None:
    params = load_params()
    out = ROOT / params["data"]["raw_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df = fetch()
    df.to_csv(out, index=False)
    write_provenance(
        params, "REAL — ULB creditcard via OpenML dataset 1597",
        len(df), int(df["Class"].sum()),
    )
    print(
        f"[REAL] wrote {out} — {len(df):,} rows, {int(df['Class'].sum())} fraud "
        f"({df['Class'].mean():.6f}). Source: OpenML dataset 1597 (ULB creditcard)."
    )


if __name__ == "__main__":
    main()
