"""Fetch the raw StockX 2019 dataset (the "real" data source).

`fetch_stockx` downloads the CSV over HTTP (or via the Kaggle CLI when
`--kaggle` is requested and credentials exist), writes it atomically, and
validates a minimum row count so a truncated/corrupt download fails loudly
instead of silently poisoning the pipeline.
"""

from __future__ import annotations

import shutil
import ssl
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from sneakerml.config import MIN_EXPECTED_ROWS, RAW_DIR, STOCKX_URL

RETRIES = 3
KAGGLE_DATASET = "hudsonstuck/stockx-data-contest"


def _ssl_context() -> ssl.SSLContext | None:
    """Build an HTTPS context using certifi's CA bundle when available.

    Some Python installs (notably python.org builds on macOS) ship without
    a usable system CA bundle, which makes plain `urlopen` fail HTTPS
    verification even though the site's certificate is fine. Falling back
    to certifi's bundle when it's installed avoids that; if certifi isn't
    available, `urlopen` uses its own default context.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def _download_bytes(url: str, retries: int = RETRIES) -> bytes:
    """Fetch `url` with up to `retries` attempts, returning the raw bytes."""
    context = _ssl_context()
    last_exc: Exception | None = None
    for _attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30, context=context) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_exc


def _write_atomically(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with open(fd, "wb") as f:
            f.write(data)
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)


def _fetch_via_kaggle(dest: Path) -> None:
    """Shell out to the Kaggle CLI and move the resulting CSV to `dest`."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                KAGGLE_DATASET,
                "-p",
                tmp,
                "--unzip",
            ],
            check=True,
        )
        csvs = sorted(Path(tmp).glob("*.csv"))
        if not csvs:
            raise RuntimeError("kaggle download produced no CSV file")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(csvs[0]), str(dest))


def _validate_row_count(dest: Path, min_rows: int) -> Path:
    """Count data rows (excluding header), stripping a possible BOM."""
    with open(dest, encoding="utf-8-sig") as f:
        line_count = sum(1 for _ in f)
    row_count = max(line_count - 1, 0)
    if row_count < min_rows:
        raise RuntimeError(
            f"{dest} has {row_count} data rows, expected at least {min_rows}"
        )
    return dest


def fetch_stockx(
    dest: Path = RAW_DIR / "stockx_2019.csv",
    url: str = STOCKX_URL,
    min_rows: int = MIN_EXPECTED_ROWS,
    kaggle: bool = False,
) -> Path:
    """Download the StockX CSV to `dest` and validate its row count.

    Parameters
    ----------
    dest: where to write the CSV.
    url: source URL; accepts http(s):// and file:// (urllib handles both).
    min_rows: minimum number of data rows (excluding header) required;
        raises RuntimeError if the downloaded file has fewer.
    kaggle: if True and `~/.kaggle/kaggle.json` exists, fetch via the
        `kaggle datasets download` CLI instead of the plain HTTP `url`.
    """
    dest = Path(dest)

    if kaggle and (Path.home() / ".kaggle" / "kaggle.json").exists():
        _fetch_via_kaggle(dest)
        return _validate_row_count(dest, min_rows)

    data = _download_bytes(url)
    _write_atomically(dest, data)
    return _validate_row_count(dest, min_rows)
