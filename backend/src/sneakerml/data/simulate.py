"""Generate a simulated second marketplace source ("goatish") from StockX sales.

The goal is to produce realistic cross-source messiness for Task 3's cleaning
and dedupe step to reverse: some listings are the exact same underlying sale
seen on StockX (should be caught by dedupe), some are genuinely new listings,
and every listing is "dirtied" the way a scraped competitor site would be
(fee-inclusive string prices, EU sizing, missing sizes, slugified titles,
occasional exact-duplicate scrape artifacts).

All transforms below are deterministic given `seed` and documented exactly
(including the fee/size formulas) so a downstream cleaning step can invert
them precisely.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from sneakerml.config import RANDOM_SEED

SOURCE_NAME = "goatish"

# --- fee-inclusive pricing -------------------------------------------------
# Every simulated listing's price is rewritten fee-inclusive:
#   goatish_price = stockx_price * FEE_MULTIPLIER + FEE_FLAT_FEE
# To invert (Task 3): stockx_price = (goatish_price - FEE_FLAT_FEE) / FEE_MULTIPLIER
FEE_MULTIPLIER = 1.095
FEE_FLAT_FEE = 13.95

# --- US -> EU shoe size (men's sizing approximation) ------------------------
#   eu_size = us_size + EU_OFFSET
# To invert: us_size = eu_size - EU_OFFSET
EU_OFFSET = 33

# --- population split & jitter ----------------------------------------------
CROSSLISTED_FRACTION = 0.60
NEW_PRICE_JITTER_MIN = 0.03
NEW_PRICE_JITTER_MAX = 0.08
NEW_DATE_SHIFT_MIN_DAYS = 1
NEW_DATE_SHIFT_MAX_DAYS = 10

# --- dirtying rates (exact counts, not per-row coin flips) ------------------
EU_SIZE_FRACTION = 0.30
MISSING_SIZE_FRACTION = 0.08
DUPLICATE_FRACTION = 0.10
GS_SUFFIX_FRACTION = 0.05
SIZE_SUFFIX_FRACTION = 0.05

OUTPUT_COLUMNS = [
    "listing_id",
    "sku",
    "title",
    "price",
    "size",
    "size_system",
    "listed_at",
    "source",
    "_kind",
]


def slugify(name: str) -> str:
    """Lowercase, space-separated slug (hyphens -> spaces) used for titles/skus."""
    return re.sub(r"\s+", " ", str(name).replace("-", " ")).strip().lower()


def us_to_eu(us_size: np.ndarray) -> np.ndarray:
    """Convert US shoe size(s) to the approximate EU equivalent."""
    return us_size + EU_OFFSET


def sku_from_slug(slug: str) -> str:
    """Deterministic 8-char SKU derived from a title slug (same slug -> same sku)."""
    digest = hashlib.md5(slug.encode("utf-8")).hexdigest()
    return digest[:8].upper()


def _format_price(value: float) -> str:
    return f"${value:,.2f}"


def make_listings(
    stockx: pd.DataFrame,
    n: int = 20_000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Build a simulated "goatish" marketplace listings table from StockX sales.

    Two populations, exact counts (not probabilistic averages):
    - 60% cross-listed (`_kind="crosslisted"`): the same sale as a StockX
      row (same date, size, and price, before dirtying).
    - 40% new (`_kind="new"`): price jittered +-3-8%, date shifted +-1-10
      days, so they survive cross-source dedupe as genuinely new rows.

    Every row (both kinds) is then dirtied:
    - price rewritten fee-inclusive (`* 1.095 + 13.95`) and formatted as a
      "$1,234.56" string; source is always "goatish".
    - exactly ~30% of sizes converted to EU (`us_to_eu`), ~8% blanked out.
    - titles lower-cased with hyphens replaced by spaces, with an
      occasional "(gs)" or "size N" suffix.
    - ~10% of the final rows re-appended as exact duplicates (scrape
      artifacts).
    - sku is a deterministic hash of the (pre-suffix) title slug, so the
      same shoe always gets the same sku.

    Returns `n + round(n * 0.10)` rows. Deterministic for a given `seed`.
    """
    rng = np.random.default_rng(seed)

    stockx = stockx.reset_index(drop=True)
    pool_size = len(stockx)
    dates = pd.to_datetime(stockx["Order Date"], format="%m/%d/%y")

    n_cross = round(n * CROSSLISTED_FRACTION)
    n_new = n - n_cross

    cross_idx = rng.integers(0, pool_size, size=n_cross)
    new_idx = rng.integers(0, pool_size, size=n_new)

    records = []

    # -- cross-listed: identical sale (date, size, price before dirtying) --
    for i in cross_idx:
        row = stockx.iloc[i]
        records.append(
            {
                "title": row["Sneaker Name"],
                "price": float(row["Sale Price"]),
                "size": float(row["Shoe Size"]),
                "listed_at": dates.iloc[i],
                "_kind": "crosslisted",
            }
        )

    # -- new: jittered price, shifted date --
    jitter_pct = rng.uniform(NEW_PRICE_JITTER_MIN, NEW_PRICE_JITTER_MAX, size=n_new)
    jitter_sign = rng.choice([-1.0, 1.0], size=n_new)
    date_shift = rng.integers(NEW_DATE_SHIFT_MIN_DAYS, NEW_DATE_SHIFT_MAX_DAYS + 1, size=n_new)
    shift_sign = rng.choice([-1, 1], size=n_new)

    for j, i in enumerate(new_idx):
        row = stockx.iloc[i]
        new_price = float(row["Sale Price"]) * (1 + jitter_sign[j] * jitter_pct[j])
        new_date = dates.iloc[i] + pd.Timedelta(days=int(shift_sign[j] * date_shift[j]))
        records.append(
            {
                "title": row["Sneaker Name"],
                "price": new_price,
                "size": float(row["Shoe Size"]),
                "listed_at": new_date,
                "_kind": "new",
            }
        )

    df = pd.DataFrame.from_records(records)
    shuffle_seed = int(rng.integers(0, 2**31 - 1))
    df = df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    total = len(df)

    # --- size dirtying: exact, disjoint EU / missing counts ---
    perm = rng.permutation(total)
    n_missing = round(total * MISSING_SIZE_FRACTION)
    n_eu = round(total * EU_SIZE_FRACTION)
    missing_positions = perm[:n_missing]
    eu_positions = perm[n_missing : n_missing + n_eu]

    size_system = np.full(total, "US", dtype=object)
    size_system[eu_positions] = "EU"

    sizes = df["size"].to_numpy(dtype=float).copy()
    sizes[eu_positions] = us_to_eu(sizes[eu_positions])
    sizes[missing_positions] = np.nan
    size_system[missing_positions] = None

    df["size"] = sizes
    df["size_system"] = size_system

    # --- price dirtying: fee-inclusive $-string (applies to every row) ---
    dirtied_price = df["price"].to_numpy(dtype=float) * FEE_MULTIPLIER + FEE_FLAT_FEE
    df["price"] = [_format_price(p) for p in dirtied_price]

    # --- title dirtying: slug + occasional suffix ---
    slugs = df["title"].map(slugify)
    gs_mask = rng.random(total) < GS_SUFFIX_FRACTION
    size_suffix_mask = (~gs_mask) & (rng.random(total) < SIZE_SUFFIX_FRACTION)

    titles = slugs.to_numpy(dtype=object).copy()
    display_sizes = df["size"].to_numpy(dtype=float)
    for idx in np.where(gs_mask)[0]:
        titles[idx] = f"{titles[idx]} (gs)"
    for idx in np.where(size_suffix_mask)[0]:
        size_val = display_sizes[idx]
        if not np.isnan(size_val):
            titles[idx] = f"{titles[idx]} size {size_val:g}"

    df["title"] = titles
    df["sku"] = slugs.map(sku_from_slug)
    df["listed_at"] = df["listed_at"].dt.strftime("%Y-%m-%d")
    df["source"] = SOURCE_NAME
    df["listing_id"] = [f"L{i:07d}" for i in range(total)]

    df = df[OUTPUT_COLUMNS]

    # --- ~10% exact duplicate rows appended (scrape artifacts) ---
    n_dupes = round(total * DUPLICATE_FRACTION)
    dupe_positions = rng.integers(0, total, size=n_dupes)
    dupes = df.iloc[dupe_positions].reset_index(drop=True)
    df = pd.concat([df, dupes], ignore_index=True)

    return df
