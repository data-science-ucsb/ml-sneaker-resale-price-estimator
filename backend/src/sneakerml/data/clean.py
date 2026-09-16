"""Clean and merge the StockX ("real") and simulated listings ("goatish")
sources into one unified sale-price dataset.

Two rows can describe the same underlying sale in two different ways:

- StockX (`load_stockx`): the ground-truth 2019 dataset. Columns are
  renamed to snake_case; `Days Since` and `Profit` are dropped because
  they leak information not available at prediction time (they're
  computed *after* the sale/from the sale price itself).
- Listings (`load_listings`): a simulated second marketplace
  ("goatish") produced by `simulate.make_listings`. Prices are
  fee-inclusive dollar strings, sizes are sometimes EU or missing,
  titles are slugified. This module reverses each of those
  transformations so both sources land on the same schema before
  merging.

After both sources are loaded, `impute_size` fills missing sizes with
the per-sneaker median, `dedupe` removes both within-source scrape
duplicates and cross-source duplicates (the same sale appearing on both
platforms), and `build_clean_dataset` ties it all together, writing a
parquet file plus a JSON report of row counts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sneakerml.config import PROCESSED_DIR, RAW_DIR

STOCKX_SOURCE = "stockx"
GOATISH_SOURCE = "goatish"

# Must match simulate.py's FEE_MULTIPLIER / FEE_FLAT_FEE / EU_OFFSET exactly
# (inverting the transforms applied there).
GOATISH_FEE_MULTIPLIER = 1.095
GOATISH_FEE_FLAT_FEE = 13.95
EU_OFFSET = 33

CROSS_SOURCE_PRICE_TOLERANCE = 2.0

UNIFIED_COLUMNS = [
    "order_date",
    "brand",
    "sneaker_name",
    "slug",
    "sale_price",
    "retail_price",
    "release_date",
    "shoe_size",
    "buyer_region",
    "source",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def slugify(name: str) -> str:
    """Lowercase-hyphenated slug used as the cross-source join key.

    "Adidas-Yeezy-Boost-350-V2-Butter" -> "adidas-yeezy-boost-350-v2-butter"
    """
    text = str(name).replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.replace(" ", "-")


_SUFFIX_RE = re.compile(r"\s*(\(gs\)|size\s+[\d.]+)\s*$", re.IGNORECASE)


def slug_to_name(title: str) -> str:
    """Reconstruct a StockX-style hyphenated name from a listings title.

    Listings titles are lowercase with spaces instead of hyphens and may
    carry a trailing " (gs)" or " size N" suffix (stripped first), e.g.
    "adidas yeezy boost 350 v2 butter (gs)" -> "Adidas-Yeezy-Boost-350-V2-Butter".
    """
    text = str(title)
    # strip suffixes repeatedly in case of odd double-suffixing
    prev = None
    while prev != text:
        prev = text
        text = _SUFFIX_RE.sub("", text)
    words = text.strip().split(" ")
    return "-".join(w.title() for w in words if w)


def parse_price(value: str) -> float:
    """Parse a dollar string like "$1,097" or "$302.87" into a float."""
    text = str(value).strip().lstrip("$").replace(",", "")
    return float(text)


def normalize_price(price: float, source: str) -> float:
    """Reverse a source's fee formula to recover the underlying sale price.

    StockX prices need no adjustment. Goatish prices are always
    fee-inclusive: goatish_price = stockx_price * 1.095 + 13.95, so we
    invert and round to the nearest cent (the round trip is only accurate
    to ~1 cent because of the string-formatting round trip, not bit-exact).
    """
    if source == GOATISH_SOURCE:
        return round((price - GOATISH_FEE_FLAT_FEE) / GOATISH_FEE_MULTIPLIER, 2)
    return price


def normalize_size(value, system) -> float:
    """Convert a listings size to US sizing given its `size_system`.

    EU sizes are offset by 33 (`eu_size = us_size + 33`); US sizes pass
    through unchanged; a missing/NaN system or value yields NaN (an
    actually-missing size, to be imputed later — not a system to
    normalize).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if system is None or (isinstance(system, float) and np.isnan(system)):
        return np.nan
    if system == "EU":
        return float(value) - EU_OFFSET
    return float(value)


# ---------------------------------------------------------------------------
# per-source loaders
# ---------------------------------------------------------------------------


def load_stockx(path) -> pd.DataFrame:
    """Load the StockX CSV into the unified schema.

    Renames columns to snake_case, strips the BOM/whitespace pandas
    otherwise leaves in the header, strips leading whitespace out of
    `Brand` values (the raw CSV has " Yeezy" etc.), drops the leakage
    columns `Days Since` / `Profit`, and parses both date columns.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    df = df.rename(
        columns={
            "Order Date": "order_date",
            "Brand": "brand",
            "Sneaker Name": "sneaker_name",
            "Sale Price": "sale_price",
            "Retail Price": "retail_price",
            "Release Date": "release_date",
            "Shoe Size": "shoe_size",
            "Buyer Region": "buyer_region",
        }
    )
    df = df.drop(columns=["Days Since", "Profit"], errors="ignore")

    df["brand"] = df["brand"].astype(str).str.strip()
    df["order_date"] = pd.to_datetime(df["order_date"], format="%m/%d/%y")
    df["release_date"] = pd.to_datetime(df["release_date"], format="%m/%d/%y")
    df["sale_price"] = df["sale_price"].astype(float)
    df["retail_price"] = df["retail_price"].astype(float)
    df["shoe_size"] = df["shoe_size"].astype(float)
    df["slug"] = df["sneaker_name"].map(slugify)
    df["source"] = STOCKX_SOURCE

    return df[UNIFIED_COLUMNS]


def _build_catalog(stockx: pd.DataFrame) -> pd.DataFrame:
    """slug -> first-seen {brand, retail_price, release_date} lookup from StockX.

    `brand` is a per-shoe-model attribute (every StockX row sharing a slug
    has the same brand), so it is mechanically derivable the same way as
    `retail_price`/`release_date` rather than left unset for listings rows.
    """
    return (
        stockx.sort_values("order_date")
        .groupby("slug", as_index=False)
        .agg(
            brand=("brand", "first"),
            retail_price=("retail_price", "first"),
            release_date=("release_date", "first"),
        )
    )


def load_listings(path, stockx_path=None, catalog: pd.DataFrame | None = None, keep_kind: bool = False) -> pd.DataFrame:
    """Load the simulated "goatish" listings CSV into the unified schema.

    Reverses every dirtying transform from `simulate.make_listings`:
    dollar-string price -> float -> fee-reversed sale price; EU sizes ->
    US; slugified title -> reconstructed sneaker_name/slug. `brand`,
    `retail_price`, and `release_date` (not carried by listings) are
    looked up from the StockX catalog by slug; a listings row whose slug
    has no catalog match is dropped (defensive — shouldn't happen with
    this generator).

    Either `stockx_path` or a pre-built `catalog` DataFrame (as returned
    by `_build_catalog`) must be provided.
    """
    if catalog is None:
        if stockx_path is None:
            raise ValueError("load_listings requires either stockx_path or catalog")
        catalog = _build_catalog(load_stockx(stockx_path))

    df = pd.read_csv(path)

    df["sneaker_name"] = df["title"].map(slug_to_name)
    df["slug"] = df["sneaker_name"].map(slugify)

    df["sale_price"] = [
        normalize_price(parse_price(p), GOATISH_SOURCE) for p in df["price"]
    ]
    df["shoe_size"] = [
        normalize_size(v, s) for v, s in zip(df["size"], df["size_system"])
    ]
    df["order_date"] = pd.to_datetime(df["listed_at"])
    df["source"] = GOATISH_SOURCE
    df["buyer_region"] = np.nan

    n_before = len(df)
    df = df.merge(catalog, on="slug", how="left")
    unmatched = df["retail_price"].isna().sum()
    if unmatched:
        df = df[df["retail_price"].notna()].copy()

    keep_cols = list(UNIFIED_COLUMNS)
    if keep_kind and "_kind" in df.columns:
        keep_cols = keep_cols + ["_kind"]

    df = df[keep_cols].reset_index(drop=True)
    df.attrs["dropped_unmatched_slugs"] = int(unmatched)
    df.attrs["rows_before_catalog_join"] = int(n_before)
    return df


# ---------------------------------------------------------------------------
# imputation
# ---------------------------------------------------------------------------


def impute_size(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing `shoe_size` with the per-slug median size.

    Adds a `size_imputed` bool flag column so downstream consumers can
    tell real sizes from imputed ones (teaching point: group-wise
    imputation instead of a single global fallback).
    """
    df = df.copy()
    df["size_imputed"] = df["shoe_size"].isna()
    medians = df.groupby("slug")["shoe_size"].transform("median")
    df["shoe_size"] = df["shoe_size"].fillna(medians)
    # fall back to the global median for any slug with no known sizes at all
    global_median = df["shoe_size"].median()
    df["shoe_size"] = df["shoe_size"].fillna(global_median)
    return df


# ---------------------------------------------------------------------------
# dedupe
# ---------------------------------------------------------------------------


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove duplicate rows in two passes, both counted in the report.

    1. Within-source: exact duplicates on
       (source, slug, shoe_size, order_date, sale_price) — scrape
       artifacts appended by a single source.
    2. Cross-source: the same underlying sale can appear on both
       platforms. Key is (slug, shoe_size, order_date) *excluding*
       source, with sale_price matching within
       CROSS_SOURCE_PRICE_TOLERANCE dollars; the `stockx` row is kept
       over the `goatish` row.

    Preserves the original index of surviving rows (callers rely on this
    to identify which rows were removed).
    """
    n_start = len(df)

    before_within = len(df)
    df = df.drop_duplicates(
        subset=["source", "slug", "shoe_size", "order_date", "sale_price"], keep="first"
    )
    within_source_dupes_removed = before_within - len(df)

    # cross-source pass: group by (slug, order_date) only -- shoe_size is
    # excluded from the group key itself because a dirtied/missing size on
    # the listings side (NaN) would otherwise fail to match a StockX row
    # with a real size (NaN != NaN breaks equality-based grouping); we
    # still require matching sizes within the group when both are known.
    before_cross = len(df)
    df = df.sort_values(
        by=["slug", "order_date", "source"],
        key=lambda col: col.map({STOCKX_SOURCE: 0, GOATISH_SOURCE: 1}) if col.name == "source" else col,
    )

    positions = df.index.to_numpy()

    # group consecutive rows sharing (slug, order_date)
    group_cols = df[["slug", "order_date"]]
    group_id = (group_cols != group_cols.shift()).any(axis=1).cumsum().to_numpy()

    prices = df["sale_price"].to_numpy(dtype=float)
    sizes = df["shoe_size"].to_numpy(dtype=float)
    sources = df["source"].to_numpy(dtype=object)

    kept_local = np.ones(len(df), dtype=bool)
    start = 0
    n = len(df)
    while start < n:
        end = start
        while end + 1 < n and group_id[end + 1] == group_id[start]:
            end += 1
        # rows[start:end+1] share (slug, order_date)
        idxs = list(range(start, end + 1))
        for a_pos, a in enumerate(idxs):
            if not kept_local[a]:
                continue
            for b in idxs[a_pos + 1 :]:
                if not kept_local[b]:
                    continue
                if sources[a] == sources[b]:
                    # this pass only merges *cross*-source matches; two
                    # rows from the same source that share date/slug/price
                    # are two distinct sales (or an exact-duplicate scrape
                    # artifact already handled by the within-source pass),
                    # never a cross-listing of each other.
                    continue
                size_a, size_b = sizes[a], sizes[b]
                size_compatible = (
                    np.isnan(size_a) or np.isnan(size_b) or size_a == size_b
                )
                if size_compatible and abs(prices[a] - prices[b]) <= CROSS_SOURCE_PRICE_TOLERANCE:
                    # prefer stockx; rows already sorted stockx-first within group
                    if sources[a] == STOCKX_SOURCE or sources[b] != STOCKX_SOURCE:
                        kept_local[b] = False
                    else:
                        kept_local[a] = False
        start = end + 1

    df = df.loc[positions[kept_local]]
    cross_source_dupes_removed = before_cross - len(df)

    report = {
        "rows_before_dedupe": n_start,
        "within_source_dupes_removed": int(within_source_dupes_removed),
        "cross_source_dupes_removed": int(cross_source_dupes_removed),
        "rows_after_dedupe": len(df),
    }
    return df, report


# ---------------------------------------------------------------------------
# top-level pipeline
# ---------------------------------------------------------------------------


def build_clean_dataset(
    raw_dir: Path = RAW_DIR,
    out: Path = PROCESSED_DIR / "clean.parquet",
) -> pd.DataFrame:
    """Load both sources, clean/normalize, dedupe, impute, and write output.

    Writes `out` (parquet) and a `clean_report.json` alongside it with
    row counts, dedupe counts, and imputation counts.
    """
    raw_dir = Path(raw_dir)
    out = Path(out)

    stockx_path = raw_dir / "stockx_2019.csv"
    listings_path = raw_dir / "listings_sim.csv"

    stockx = load_stockx(stockx_path)
    catalog = _build_catalog(stockx)
    listings = load_listings(listings_path, catalog=catalog)

    rows_in = len(stockx) + len(listings)
    dropped_unmatched = listings.attrs.get("dropped_unmatched_slugs", 0)

    combined = pd.concat([stockx, listings], ignore_index=True)

    combined, dedupe_report = dedupe(combined)

    combined = impute_size(combined)
    sizes_imputed = int(combined["size_imputed"].sum())

    before_filter = len(combined)
    combined = combined[(combined["sale_price"] > 0) & (combined["retail_price"] > 0)]
    filtered_out = before_filter - len(combined)

    combined = combined.reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)

    report = {
        "rows_in": rows_in,
        "rows_out": len(combined),
        "dropped_unmatched_slugs": int(dropped_unmatched),
        "within_source_dupes_removed": dedupe_report["within_source_dupes_removed"],
        "cross_source_dupes_removed": dedupe_report["cross_source_dupes_removed"],
        "sizes_imputed": sizes_imputed,
        "invalid_price_rows_filtered": int(filtered_out),
    }
    report_path = out.parent / "clean_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    return combined
