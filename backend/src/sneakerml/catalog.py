"""The sneaker catalog: one row per shoe model, for search and prediction.

The cleaned dataset is a table of *sales*; the app needs a table of
*shoes*. `build_catalog` collapses sales down to one row per `slug`
(which is the stable id used everywhere else in the project) carrying the
attributes a user picks from -- display name, brand, silhouette,
colorway, retail price, release date -- plus a little popularity context
(`n_sales`, `median_sale`, `sizes_seen`).

`PricePredictor` reads this file to build a feature row for a shoe the
user selected by id, and the frontend reads it to power the search box,
so it is written out as plain JSON (`data/processed/catalog.json`) rather
than parquet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sneakerml.config import PROCESSED_DIR
from sneakerml.data.simulate import sku_from_slug
from sneakerml.features import (
    OTHER_SILHOUETTE,
    UNKNOWN_COLORWAY,
    parse_sneaker_name,
)

CATALOG_COLUMNS = [
    "id",
    "sku",
    "sneaker_name",
    "display_name",
    "brand",
    "silhouette",
    "colorway",
    "retail_price",
    "release_date",
    "n_sales",
    "median_sale",
    "sizes_seen",
]

DEFAULT_CATALOG_PATH = PROCESSED_DIR / "catalog.json"

#: Columns `search` does a case-insensitive substring match against.
SEARCH_FIELDS = ["display_name", "sku", "id"]


def sku_for_slug(slug: str) -> str:
    """Deterministic 8-char SKU for a catalog slug.

    Reuses `simulate.sku_from_slug` on the *space-separated* form of the
    slug, which is exactly the string the simulated listings hash, so a
    catalog SKU and a listing SKU for the same shoe are identical.
    """
    return sku_from_slug(str(slug).replace("-", " "))


def display_name_for(base_brand: str, silhouette: str, colorway: str) -> str:
    """Human-readable shoe name, e.g. ``Adidas Yeezy Boost 350 V2 "Beluga"``.

    Format is ``{base_brand} {silhouette} "{colorway}"``. The quoted
    colorway is dropped when it is the "unknown" sentinel, and the
    silhouette is dropped when it is the "other" sentinel, so a
    partially-parsed name still renders as something sensible rather than
    leaking sentinel strings into the UI.
    """
    parts = [str(base_brand).strip()]
    if silhouette and silhouette != OTHER_SILHOUETTE:
        parts.append(str(silhouette).strip())
    name = " ".join(p for p in parts if p)
    if colorway and colorway != UNKNOWN_COLORWAY:
        name = f'{name} "{colorway}"'
    return name.strip()


def build_catalog(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the cleaned sales table into one row per shoe.

    Accepts either the raw output of `clean.build_clean_dataset` or the
    featured frame from `features.add_features` -- the parsed name fields
    are recomputed from `sneaker_name` either way, so the two agree.

    `retail_price` / `release_date` / `brand` are taken from the earliest
    sale of each shoe (they are per-model constants in this dataset);
    `sizes_seen` is the sorted list of distinct sizes actually traded.
    """
    df = clean_df.sort_values("order_date")

    grouped = df.groupby("slug", as_index=False).agg(
        sneaker_name=("sneaker_name", "first"),
        brand=("brand", "first"),
        retail_price=("retail_price", "first"),
        release_date=("release_date", "first"),
        n_sales=("sale_price", "size"),
        median_sale=("sale_price", "median"),
    )

    sizes = (
        df.groupby("slug")["shoe_size"]
        .apply(lambda s: sorted(float(v) for v in s.dropna().unique()))
        .rename("sizes_seen")
        .reset_index()
    )
    grouped = grouped.merge(sizes, on="slug", how="left")

    parsed = pd.DataFrame(
        list(grouped["sneaker_name"].map(parse_sneaker_name)), index=grouped.index
    )
    grouped["silhouette"] = parsed["silhouette"]
    grouped["colorway"] = parsed["colorway"]
    grouped["display_name"] = [
        display_name_for(b, s, c)
        for b, s, c in zip(parsed["base_brand"], parsed["silhouette"], parsed["colorway"])
    ]

    grouped["id"] = grouped["slug"]
    grouped["sku"] = grouped["slug"].map(sku_for_slug)
    grouped["n_sales"] = grouped["n_sales"].astype(int)
    grouped["retail_price"] = grouped["retail_price"].astype(float)
    grouped["median_sale"] = grouped["median_sale"].astype(float)

    catalog = grouped[CATALOG_COLUMNS].sort_values(
        ["n_sales", "id"], ascending=[False, True]
    )
    return catalog.reset_index(drop=True)


def search(catalog: pd.DataFrame, q: str, limit: int = 10) -> pd.DataFrame:
    """Case-insensitive substring search over display name, SKU and slug.

    Results are ranked by `n_sales` (most-traded first) and truncated to
    `limit`. An empty/whitespace query returns the top sellers -- but
    this path is exercised only by tests, not by any real caller: the API
    layer (`routes.search_sneakers`) rejects a blank `q` with 400, and
    the frontend's `SearchBar` never calls the endpoint until the user
    has typed 2+ characters.
    """
    query = str(q or "").strip().lower()
    ranked = catalog.sort_values(["n_sales", "id"], ascending=[False, True])
    if not query:
        return ranked.head(limit).reset_index(drop=True)

    mask = pd.Series(False, index=ranked.index)
    for field in SEARCH_FIELDS:
        mask |= ranked[field].astype(str).str.lower().str.contains(query, regex=False)
    return ranked[mask].head(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def _to_record(row: pd.Series) -> dict:
    record = row.to_dict()
    record["release_date"] = str(pd.Timestamp(record["release_date"]).date())
    record["sizes_seen"] = [float(s) for s in record["sizes_seen"]]
    record["retail_price"] = float(record["retail_price"])
    record["median_sale"] = float(record["median_sale"])
    record["n_sales"] = int(record["n_sales"])
    return record


def save_catalog(catalog: pd.DataFrame, path: Path = DEFAULT_CATALOG_PATH) -> Path:
    """Write the catalog to JSON (a list of records, dates as YYYY-MM-DD)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    records = [_to_record(row) for _, row in catalog.iterrows()]
    dest.write_text(json.dumps(records, indent=2) + "\n")
    return dest


def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> pd.DataFrame:
    """Read a catalog JSON file back into a DataFrame."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(
            f"catalog not found at {src} (run `python -m sneakerml.cli catalog` first)"
        )
    catalog = pd.DataFrame(json.loads(src.read_text()), columns=CATALOG_COLUMNS)
    catalog["release_date"] = pd.to_datetime(catalog["release_date"])
    return catalog
