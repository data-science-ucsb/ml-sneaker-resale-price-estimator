"""Tests for the sneaker catalog (catalog.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from sneakerml.catalog import (
    CATALOG_COLUMNS,
    build_catalog,
    display_name_for,
    load_catalog,
    save_catalog,
    search,
    sku_for_slug,
)
from sneakerml.data.clean import impute_size, load_stockx
from sneakerml.data.simulate import sku_from_slug, slugify

BELUGA_SLUG = "adidas-yeezy-boost-350-low-v2-beluga"


@pytest.fixture(scope="module")
def clean_df(request) -> pd.DataFrame:
    csv_path = request.path.parent / "fixtures" / "stockx_sample.csv"
    return impute_size(load_stockx(csv_path))


@pytest.fixture(scope="module")
def catalog(clean_df) -> pd.DataFrame:
    return build_catalog(clean_df)


def test_catalog_has_expected_columns(catalog):
    assert list(catalog.columns) == CATALOG_COLUMNS


def test_catalog_is_one_row_per_slug(catalog, clean_df):
    assert len(catalog) == clean_df["slug"].nunique()
    assert catalog["id"].is_unique
    assert set(catalog["id"]) == set(clean_df["slug"])
    assert int(catalog["n_sales"].sum()) == len(clean_df)


def test_catalog_fields_are_sane(catalog):
    beluga = catalog[catalog["id"] == BELUGA_SLUG].iloc[0]
    assert beluga["brand"] == "Yeezy"
    assert beluga["silhouette"] == "Yeezy Boost 350 Low V2"
    assert beluga["colorway"] == "Beluga"
    assert beluga["display_name"] == 'Adidas Yeezy Boost 350 Low V2 "Beluga"'
    assert beluga["retail_price"] > 0
    assert beluga["median_sale"] > 0
    assert beluga["n_sales"] >= 1
    assert isinstance(beluga["sizes_seen"], list)
    assert all(isinstance(s, float) for s in beluga["sizes_seen"])
    assert beluga["sizes_seen"] == sorted(beluga["sizes_seen"])


def test_sku_matches_the_simulated_listings_formula(catalog):
    """Catalog SKUs must equal simulate.py's SKU for the same shoe."""
    row = catalog[catalog["id"] == BELUGA_SLUG].iloc[0]
    assert row["sku"] == sku_from_slug(slugify(row["sneaker_name"]))
    assert row["sku"] == sku_for_slug(BELUGA_SLUG)
    assert len(row["sku"]) == 8
    assert row["sku"].isupper() or row["sku"].isdigit()


def test_display_name_drops_unknown_colorway():
    assert display_name_for("Adidas", "Yeezy Boost 700", "Wave Runner") == (
        'Adidas Yeezy Boost 700 "Wave Runner"'
    )
    assert display_name_for("Adidas", "Yeezy Boost 700", "unknown") == (
        "Adidas Yeezy Boost 700"
    )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_finds_beluga(catalog):
    results = search(catalog, "beluga")
    assert len(results) > 0
    assert BELUGA_SLUG in set(results["id"])


def test_search_is_case_insensitive(catalog):
    assert set(search(catalog, "BELUGA")["id"]) == set(search(catalog, "beluga")["id"])


def test_search_matches_sku_and_slug(catalog):
    sku = catalog[catalog["id"] == BELUGA_SLUG].iloc[0]["sku"]
    assert BELUGA_SLUG in set(search(catalog, sku)["id"])
    assert BELUGA_SLUG in set(search(catalog, BELUGA_SLUG)["id"])


def test_search_is_ranked_by_n_sales_and_limited(catalog):
    results = search(catalog, "yeezy", limit=3)
    assert len(results) <= 3
    assert list(results["n_sales"]) == sorted(results["n_sales"], reverse=True)


def test_search_empty_query_returns_top_sellers(catalog):
    results = search(catalog, "", limit=5)
    assert len(results) == 5
    assert list(results["n_sales"]) == sorted(results["n_sales"], reverse=True)


def test_search_no_match_returns_empty(catalog):
    assert len(search(catalog, "zzz-not-a-shoe")) == 0


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def test_catalog_round_trips_through_json(catalog, tmp_path):
    path = save_catalog(catalog, tmp_path / "catalog.json")
    assert path.exists()

    payload = json.loads(path.read_text())
    assert isinstance(payload, list)
    assert payload[0]["release_date"] == str(
        pd.Timestamp(catalog.iloc[0]["release_date"]).date()
    )

    reloaded = load_catalog(path)
    assert list(reloaded.columns) == CATALOG_COLUMNS
    assert len(reloaded) == len(catalog)
    assert set(reloaded["id"]) == set(catalog["id"])
    assert pd.api.types.is_datetime64_any_dtype(reloaded["release_date"])
