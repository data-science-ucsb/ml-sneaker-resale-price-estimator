"""Tests for the cleaning/merging pipeline (clean.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sneakerml.config import RANDOM_SEED
from sneakerml.data.clean import (
    build_clean_dataset,
    dedupe,
    impute_size,
    load_listings,
    load_stockx,
    normalize_price,
    normalize_size,
    parse_price,
    slug_to_name,
    slugify,
)
from sneakerml.data.simulate import make_listings


# ---------------------------------------------------------------------------
# load_stockx
# ---------------------------------------------------------------------------


def test_load_stockx_brand_has_no_leading_space(fixture_csv):
    df = load_stockx(fixture_csv)
    assert (df["brand"].str.startswith(" ")).sum() == 0
    assert "Yeezy" in df["brand"].unique()


def test_load_stockx_leakage_columns_absent(fixture_csv):
    df = load_stockx(fixture_csv)
    assert "days_since" not in df.columns
    assert "profit" not in df.columns
    assert "Days Since" not in df.columns
    assert "Profit" not in df.columns


def test_load_stockx_column_schema(fixture_csv):
    df = load_stockx(fixture_csv)
    expected = {
        "order_date",
        "brand",
        "sneaker_name",
        "sale_price",
        "retail_price",
        "release_date",
        "shoe_size",
        "buyer_region",
        "source",
        "slug",
    }
    assert expected.issubset(set(df.columns))
    assert (df["source"] == "stockx").all()


def test_load_stockx_dates_are_datetime(fixture_csv):
    df = load_stockx(fixture_csv)
    assert np.issubdtype(df["order_date"].dtype, np.datetime64)
    assert np.issubdtype(df["release_date"].dtype, np.datetime64)


def test_load_stockx_sizes_are_float(fixture_csv):
    df = load_stockx(fixture_csv)
    assert np.issubdtype(df["shoe_size"].dtype, np.floating)


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------


def test_parse_price_basic():
    assert parse_price("$1,097") == 1097.0
    assert parse_price("$302.87") == pytest.approx(302.87)


def test_normalize_price_goatish_round_trips_within_a_dollar():
    original = 500.0
    goatish_price = original * 1.095 + 13.95
    goatish_str = f"${goatish_price:,.2f}"
    recovered = normalize_price(parse_price(goatish_str), "goatish")
    assert abs(recovered - original) < 1.0


def test_normalize_size_eu_to_us():
    assert normalize_size(44, "EU") == pytest.approx(11)
    assert normalize_size(46, "EU") == pytest.approx(13)


def test_normalize_size_us_passthrough():
    assert normalize_size(10.5, "US") == pytest.approx(10.5)


def test_normalize_size_missing_returns_nan():
    assert np.isnan(normalize_size(np.nan, np.nan))
    assert np.isnan(normalize_size(np.nan, None))


def test_slug_to_name_reconstructs_stockx_style():
    assert slug_to_name("adidas yeezy boost 350 v2 butter") == "Adidas-Yeezy-Boost-350-V2-Butter"


def test_slug_to_name_strips_gs_and_size_suffixes():
    assert slug_to_name("adidas yeezy boost 350 v2 butter (gs)") == "Adidas-Yeezy-Boost-350-V2-Butter"
    assert slug_to_name("adidas yeezy boost 350 v2 butter size 10.5") == "Adidas-Yeezy-Boost-350-V2-Butter"


def test_slugify_lowercases_and_hyphenates():
    assert slugify("Adidas-Yeezy-Boost-350-V2-Butter") == "adidas-yeezy-boost-350-v2-butter"


# ---------------------------------------------------------------------------
# load_listings
# ---------------------------------------------------------------------------


@pytest.fixture
def listings_csv(tmp_path, stockx_sample_df):
    listings = make_listings(stockx_sample_df, n=500, seed=RANDOM_SEED)
    path = tmp_path / "listings_sim.csv"
    listings.to_csv(path, index=False)
    return path


def test_load_listings_column_schema(listings_csv, fixture_csv):
    df = load_listings(listings_csv, stockx_path=fixture_csv)
    expected = {
        "order_date",
        "brand",
        "sneaker_name",
        "sale_price",
        "retail_price",
        "release_date",
        "shoe_size",
        "buyer_region",
        "source",
        "slug",
    }
    assert expected.issubset(set(df.columns))
    assert (df["source"] == "goatish").all()
    assert "_kind" not in df.columns


def test_load_listings_prices_recovered_within_dollar(listings_csv, fixture_csv):
    df = load_listings(listings_csv, stockx_path=fixture_csv)
    assert (df["sale_price"] > 0).all()


def test_load_listings_sizes_are_float(listings_csv, fixture_csv):
    df = load_listings(listings_csv, stockx_path=fixture_csv)
    assert np.issubdtype(df["shoe_size"].dtype, np.floating)


def test_load_listings_dates_are_datetime(listings_csv, fixture_csv):
    df = load_listings(listings_csv, stockx_path=fixture_csv)
    assert np.issubdtype(df["order_date"].dtype, np.datetime64)


def test_load_listings_brand_is_looked_up_from_catalog(listings_csv, fixture_csv, stockx_sample_df):
    """brand is a per-slug attribute derivable from the StockX catalog (like
    retail_price/release_date) -- it must not be left NaN for goatish rows."""
    df = load_listings(listings_csv, stockx_path=fixture_csv)

    assert df["brand"].notna().all()

    stockx = load_stockx(fixture_csv)
    slug_to_brand = dict(zip(stockx["slug"], stockx["brand"]))
    for slug, brand in zip(df["slug"], df["brand"]):
        assert brand == slug_to_brand[slug]

    yeezy_slugs = set(stockx.loc[stockx["brand"] == "Yeezy", "slug"])
    matched = df[df["slug"].isin(yeezy_slugs)]
    assert len(matched) > 0
    assert (matched["brand"] == "Yeezy").all()


# ---------------------------------------------------------------------------
# impute_size
# ---------------------------------------------------------------------------


def test_impute_size_fills_missing_and_flags():
    df = pd.DataFrame(
        {
            "slug": ["a", "a", "a", "b", "b"],
            "shoe_size": [10.0, 11.0, np.nan, 9.0, 9.0],
        }
    )
    out = impute_size(df)
    assert out["shoe_size"].isna().sum() == 0
    assert out.loc[2, "size_imputed"] == True  # noqa: E712
    assert out.loc[2, "shoe_size"] == pytest.approx(10.5)
    assert out["size_imputed"].sum() == 1


# ---------------------------------------------------------------------------
# dedupe
# ---------------------------------------------------------------------------


def test_dedupe_within_source_removes_exact_duplicates():
    df = pd.DataFrame(
        {
            "source": ["stockx", "stockx", "stockx"],
            "slug": ["a", "a", "b"],
            "shoe_size": [10.0, 10.0, 9.0],
            "order_date": pd.to_datetime(["2018-01-01", "2018-01-01", "2018-01-02"]),
            "sale_price": [100.0, 100.0, 200.0],
        }
    )
    result, report = dedupe(df)
    assert len(result) == 2
    assert report["within_source_dupes_removed"] == 1


def test_dedupe_cross_source_removal_rates(fixture_csv, stockx_sample_df):
    stockx = load_stockx(fixture_csv)

    raw_listings = make_listings(stockx_sample_df, n=800, seed=RANDOM_SEED)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "listings_sim.csv"
        raw_listings.to_csv(path, index=False)
        listings = load_listings(path, stockx_path=fixture_csv, keep_kind=True)

    combined = pd.concat([stockx.assign(_kind="stockx"), listings], ignore_index=True)
    kind_col = combined["_kind"]
    combined_for_dedupe = combined.drop(columns=["_kind"])

    # isolate the cross-source pass: first remove exact within-source
    # duplicates (legitimate scrape-artifact dupes, unrelated to the
    # crosslisted/new distinction) the same way dedupe() does, then check
    # what the *cross-source* pass alone removes from what's left.
    after_within = combined_for_dedupe.drop_duplicates(
        subset=["source", "slug", "shoe_size", "order_date", "sale_price"], keep="first"
    )
    base_crosslisted_idx = set(after_within.index) & set(combined.index[kind_col == "crosslisted"])
    base_new_idx = set(after_within.index) & set(combined.index[kind_col == "new"])

    result, report = dedupe(combined_for_dedupe)

    # dedupe must preserve the original index of kept rows for this to work
    kept_index = set(result.index)

    crosslisted_removed = len(base_crosslisted_idx - kept_index) / len(base_crosslisted_idx)
    new_removed = len(base_new_idx - kept_index) / len(base_new_idx) if base_new_idx else 0.0

    assert crosslisted_removed >= 0.95
    assert new_removed < 0.05
    assert report["cross_source_dupes_removed"] >= 1


# ---------------------------------------------------------------------------
# build_clean_dataset
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_dir(tmp_path, fixture_csv, stockx_sample_df):
    raw = tmp_path / "raw"
    raw.mkdir()
    stockx_dest = raw / "stockx_2019.csv"
    stockx_dest.write_bytes(fixture_csv.read_bytes())

    listings = make_listings(stockx_sample_df, n=500, seed=RANDOM_SEED)
    listings.to_csv(raw / "listings_sim.csv", index=False)
    return raw


def test_build_clean_dataset_writes_parquet_and_report(raw_dir, tmp_path):
    out = tmp_path / "processed" / "clean.parquet"
    df = build_clean_dataset(raw_dir=raw_dir, out=out)

    assert out.exists()
    report_path = out.parent / "clean_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text())
    assert "rows_in" in report
    assert "rows_out" in report
    assert "within_source_dupes_removed" in report
    assert "cross_source_dupes_removed" in report
    assert "sizes_imputed" in report

    assert (df["sale_price"] > 0).all()
    assert (df["retail_price"] > 0).all()
    assert np.issubdtype(df["order_date"].dtype, np.datetime64)
    assert np.issubdtype(df["shoe_size"].dtype, np.floating)
    assert "slug" in df.columns
    assert "_kind" not in df.columns
    assert "sku" not in df.columns
    assert "listing_id" not in df.columns
