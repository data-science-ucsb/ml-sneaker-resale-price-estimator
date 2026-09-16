"""Tests for the simulated second-source ("goatish") listings generator."""

import re

import pandas as pd
import pytest

from sneakerml.config import RANDOM_SEED
from sneakerml.data.download import fetch_stockx
from sneakerml.data.simulate import make_listings

EXPECTED_COLUMNS = {
    "listing_id",
    "sku",
    "title",
    "price",
    "size",
    "size_system",
    "listed_at",
    "source",
    "_kind",
}

PRICE_RE = re.compile(r"^\$[0-9,]+\.\d{2}$")


def test_make_listings_row_count_includes_exact_duplicates(stockx_sample_df):
    n = 500
    result = make_listings(stockx_sample_df, n=n, seed=RANDOM_SEED)
    expected_dupes = round(n * 0.10)
    assert len(result) == n + expected_dupes
    assert len(result) > n


def test_make_listings_columns_present(stockx_sample_df):
    result = make_listings(stockx_sample_df, n=500, seed=RANDOM_SEED)
    assert set(result.columns) == EXPECTED_COLUMNS


def test_make_listings_deterministic_with_same_seed(stockx_sample_df):
    a = make_listings(stockx_sample_df, n=500, seed=RANDOM_SEED)
    b = make_listings(stockx_sample_df, n=500, seed=RANDOM_SEED)
    pd.testing.assert_frame_equal(a, b)


def test_make_listings_different_seed_differs(stockx_sample_df):
    a = make_listings(stockx_sample_df, n=500, seed=RANDOM_SEED)
    b = make_listings(stockx_sample_df, n=500, seed=RANDOM_SEED + 1)
    assert not a["price"].equals(b["price"])


def test_make_listings_has_missing_and_eu_sizes(stockx_sample_df):
    result = make_listings(stockx_sample_df, n=500, seed=RANDOM_SEED)
    assert result["size"].isna().sum() >= 1
    assert (result["size_system"] == "EU").sum() >= 1


def test_make_listings_prices_are_dollar_strings(stockx_sample_df):
    result = make_listings(stockx_sample_df, n=500, seed=RANDOM_SEED)
    assert result["price"].str.match(PRICE_RE).all()
    assert (result["price"].str.len() >= 1).all()


def test_make_listings_kind_proportions_are_60_40(stockx_sample_df):
    n = 500
    result = make_listings(stockx_sample_df, n=n, seed=RANDOM_SEED)
    base = result.iloc[:n]  # exact duplicates are appended after the base rows
    counts = base["_kind"].value_counts(normalize=True)
    assert counts["crosslisted"] == pytest.approx(0.60, abs=0.02)
    assert counts["new"] == pytest.approx(0.40, abs=0.02)


def test_fetch_stockx_file_url_succeeds_with_low_min_rows(fixture_csv, tmp_path):
    url = fixture_csv.resolve().as_uri()
    dest = tmp_path / "stockx_2019.csv"
    result = fetch_stockx(dest=dest, url=url, min_rows=250)
    assert result == dest
    assert dest.exists()


def test_fetch_stockx_file_url_raises_with_high_min_rows(fixture_csv, tmp_path):
    url = fixture_csv.resolve().as_uri()
    dest = tmp_path / "stockx_2019.csv"
    with pytest.raises(RuntimeError):
        fetch_stockx(dest=dest, url=url, min_rows=10_000)
