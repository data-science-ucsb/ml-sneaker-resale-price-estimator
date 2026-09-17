"""Tests for name parsing and feature engineering (features.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sneakerml.features import (
    CATEGORICAL,
    NUMERIC,
    TARGET,
    add_features,
    parse_sneaker_name,
)


# ---------------------------------------------------------------------------
# parse_sneaker_name
# ---------------------------------------------------------------------------


def test_parse_yeezy_350_v2():
    result = parse_sneaker_name("Adidas-Yeezy-Boost-350-V2-Beluga")
    assert result["base_brand"] == "Adidas"
    assert result["silhouette"] == "Yeezy Boost 350 V2"
    assert result["colorway"] == "Beluga"
    assert result["collab_partner"] == "Yeezy"
    assert result["is_collab"] is True


def test_parse_yeezy_700():
    result = parse_sneaker_name("Adidas-Yeezy-Boost-700-Wave-Runner")
    assert result["base_brand"] == "Adidas"
    assert result["silhouette"] == "Yeezy Boost 700"
    assert result["colorway"] == "Wave Runner"
    assert result["collab_partner"] == "Yeezy"
    assert result["is_collab"] is True


def test_parse_off_white_jordan_1():
    result = parse_sneaker_name("Air-Jordan-1-Retro-High-Off-White-Chicago")
    assert result["base_brand"] == "Nike"
    assert result["silhouette"] == "Air Jordan 1"
    assert result["colorway"] == "Chicago"
    assert result["collab_partner"] == "Off-White"
    assert result["is_collab"] is True


def test_parse_off_white_presto():
    result = parse_sneaker_name("Nike-Air-Presto-Off-White-Black-2018")
    assert result["base_brand"] == "Nike"
    assert result["silhouette"] == "Air Presto"
    assert result["colorway"] == "Black 2018"
    assert result["collab_partner"] == "Off-White"
    assert result["is_collab"] is True


def test_parse_off_white_blazer():
    result = parse_sneaker_name("Nike-Blazer-Mid-Off-White-Wolf-Grey")
    assert result["base_brand"] == "Nike"
    assert result["silhouette"] == "Blazer Mid"
    assert result["colorway"] == "Wolf Grey"
    assert result["collab_partner"] == "Off-White"
    assert result["is_collab"] is True


def test_parse_unknown_falls_back_to_other():
    result = parse_sneaker_name("New-Balance-990-Grey")
    assert result["silhouette"] == "other"
    assert result["collab_partner"] == "none"
    assert result["is_collab"] is False
    # colorway must never be null, even for an unmatched name
    assert result["colorway"]
    assert isinstance(result["colorway"], str)


def test_parse_is_case_insensitive_lowercase_adidas():
    result = parse_sneaker_name("adidas-Yeezy-Boost-350-V2-Butter")
    assert result["base_brand"] == "Adidas"
    assert result["silhouette"] == "Yeezy Boost 350 V2"
    assert result["colorway"] == "Butter"
    assert result["collab_partner"] == "Yeezy"


def test_parse_v2_does_not_fall_through_to_bare_350():
    result = parse_sneaker_name("Adidas-Yeezy-Boost-350-V2-Core-Black-Red-2017")
    assert result["silhouette"] == "Yeezy Boost 350 V2"
    assert result["silhouette"] != "350"


def test_parse_low_v2_variant():
    result = parse_sneaker_name("Adidas-Yeezy-Boost-350-Low-V2-Beluga")
    assert result["silhouette"] == "Yeezy Boost 350 Low V2"


def test_parse_low_without_v2():
    result = parse_sneaker_name("Adidas-Yeezy-Boost-350-Low-Moonrock")
    assert result["silhouette"] == "Yeezy Boost 350 Low"
    assert result["colorway"] == "Moonrock"


def test_parse_never_returns_null_colorway_or_collab_partner():
    for name in [
        "New-Balance-990-Grey",
        "Adidas-Yeezy-Boost-350-V2-Beluga",
        "Nike-Air-Force-1-Low-Off-White",
    ]:
        result = parse_sneaker_name(name)
        assert result["colorway"] not in (None, "")
        assert result["collab_partner"] not in (None, "")


# ---------------------------------------------------------------------------
# add_features
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2018-06-01", "2018-01-01", "2019-03-15"]),
            "brand": pd.array(["Yeezy", "Off-White", "Yeezy"], dtype="string"),
            "sneaker_name": pd.array(
                [
                    "Adidas-Yeezy-Boost-350-V2-Beluga",
                    "Air-Jordan-1-Retro-High-Off-White-Chicago",
                    "Adidas-Yeezy-Boost-350-Low-Moonrock",
                ],
                dtype="string",
            ),
            "slug": pd.array(
                [
                    "adidas-yeezy-boost-350-v2-beluga",
                    "air-jordan-1-retro-high-off-white-chicago",
                    "adidas-yeezy-boost-350-low-moonrock",
                ],
                dtype="string",
            ),
            "sale_price": [220.0, 1500.0, 300.0],
            "retail_price": [220.0, 190.0, 200.0],
            "release_date": pd.to_datetime(["2018-01-01", "2017-11-01", "2019-01-01"]),
            "shoe_size": [10.0, 7.5, 13.0],
            "buyer_region": pd.array(["California", None, "Texas"], dtype="string"),
            "source": pd.array(["stockx", "goatish", "stockx"], dtype="string"),
            "size_imputed": [False, True, False],
        }
    )


def test_add_features_has_all_categorical_and_numeric_columns(raw_df):
    out = add_features(raw_df)
    for col in CATEGORICAL + NUMERIC + [TARGET]:
        assert col in out.columns


def test_add_features_no_nulls_in_categorical_or_numeric(raw_df):
    out = add_features(raw_df)
    for col in CATEGORICAL + NUMERIC:
        assert out[col].isna().sum() == 0, f"{col} has nulls"


def test_add_features_days_since_release_non_negative(raw_df):
    out = add_features(raw_df)
    assert (out["days_since_release"] >= 0).all()


def test_add_features_days_since_release_clips_negative_gap():
    df = pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2018-01-01"]),
            "brand": pd.array(["Yeezy"], dtype="string"),
            "sneaker_name": pd.array(["Adidas-Yeezy-Boost-350-V2-Beluga"], dtype="string"),
            "slug": pd.array(["adidas-yeezy-boost-350-v2-beluga"], dtype="string"),
            "sale_price": [220.0],
            "retail_price": [220.0],
            "release_date": pd.to_datetime(["2019-01-01"]),  # after order_date
            "shoe_size": [10.0],
            "buyer_region": pd.array(["California"], dtype="string"),
            "source": pd.array(["stockx"], dtype="string"),
            "size_imputed": [False],
        }
    )
    out = add_features(df)
    assert out["days_since_release"].iloc[0] == 0


def test_add_features_no_order_year_column(raw_df):
    out = add_features(raw_df)
    assert "order_year" not in out.columns


def test_add_features_order_month_is_calendar_month(raw_df):
    out = add_features(raw_df)
    assert list(out["order_month"]) == [6, 1, 3]


def test_order_month_excluded_from_numeric_features():
    # order_month is computed (see test above) but deliberately excluded from
    # NUMERIC: with only ~18 months of training data, per-month buckets alias
    # with year and leak future-relative-to-training information, which was
    # shown empirically (Task 5) to wreck real held-out metrics (coverage
    # 0.173 / MAPE 34.4% with it in NUMERIC vs. 0.768 / 9.5% without).
    assert "order_month" not in NUMERIC


def test_add_features_size_bucket(raw_df):
    out = add_features(raw_df)
    assert list(out["size_bucket"]) == ["core", "small", "large"]


def test_add_features_is_half_size(raw_df):
    out = add_features(raw_df)
    assert list(out["is_half_size"]) == [False, True, False]
    assert out["is_half_size"].dtype == bool


def test_add_features_fills_missing_buyer_region(raw_df):
    out = add_features(raw_df)
    assert out["buyer_region"].iloc[1] == "unknown"


def test_add_features_retail_price_passthrough(raw_df):
    out = add_features(raw_df)
    assert list(out["retail_price"]) == [220.0, 190.0, 200.0]


def test_add_features_collab_partner_never_null(raw_df):
    out = add_features(raw_df)
    assert out["collab_partner"].isna().sum() == 0
    assert set(out["collab_partner"]) == {"Yeezy", "Off-White"}
