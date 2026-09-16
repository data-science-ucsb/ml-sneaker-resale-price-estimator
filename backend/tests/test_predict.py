"""Tests for the user-facing price predictor (predict.py)."""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from sneakerml.features import CATEGORICAL, NUMERIC
from sneakerml.predict import MAX_SIZE, MIN_SIZE, PricePredictor

BELUGA_SLUG = "adidas-yeezy-boost-350-low-v2-beluga"


@pytest.fixture(scope="module")
def predictor(tiny_models) -> PricePredictor:
    return PricePredictor.load(
        models_dir=tiny_models["models_dir"],
        catalog_path=tiny_models["catalog_path"],
    )


@pytest.fixture(scope="module")
def metadata(tiny_models) -> dict:
    return json.loads((tiny_models["models_dir"] / "metadata.json").read_text())


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_predict_returns_the_documented_shape(predictor):
    result = predictor.predict(BELUGA_SLUG, 10.0)
    expected = {
        "sneaker",
        "size",
        "as_of",
        "low",
        "mid",
        "high",
        "crossed",
        "retail_price",
        "premium_pct",
        "extrapolated",
        "contributions",
        "model_version",
    }
    assert set(result) == expected
    assert result["sneaker"]["id"] == BELUGA_SLUG
    assert result["sneaker"]["display_name"]
    assert result["size"] == 10.0
    assert isinstance(result["crossed"], bool)
    assert isinstance(result["extrapolated"], bool)
    assert result["model_version"]


def test_prices_are_positive_and_ordered_for_many_shoes(predictor):
    ids = list(predictor.catalog["id"])[:20]
    assert len(ids) == 20
    for sneaker_id in ids:
        result = predictor.predict(sneaker_id, 10.0)
        assert result["low"] > 0
        assert result["low"] <= result["mid"] <= result["high"], sneaker_id


def test_premium_pct_is_relative_to_retail(predictor):
    result = predictor.predict(BELUGA_SLUG, 10.0)
    retail = result["retail_price"]
    assert retail > 0
    expected = (result["mid"] - retail) / retail * 100.0
    assert result["premium_pct"] == pytest.approx(expected)


def test_contributions_are_original_feature_names(predictor):
    result = predictor.predict(BELUGA_SLUG, 10.0)
    contributions = result["contributions"]
    assert 0 < len(contributions) <= 8
    for item in contributions:
        assert set(item) == {"feature", "value", "effect_pct"}
        assert item["feature"] in CATEGORICAL + NUMERIC
    effects = [abs(item["effect_pct"]) for item in contributions]
    assert effects == sorted(effects, reverse=True)


def test_result_is_json_serialisable(predictor):
    """The API in Task 6 will json.dumps this straight onto the wire."""
    json.dumps(predictor.predict(BELUGA_SLUG, 10.5))


# ---------------------------------------------------------------------------
# time handling
# ---------------------------------------------------------------------------


def test_as_of_defaults_to_the_data_end_date(predictor, metadata):
    result = predictor.predict(BELUGA_SLUG, 10.0)
    assert result["as_of"] == metadata["data_end_date"]
    assert predictor.data_end_date.date().isoformat() == metadata["data_end_date"]


def test_explicit_as_of_is_respected(predictor):
    result = predictor.predict(BELUGA_SLUG, 10.0, as_of=dt.date(2018, 9, 1))
    assert result["as_of"] == "2018-09-01"


def test_far_future_as_of_is_flagged_as_extrapolated(predictor, metadata):
    assert metadata["max_days_since_release"] is not None
    result = predictor.predict(BELUGA_SLUG, 10.0, as_of=dt.date(2030, 1, 1))
    assert result["extrapolated"] is True
    assert result["mid"] > 0  # still answers, just flagged


def test_in_range_as_of_is_not_extrapolated(predictor):
    """A date inside the training envelope must not raise the flag."""
    entry = predictor.catalog.set_index("id").loc[BELUGA_SLUG]
    as_of = (entry["release_date"] + pd.Timedelta(days=1)).date()
    result = predictor.predict(BELUGA_SLUG, 10.0, as_of=as_of)
    assert result["extrapolated"] is False


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


def test_unknown_sneaker_id_raises_key_error(predictor):
    with pytest.raises(KeyError):
        predictor.predict("not-a-real-sneaker", 10.0)


@pytest.mark.parametrize("size", [3.0, 2.0, 18.5, 25.0, 0.0, -1.0])
def test_size_out_of_range_raises_value_error(predictor, size):
    with pytest.raises(ValueError):
        predictor.predict(BELUGA_SLUG, size)


@pytest.mark.parametrize("size", [MIN_SIZE, 10.0, MAX_SIZE])
def test_sizes_at_the_boundary_are_accepted(predictor, size):
    assert predictor.predict(BELUGA_SLUG, size)["mid"] > 0


def test_non_numeric_size_raises_value_error(predictor):
    with pytest.raises(ValueError):
        predictor.predict(BELUGA_SLUG, "ten")


# ---------------------------------------------------------------------------
# search passthrough
# ---------------------------------------------------------------------------


def test_predictor_exposes_search(predictor):
    results = predictor.search("beluga")
    assert BELUGA_SLUG in {row["id"] for row in results}
