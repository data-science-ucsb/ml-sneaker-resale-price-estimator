"""Tests for the SHAP explanation wrapper (explain.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sneakerml.data.clean import impute_size, load_stockx
from sneakerml.explain import Explainer, save_global_importance
from sneakerml.features import CATEGORICAL, NUMERIC, add_features
from sneakerml.train import FEATURES, build_pipeline, temporal_split, train_models


@pytest.fixture(scope="module")
def featured_df(request) -> pd.DataFrame:
    csv_path = request.path.parent / "fixtures" / "stockx_sample.csv"
    return add_features(impute_size(load_stockx(csv_path)))


@pytest.fixture(scope="module")
def q50(featured_df):
    train_df, _ = temporal_split(featured_df, split_date="2018-07-01")
    return train_models(train_df, max_iter=30)["q50"]


@pytest.fixture(scope="module")
def explainer(q50):
    return Explainer(q50)


# ---------------------------------------------------------------------------
# feature-name mapping
# ---------------------------------------------------------------------------


def test_transformed_columns_map_back_to_original_columns(explainer, q50, featured_df):
    owners = explainer.feature_owners
    n_transformed = q50.named_steps["preprocessor"].transform(featured_df[FEATURES]).shape[1]
    assert len(owners) == n_transformed
    assert set(owners) <= set(FEATURES)
    # every original column owns at least one transformed column
    assert set(owners) == set(FEATURES)


# ---------------------------------------------------------------------------
# contributions
# ---------------------------------------------------------------------------


def test_contributions_shape_and_names(explainer, featured_df):
    result = explainer.contributions(featured_df[FEATURES].head(1))
    assert isinstance(result, list)
    assert 0 < len(result) <= 8
    for item in result:
        assert set(item) == {"feature", "value", "effect_pct"}
        assert item["feature"] in CATEGORICAL + NUMERIC
        # original column names only -- never "cat__silhouette_Yeezy Boost 350 V2"
        assert not item["feature"].startswith(("cat__", "num__"))
        assert np.isfinite(item["effect_pct"])
    features = [item["feature"] for item in result]
    assert len(features) == len(set(features)), "one entry per original column"


def test_contributions_sorted_by_absolute_effect(explainer, featured_df):
    result = explainer.contributions(featured_df[FEATURES].head(1))
    effects = [abs(item["effect_pct"]) for item in result]
    assert effects == sorted(effects, reverse=True)


def test_contributions_values_come_from_the_input_row(explainer, featured_df):
    row = featured_df[FEATURES].head(1)
    result = explainer.contributions(row)
    for item in result:
        expected = row.iloc[0][item["feature"]]
        if isinstance(expected, (bool, np.bool_)):
            assert item["value"] == bool(expected)
        elif isinstance(expected, (int, float, np.number)):
            assert item["value"] == pytest.approx(float(expected))
        else:
            assert item["value"] == expected


def test_contributions_rejects_empty_frame(explainer, featured_df):
    with pytest.raises(ValueError):
        explainer.contributions(featured_df[FEATURES].head(0))


def test_contributions_top_n_is_configurable(explainer, featured_df):
    result = explainer.contributions(featured_df[FEATURES].head(1), top_n=3)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# global_importance
# ---------------------------------------------------------------------------


def test_global_importance_covers_all_features(explainer, featured_df):
    importance = explainer.global_importance(featured_df[FEATURES], sample=50)
    assert set(importance) == set(FEATURES)
    assert all(v >= 0 for v in importance.values())
    assert list(importance.values()) == sorted(importance.values(), reverse=True)


def test_save_global_importance_writes_json(explainer, featured_df, tmp_path):
    importance = explainer.global_importance(featured_df[FEATURES], sample=50)
    path = save_global_importance(importance, tmp_path / "global_importance.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert set(payload) == set(FEATURES)


def test_explainer_rejects_a_non_tree_final_step():
    with pytest.raises(TypeError):
        Explainer(build_pipeline(0.5, max_iter=5).named_steps["preprocessor"])
