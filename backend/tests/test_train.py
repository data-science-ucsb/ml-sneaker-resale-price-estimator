"""Tests for the quantile GBM training pipeline (train.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sneakerml.data.clean import impute_size, load_stockx
from sneakerml.features import CATEGORICAL, NUMERIC, TARGET, add_features
from sneakerml.train import (
    FEATURES,
    QUANTILES,
    build_pipeline,
    evaluate,
    load_artifacts,
    save_artifacts,
    temporal_split,
    train_models,
)

FIXTURE_SPLIT_DATE = "2018-07-01"


def featured(csv_path) -> pd.DataFrame:
    """Run the fixture CSV through the real pipeline: clean -> impute -> features."""
    return add_features(impute_size(load_stockx(csv_path)))


@pytest.fixture(scope="module")
def featured_df(request) -> pd.DataFrame:
    csv_path = request.path.parent / "fixtures" / "stockx_sample.csv"
    return featured(csv_path)


@pytest.fixture(scope="module")
def split(featured_df):
    return temporal_split(featured_df, split_date=FIXTURE_SPLIT_DATE)


@pytest.fixture(scope="module")
def models(split):
    train_df, _ = split
    return train_models(train_df, max_iter=30)


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------


def test_build_pipeline_shape():
    pipe = build_pipeline(0.5, max_iter=30)
    assert isinstance(pipe, Pipeline)
    encoder = pipe.named_steps["preprocessor"].transformers[0][1]
    assert isinstance(encoder, OneHotEncoder)
    # dense output is required: HistGB and shap.TreeExplainer both need it
    assert encoder.sparse_output is False
    assert encoder.handle_unknown == "ignore"
    assert encoder.min_frequency == 20
    model = pipe.named_steps["model"]
    assert isinstance(model, HistGradientBoostingRegressor)
    assert model.loss == "quantile"
    assert model.quantile == 0.5
    assert model.max_iter == 30
    # early stopping parameters are critical: without tol=1e-3, sklearn's default
    # tol=1e-7 never triggers early stopping in practice, and the model overfits
    # (q10-q90 coverage collapses from 0.804 to 0.493 on real data, even though
    # MAPE looks fine). regression test to prevent accidental reverts.
    assert model.early_stopping is True
    assert model.n_iter_no_change == 10
    assert model.validation_fraction == 0.1
    assert model.tol == 1e-3


def test_pipeline_columns_are_leak_free():
    assert TARGET not in FEATURES
    assert "order_year" not in FEATURES
    assert "profit" not in FEATURES
    assert "days_since" not in FEATURES
    assert FEATURES == CATEGORICAL + NUMERIC


def test_build_pipeline_fits_and_predicts(featured_df):
    pipe = build_pipeline(0.5, max_iter=20)
    pipe.fit(featured_df[FEATURES], np.log1p(featured_df[TARGET]))
    preds = pipe.predict(featured_df[FEATURES].head(5))
    assert preds.shape == (5,)
    assert np.isfinite(preds).all()


# ---------------------------------------------------------------------------
# temporal_split
# ---------------------------------------------------------------------------


def test_temporal_split_is_chronological(featured_df):
    train_df, test_df = temporal_split(featured_df, split_date=FIXTURE_SPLIT_DATE)
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert len(train_df) + len(test_df) == len(featured_df)
    assert train_df["order_date"].max() < pd.Timestamp(FIXTURE_SPLIT_DATE)
    assert test_df["order_date"].min() >= pd.Timestamp(FIXTURE_SPLIT_DATE)


# ---------------------------------------------------------------------------
# train_models
# ---------------------------------------------------------------------------


def test_train_models_returns_three_quantiles(models):
    assert set(models) == {"q10", "q50", "q90"}
    for name, pipe in models.items():
        assert pipe.named_steps["model"].quantile == QUANTILES[name]


def test_predictions_are_positive_and_ordered(models, split):
    _, test_df = split
    rows = test_df.head(20)
    assert len(rows) == 20
    preds = {k: np.expm1(m.predict(rows[FEATURES])) for k, m in models.items()}
    for name, values in preds.items():
        assert (values > 0).all(), f"{name} produced a non-positive price"
    triples = np.column_stack([preds["q10"], preds["q50"], preds["q90"]])
    ordered = np.sort(triples, axis=1)
    assert (ordered[:, 0] <= ordered[:, 1]).all()
    assert (ordered[:, 1] <= ordered[:, 2]).all()
    # the quantile models should be ordered on average even if individual
    # rows cross
    assert preds["q10"].mean() <= preds["q50"].mean() <= preds["q90"].mean()


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_expected_keys(models, split):
    train_df, test_df = split
    metrics = evaluate(models, test_df, train_df=train_df, max_iter=30)
    for key in (
        "n_test",
        "mae",
        "mape",
        "coverage",
        "pinball_q10",
        "pinball_q50",
        "pinball_q90",
        "temporal_split_mae",
        "random_split_mae",
    ):
        assert key in metrics, key
    assert metrics["n_test"] == len(test_df)
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert metrics["mae"] > 0
    assert metrics["temporal_split_mae"] == pytest.approx(metrics["mae"])
    assert metrics["random_split_mae"] is not None


def test_evaluate_without_train_df_skips_random_baseline(models, split):
    _, test_df = split
    metrics = evaluate(models, test_df)
    assert metrics["random_split_mae"] is None


# ---------------------------------------------------------------------------
# save_artifacts / load_artifacts
# ---------------------------------------------------------------------------


def test_artifacts_round_trip(models, split, tmp_path):
    train_df, test_df = split
    metrics = evaluate(models, test_df)
    save_artifacts(
        models,
        metrics,
        {"categorical": CATEGORICAL, "numeric": NUMERIC, "target": TARGET},
        dir=tmp_path,
        n_train=len(train_df),
        n_test=len(test_df),
        data_end_date=test_df["order_date"].max(),
        max_days_since_release=int(train_df["days_since_release"].max()),
    )
    for name in ("q10", "q50", "q90"):
        assert (tmp_path / f"{name}.joblib").exists()
    assert (tmp_path / "metadata.json").exists()

    loaded, metadata = load_artifacts(tmp_path)
    assert set(loaded) == {"q10", "q50", "q90"}
    rows = test_df.head(20)[FEATURES]
    for name in loaded:
        np.testing.assert_allclose(
            loaded[name].predict(rows), models[name].predict(rows)
        )

    assert metadata["n_train"] == len(train_df)
    assert metadata["n_test"] == len(test_df)
    assert metadata["data_end_date"] == str(test_df["order_date"].max().date())
    assert metadata["max_days_since_release"] == int(train_df["days_since_release"].max())
    assert metadata["features"]["categorical"] == CATEGORICAL
    assert metadata["metrics"]["mae"] == pytest.approx(metrics["mae"])
    assert "trained_at" in metadata
    assert "git_sha" in metadata
    assert "model_version" in metadata

    # metadata.json is plain JSON that notebooks can read directly
    raw = json.loads((tmp_path / "metadata.json").read_text())
    assert raw["data_end_date"] == metadata["data_end_date"]
