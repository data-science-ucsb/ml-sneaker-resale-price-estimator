"""Pytest fixtures for sneaker ML tests."""

from pathlib import Path
import pandas as pd
import pytest


@pytest.fixture
def fixture_csv() -> Path:
    """Return the path to the StockX sample CSV fixture."""
    return Path(__file__).parent / "fixtures" / "stockx_sample.csv"


@pytest.fixture
def stockx_sample_df(fixture_csv) -> pd.DataFrame:
    """Load and return the StockX sample data as a DataFrame."""
    return pd.read_csv(fixture_csv)


#: Temporal cutoff for the tiny fixture models. The 300-row sample spans
#: 2017-09-15 to 2019-02-13, so this leaves 101 training rows and 199 test
#: rows -- both non-empty, which the real SPLIT_DATE (2019-01-01) would
#: also manage but with only 49 test rows and far fewer categories seen in
#: training.
TINY_SPLIT_DATE = "2018-07-01"
TINY_MAX_ITER = 30


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory) -> dict:
    """Train real (but tiny) artifacts from the sample CSV, once per session.

    Runs the genuine pipeline end to end -- `load_stockx` -> `impute_size`
    -> `add_features` -> `build_catalog` / `temporal_split` +
    `train_models(max_iter=30)` -> `save_artifacts` -- so anything built on
    top of it (`PricePredictor`, the API in Task 6) exercises the same code
    paths as production, just on 300 rows.

    The sample contains StockX rows only, so the simulated-listings branch
    of the cleaner is deliberately not involved.

    Yields ``{"models_dir": Path, "catalog_path": Path}``.
    """
    from sneakerml.catalog import build_catalog, save_catalog
    from sneakerml.data.clean import impute_size, load_stockx
    from sneakerml.features import CATEGORICAL, NUMERIC, TARGET, add_features
    from sneakerml.train import evaluate, save_artifacts, temporal_split, train_models

    base = tmp_path_factory.mktemp("tiny_models")
    models_dir = base / "models"
    catalog_path = base / "catalog.json"

    csv_path = Path(__file__).parent / "fixtures" / "stockx_sample.csv"
    clean_df = impute_size(load_stockx(csv_path))
    featured = add_features(clean_df)

    save_catalog(build_catalog(clean_df), catalog_path)

    train_df, test_df = temporal_split(featured, split_date=TINY_SPLIT_DATE)
    models = train_models(train_df, max_iter=TINY_MAX_ITER)
    metrics = evaluate(models, test_df, train_df=train_df, max_iter=TINY_MAX_ITER)
    save_artifacts(
        models,
        metrics,
        {"categorical": CATEGORICAL, "numeric": NUMERIC, "target": TARGET},
        dir=models_dir,
        n_train=len(train_df),
        n_test=len(test_df),
        data_end_date=featured["order_date"].max(),
        max_days_since_release=int(train_df["days_since_release"].max()),
    )

    yield {"models_dir": models_dir, "catalog_path": catalog_path}
