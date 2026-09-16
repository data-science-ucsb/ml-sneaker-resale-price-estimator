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
