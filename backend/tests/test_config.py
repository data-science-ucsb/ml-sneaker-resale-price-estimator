"""Smoke tests for configuration."""

from pathlib import Path
from sneakerml import config


def test_root_is_backend():
    """Test that ROOT points to the backend directory."""
    assert config.ROOT.name == "backend"
    assert (config.ROOT / "src" / "sneakerml" / "config.py").exists()


def test_random_seed_is_42():
    """Test that RANDOM_SEED is correctly set to 42."""
    assert config.RANDOM_SEED == 42


def test_data_directories_under_root():
    """Test that data directories are under ROOT."""
    assert config.RAW_DIR.parent.parent == config.ROOT
    assert config.PROCESSED_DIR.parent.parent == config.ROOT
    assert config.MODELS_DIR.parent == config.ROOT


def test_db_path_under_root():
    """Test that DB_PATH is under ROOT."""
    assert config.DB_PATH.parent == config.ROOT


def test_split_date_format():
    """Test that SPLIT_DATE is a valid date string."""
    assert config.SPLIT_DATE == "2019-01-01"


def test_min_expected_rows():
    """Test that MIN_EXPECTED_ROWS is set correctly."""
    assert config.MIN_EXPECTED_ROWS == 99_000


def test_stockx_url_is_valid():
    """Test that STOCKX_URL is set."""
    assert config.STOCKX_URL.startswith("https://")
    assert "stockx" in config.STOCKX_URL.lower()
