"""Configuration constants for the sneaker ML project."""

from pathlib import Path

# Root directory is backend/ (parents[2] means go up 2 levels from config.py)
ROOT = Path(__file__).resolve().parents[2]

# Data directories
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# Database
DB_PATH = ROOT / "watchlist.db"

# Data source
STOCKX_URL = "https://raw.githubusercontent.com/saromleang/stockx-dc19/master/StockX-Data-Contest-2019-3.csv"

# ML parameters
RANDOM_SEED = 42
SPLIT_DATE = "2019-01-01"
MIN_EXPECTED_ROWS = 99_000
