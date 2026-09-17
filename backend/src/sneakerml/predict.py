"""`PricePredictor`: the user-facing "what is this shoe worth?" entry point.

Loads the three saved quantile models, their `metadata.json` and the shoe
catalog, then turns a `(sneaker_id, size, as_of)` request into a priced,
explained answer. Everything the API and the frontend need comes out of
`predict()` in one dict.

Three things here exist because of how this particular dataset behaves:

- **"Today" is 2019.** The data ends in February 2019, so the wall clock
  is years outside the training window. `as_of` therefore defaults to
  `metadata["data_end_date"]`, never `date.today()`.
- **Extrapolation is flagged, not blocked.** If the requested date puts
  the shoe further past its release than anything in the training split
  (`metadata["max_days_since_release"]`), the models still return a
  number -- gradient-boosted trees simply flat-line past the edge of
  their training data -- and the answer carries `extrapolated=True` so
  the UI can say so.
- **Quantiles can cross.** The q10/q50/q90 models are trained
  independently and nothing forces q10 <= q50 <= q90 on a given row. The
  three predictions are sorted into `low`/`mid`/`high`, and `crossed`
  records whether that sort actually had to change anything.

The feature row handed to the models is built by running a one-row clean
frame through `features.add_features`, exactly as training does, rather
than recomputing `days_since_release` / `size_bucket` / ... by hand --
so training and serving cannot drift apart.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from sneakerml.catalog import DEFAULT_CATALOG_PATH, load_catalog, search as catalog_search
from sneakerml.config import MODELS_DIR
from sneakerml.data.clean import STOCKX_SOURCE
from sneakerml.explain import Explainer
from sneakerml.features import TARGET, add_features
from sneakerml.train import FEATURES, load_artifacts

#: Accepted US shoe sizes (inclusive). Outside this range is a caller bug,
#: not an extrapolation: no such sneaker exists.
MIN_SIZE = 3.5
MAX_SIZE = 18.0

#: Buyer region used for a prediction. The user is asking "what is this
#: shoe worth?", not "what is it worth in Ohio?", so a region has to be
#: assumed -- California is the most common buyer region in the StockX
#: data, which makes it the least surprising default.
DEFAULT_BUYER_REGION = "California"

#: Marketplace assumed for a prediction (the real, non-simulated source).
DEFAULT_SOURCE = STOCKX_SOURCE


class PricePredictor:
    """Price a sneaker at a given size and date, with an explanation."""

    def __init__(self, models: dict, metadata: dict, catalog: pd.DataFrame):
        self.models = models
        self.metadata = metadata
        self.catalog = catalog
        self._by_id = catalog.set_index("id")
        self.model_version = metadata.get("model_version", "unknown")
        self.data_end_date = pd.Timestamp(metadata["data_end_date"])
        self.max_days_since_release = metadata.get("max_days_since_release")
        self.explainer = Explainer(models["q50"])

    # -- construction -------------------------------------------------------

    @classmethod
    def load(
        cls,
        models_dir: Path = MODELS_DIR,
        catalog_path: Path = DEFAULT_CATALOG_PATH,
    ) -> "PricePredictor":
        """Load models + metadata + catalog from disk."""
        models, metadata = load_artifacts(models_dir)
        catalog = load_catalog(catalog_path)
        return cls(models, metadata, catalog)

    # -- helpers ------------------------------------------------------------

    def _entry(self, sneaker_id: str) -> pd.Series:
        try:
            return self._by_id.loc[sneaker_id]
        except KeyError:
            raise KeyError(f"unknown sneaker_id: {sneaker_id!r}") from None

    @staticmethod
    def _validated_size(size) -> float:
        try:
            value = float(size)
        except (TypeError, ValueError):
            raise ValueError(f"size must be a number, got {size!r}") from None
        if not np.isfinite(value) or not (MIN_SIZE <= value <= MAX_SIZE):
            raise ValueError(
                f"size {value} is outside the supported range {MIN_SIZE}-{MAX_SIZE}"
            )
        return value

    def _as_of_timestamp(self, as_of) -> pd.Timestamp:
        if as_of is None:
            return self.data_end_date
        return pd.Timestamp(as_of).normalize()

    def _feature_row(
        self, entry: pd.Series, size: float, as_of: pd.Timestamp
    ) -> pd.DataFrame:
        """One-row frame in the *clean* schema, pushed through `add_features`.

        Reusing `add_features` is the point: `days_since_release`,
        `order_month`, `size_bucket`, `is_half_size` and the parsed name
        fields are then computed by the same code that built the training
        set.
        """
        row = pd.DataFrame(
            [
                {
                    "order_date": as_of,
                    "brand": entry["brand"],
                    "sneaker_name": entry["sneaker_name"],
                    "slug": entry.name,
                    TARGET: np.nan,  # unknown -- that is what we are predicting
                    "retail_price": float(entry["retail_price"]),
                    "release_date": pd.Timestamp(entry["release_date"]),
                    "shoe_size": size,
                    "buyer_region": DEFAULT_BUYER_REGION,
                    "source": DEFAULT_SOURCE,
                    "size_imputed": False,
                }
            ]
        )
        return add_features(row)

    # -- main entry point ---------------------------------------------------

    def predict(
        self, sneaker_id: str, size: float, as_of: dt.date | None = None
    ) -> dict:
        """Price one shoe/size/date.

        Raises `KeyError` for an unknown `sneaker_id` and `ValueError`
        for a size outside 3.5-18. `as_of` defaults to the dataset's last
        order date (see the module docstring); a date beyond the training
        recency envelope still returns a price, flagged
        `extrapolated=True`.
        """
        entry = self._entry(sneaker_id)
        size = self._validated_size(size)
        as_of_ts = self._as_of_timestamp(as_of)

        featured = self._feature_row(entry, size, as_of_ts)
        X = featured[FEATURES]

        q10, q50, q90 = (
            float(np.expm1(self.models[name].predict(X)[0]))
            for name in ("q10", "q50", "q90")
        )
        crossed = not (q10 <= q50 <= q90)
        low, mid, high = sorted([q10, q50, q90])

        days_since_release = int(featured["days_since_release"].iloc[0])
        extrapolated = (
            self.max_days_since_release is not None
            and days_since_release > self.max_days_since_release
        )

        retail_price = float(entry["retail_price"])
        premium_pct = (mid - retail_price) / retail_price * 100.0 if retail_price else None

        return {
            "sneaker": self._sneaker_record(entry),
            "size": size,
            "as_of": as_of_ts.date().isoformat(),
            "low": low,
            "mid": mid,
            "high": high,
            "crossed": bool(crossed),
            "retail_price": retail_price,
            "premium_pct": premium_pct,
            "extrapolated": bool(extrapolated),
            "contributions": self.explainer.contributions(X),
            "model_version": self.model_version,
        }

    # -- catalog passthrough ------------------------------------------------

    def search(self, q: str, limit: int = 10) -> list[dict]:
        """Catalog search, as JSON-ready records (for the API/frontend)."""
        results = catalog_search(self.catalog, q, limit=limit)
        return [self._sneaker_record(row) for _, row in results.set_index("id").iterrows()]

    @staticmethod
    def _sneaker_record(entry: pd.Series) -> dict:
        """Catalog row -> plain JSON-serialisable dict."""
        return {
            "id": str(entry.name),
            "sku": str(entry["sku"]),
            "sneaker_name": str(entry["sneaker_name"]),
            "display_name": str(entry["display_name"]),
            "brand": str(entry["brand"]),
            "silhouette": str(entry["silhouette"]),
            "colorway": str(entry["colorway"]),
            "retail_price": float(entry["retail_price"]),
            "release_date": str(pd.Timestamp(entry["release_date"]).date()),
            "n_sales": int(entry["n_sales"]),
            "median_sale": float(entry["median_sale"]),
            "sizes_seen": [float(s) for s in entry["sizes_seen"]],
        }
