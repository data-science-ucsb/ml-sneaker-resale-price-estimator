"""Name parsing and feature engineering for the sneaker resale price model.

`parse_sneaker_name` turns a hyphenated StockX-style name (e.g.
"Adidas-Yeezy-Boost-350-V2-Beluga") into structured fields (base brand,
silhouette, colorway, collab partner). `add_features` applies that parser
to a cleaned dataframe (the output of `sneakerml.data.clean.build_clean_dataset`)
and adds the remaining model-ready features (recency, seasonality, size
buckets, ...).

Two columns are defensively guaranteed to never be null in the output of
`add_features`, because `sklearn.OneHotEncoder` chokes on NaN:
`collab_partner` (falls back to the literal string "none") and `colorway`
(falls back to "unknown"). `buyer_region` is genuinely NaN for every
simulated/"goatish" row upstream (buyer region isn't derivable for a
simulated listing) -- that gap is filled here with the sentinel "unknown"
rather than "fixed" in clean.py.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

CATEGORICAL = [
    "brand",
    "base_brand",
    "silhouette",
    "collab_partner",
    "colorway",
    "buyer_region",
    "size_bucket",
    "source",
]
# `order_month` is deliberately EXCLUDED from NUMERIC even though `add_features`
# still computes and adds the column below. The training data only spans ~18
# months (Aug 2017 - Feb 2019), so calendar months 1-7 (Jan-Jul) each occur in
# exactly one year (2018) in the data. A model trained on `order_month` learns
# "January ~= $385" from 2018 and wrongly extrapolates that price level onto
# January 2019 in the test split -- an implicit year-proxy, exactly the
# leakage trap that omitting `order_year` was meant to avoid in the first
# place. This was verified empirically in Task 5's real-data training run:
# with `order_month` in NUMERIC, MAE $118.58 / MAPE 34.4% / coverage 0.173;
# without it, MAE $42.06 / MAPE 9.5% / coverage 0.768. The column is kept in
# the output of `add_features` anyway -- it's a useful teaching artifact for
# a later notebook task that demonstrates this exact seasonality/leakage-proxy
# phenomenon via an ablation.
NUMERIC = [
    "shoe_size",
    "retail_price",
    "days_since_release",
    "is_half_size",
    "size_imputed",
]
TARGET = "sale_price"

# Sentinels -- must be non-null strings so OneHotEncoder never sees a NaN.
NO_COLLAB = "none"
UNKNOWN_COLORWAY = "unknown"
UNKNOWN_REGION = "unknown"
OTHER_SILHOUETTE = "other"
OTHER_BRAND = "other"

_KNOWN_BRANDS = {"nike", "adidas"}

# Tokens stripped out of the remainder text before it becomes `colorway`,
# in addition to whatever the matched silhouette pattern already consumed.
_COLLAB_TOKEN_RE = re.compile(r"off[\s-]*white|virgil|abloh", re.IGNORECASE)

# Ordered silhouette regex table: more specific patterns MUST come before
# more general ones (e.g. "yeezy boost 350 low v2" before "... 350 v2"
# before the bare "350"), otherwise a specific shoe falls through to a
# less specific label. Matched case-insensitively against the normalized
# (hyphens -> spaces) `sneaker_name`, not `slug`.
SILHOUETTE_PATTERNS: list[tuple[str, str]] = [
    (r"air jordan 1(?:\s+retro\s+high)?", "Air Jordan 1"),
    (r"air max 90", "Air Max 90"),
    (r"air max 97", "Air Max 97"),
    (r"air presto", "Air Presto"),
    (r"blazer mid", "Blazer Mid"),
    (r"air vapormax", "Air VaporMax"),
    (r"zoom fly", "Zoom Fly"),
    (r"react hyperdunk", "React Hyperdunk"),
    (r"hyperdunk", "Hyperdunk"),
    (r"air force 1(?:\s+low)?", "Air Force 1"),
    (r"yeezy boost 350 low v2", "Yeezy Boost 350 Low V2"),
    (r"yeezy boost 350 v2", "Yeezy Boost 350 V2"),
    (r"yeezy boost 350 low", "Yeezy Boost 350 Low"),
    (r"yeezy boost 700", "Yeezy Boost 700"),
    (r"yeezy boost 750", "Yeezy Boost 750"),
    (r"yeezy boost 500", "Yeezy Boost 500"),
    (r"yeezy boost 350", "Yeezy Boost 350"),
    (r"\b350\b", "350"),
    (r"\b700\b", "700"),
    (r"\b750\b", "750"),
    (r"\b500\b", "500"),
]


def _detect_collab_partner(lower_name: str) -> str:
    if re.search(r"off[\s-]*white", lower_name):
        return "Off-White"
    if re.search(r"\byeezy\b", lower_name):
        return "Yeezy"
    return NO_COLLAB


def _detect_base_brand(first_word: str, collab_partner: str) -> str:
    if collab_partner == "Yeezy":
        return "Adidas"
    if collab_partner == "Off-White":
        return "Nike"
    if first_word.lower() in _KNOWN_BRANDS:
        return first_word.capitalize()
    return OTHER_BRAND


def _detect_silhouette(lower_name: str) -> tuple[str, tuple[int, int] | None]:
    for pattern, label in SILHOUETTE_PATTERNS:
        match = re.search(pattern, lower_name)
        if match:
            return label, match.span()
    return OTHER_SILHOUETTE, None


def _build_colorway(normalized: str, silhouette_span: tuple[int, int] | None, first_word: str) -> str:
    chars = list(normalized)

    def blank(start: int, end: int) -> None:
        for i in range(start, min(end, len(chars))):
            chars[i] = " "

    if silhouette_span is not None:
        blank(*silhouette_span)

    for match in _COLLAB_TOKEN_RE.finditer(normalized):
        blank(*match.span())

    if first_word.lower() in _KNOWN_BRANDS:
        blank(0, len(first_word))

    remainder_words = [w for w in "".join(chars).split() if w]
    if not remainder_words:
        return UNKNOWN_COLORWAY
    return " ".join(w.title() for w in remainder_words)


def parse_sneaker_name(name: str) -> dict:
    """Parse a hyphenated `sneaker_name` into structured catalog fields.

    >>> parse_sneaker_name("Adidas-Yeezy-Boost-350-V2-Beluga")
    {'base_brand': 'Adidas', 'silhouette': 'Yeezy Boost 350 V2',
     'colorway': 'Beluga', 'collab_partner': 'Yeezy', 'is_collab': True}

    Matching is case-insensitive (a real row in the dataset spells it
    "adidas-Yeezy-Boost-350-V2-Butter" with a lowercase "adidas") and the
    silhouette table is ordered so more specific patterns ("... 350 V2")
    are tried before more general ones (bare "350"). `colorway` and
    `collab_partner` are never null -- unmatched cases fall back to the
    sentinels "unknown" and "none" respectively, since these fields feed
    a OneHotEncoder downstream that cannot handle NaN.
    """
    normalized = re.sub(r"\s+", " ", str(name).replace("-", " ")).strip()
    lower = normalized.lower()
    first_word = normalized.split()[0] if normalized else ""

    collab_partner = _detect_collab_partner(lower)
    is_collab = collab_partner != NO_COLLAB
    base_brand = _detect_base_brand(first_word, collab_partner)
    silhouette, span = _detect_silhouette(lower)
    colorway = _build_colorway(normalized, span, first_word)

    return {
        "base_brand": base_brand,
        "silhouette": silhouette,
        "colorway": colorway,
        "collab_partner": collab_partner,
        "is_collab": is_collab,
    }


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model-ready features to a cleaned dataframe.

    Adds `base_brand`, `silhouette`, `colorway`, `collab_partner`,
    `is_collab` (from `parse_sneaker_name`), `days_since_release`
    (order - release, clipped >= 0), `order_month` (calendar month 1-12;
    computed here but deliberately excluded from `NUMERIC` -- see the
    comment above that constant), `size_bucket`, and
    `is_half_size`. `retail_price` already exists on the input and is
    passed through unchanged. `buyer_region` is filled with the sentinel
    "unknown" wherever it's missing (all simulated/"goatish" rows) since
    it's a CATEGORICAL column and must never be null.
    """
    out = df.copy()

    parsed = pd.DataFrame(
        list(out["sneaker_name"].map(parse_sneaker_name)),
        index=out.index,
    )
    out["base_brand"] = parsed["base_brand"]
    out["silhouette"] = parsed["silhouette"]
    out["colorway"] = parsed["colorway"]
    out["collab_partner"] = parsed["collab_partner"]
    out["is_collab"] = parsed["is_collab"]

    out["days_since_release"] = (out["order_date"] - out["release_date"]).dt.days.clip(lower=0)
    out["order_month"] = out["order_date"].dt.month.astype(int)

    out["size_bucket"] = np.select(
        [out["shoe_size"] < 8, out["shoe_size"] > 12],
        ["small", "large"],
        default="core",
    )

    out["is_half_size"] = (out["shoe_size"] % 1 != 0).astype(bool)

    out["buyer_region"] = out["buyer_region"].fillna(UNKNOWN_REGION)

    return out
