"""Train, evaluate and persist the quantile gradient-boosting price models.

Three independent `HistGradientBoostingRegressor`s are fit with
`loss="quantile"` at q=0.10 / 0.50 / 0.90, giving a prediction *interval*
(low / mid / high) rather than a single point estimate. All three share
one preprocessing `ColumnTransformer` definition (one-hot for
`features.CATEGORICAL`, passthrough for `features.NUMERIC`).

Two modelling decisions worth spelling out for the teaching notebooks:

1. **Target transform.** The models are fit on `np.log1p(sale_price)`,
   not raw dollars. Resale prices are heavily right-skewed (a $200 shoe
   and a $4,000 shoe live in the same table), and a log target turns
   multiplicative price effects into additive ones -- which is also what
   makes the SHAP values in `explain.py` interpretable as percentages.
   Every reported metric is converted back to USD with `np.expm1` first.
2. **Temporal split, not random split.** `temporal_split` holds out the
   *last* slice of time (`SPLIT_DATE` onwards). A random split would let
   the model see sales of the same shoe on the day before and the day
   after a held-out sale, which is not the situation it faces in
   production. `evaluate` deliberately also fits a throwaway q50 model on
   a random split so the two MAEs can be compared side by side
   (`temporal_split_mae` vs `random_split_mae`); the random split looks
   better, and that gap is the point.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sneakerml.config import MODELS_DIR, RANDOM_SEED, SPLIT_DATE
from sneakerml.features import CATEGORICAL, NUMERIC, TARGET

#: Quantile of each saved model, keyed by artifact name.
QUANTILES: dict[str, float] = {"q10": 0.10, "q50": 0.50, "q90": 0.90}

#: The model's input columns, in the order the ColumnTransformer expects.
FEATURES: list[str] = CATEGORICAL + NUMERIC

#: Upper cap on boosting iterations. Early stopping (below) means the
#: models rarely run this far in practice -- it exists as a safety cap,
#: not the effective iteration count.
DEFAULT_MAX_ITER = 1000
LEARNING_RATE = 0.06
MAX_DEPTH = 6

#: Early-stopping knobs for the quantile models. Without early stopping,
#: `max_iter=400` let the q10/q90 models keep fitting past the point
#: where their calibration holds on held-out data: on the real ~91k-row
#: dataset this collapsed q10-q90 coverage to 0.596 (target 0.7-0.9)
#: even though MAPE looked fine.
#:
#: `tol` matters more here than it looks: sklearn's default (`1e-7`) is
#: so small that the internal validation pinball loss "improves" by a
#: hair almost every round on a dataset this size, so early stopping
#: never actually fires before `max_iter` -- confirmed empirically
#: (`n_iter_` came back at the `max_iter=1000` cap for all three
#: quantiles with the default tol). Raising `tol` to `1e-3` (validation
#: pinball loss is in log1p(price) units, so 1e-3 is a real, not
#: numerical-noise, plateau) lets early stopping actually trigger --
#: q10/q50/q90 then stop themselves around 90-110 iterations, which is
#: in the same regime as the manually-verified `max_iter=150` fix and
#: restores coverage to ~0.80. `n_iter_no_change=10` on a 10%
#: (`validation_fraction=0.1`) internal split gives enough rounds to
#: avoid stopping on a single unlucky iteration.
EARLY_STOPPING = True
N_ITER_NO_CHANGE = 10
VALIDATION_FRACTION = 0.1
TOL = 1e-3

#: Categories seen fewer than this many times are folded into a single
#: "infrequent" bucket by the OneHotEncoder, which keeps the one-hot
#: matrix narrow and stops rare colorways from being memorised.
MIN_CATEGORY_FREQUENCY = 20

MODEL_FILENAMES = {name: f"{name}.joblib" for name in QUANTILES}
METADATA_FILENAME = "metadata.json"


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def build_pipeline(quantile: float, max_iter: int = DEFAULT_MAX_ITER) -> Pipeline:
    """Build the (unfitted) preprocessing + quantile-GBM pipeline.

    `sparse_output=False` on the encoder is not optional:
    `HistGradientBoostingRegressor` rejects sparse input, and
    `shap.TreeExplainer` needs a dense matrix to attribute over.
    `handle_unknown="ignore"` means a colorway or buyer region never seen
    in training encodes as all-zeros instead of raising at predict time.
    """
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                    sparse_output=False,
                ),
                CATEGORICAL,
            ),
            ("num", "passthrough", NUMERIC),
        ]
    )
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=max_iter,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_SEED,
        early_stopping=EARLY_STOPPING,
        n_iter_no_change=N_ITER_NO_CHANGE,
        validation_fraction=VALIDATION_FRACTION,
        tol=TOL,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------


def temporal_split(
    df: pd.DataFrame, split_date: str = SPLIT_DATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on `order_date`: everything before `split_date` trains, the rest tests.

    This is the honest evaluation for a forecasting-flavoured problem --
    see the module docstring for why a random split flatters the model.
    """
    cutoff = pd.Timestamp(split_date)
    train_df = df[df["order_date"] < cutoff].copy()
    test_df = df[df["order_date"] >= cutoff].copy()
    return train_df, test_df


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def train_models(
    df: pd.DataFrame, max_iter: int = DEFAULT_MAX_ITER
) -> dict[str, Pipeline]:
    """Fit one pipeline per quantile on `np.log1p(sale_price)`.

    `df` must be the output of `features.add_features` (all of
    `FEATURES` plus `TARGET` present and non-null). Tests pass a small
    `max_iter` to keep the fixture fast.
    """
    X = df[FEATURES]
    y = np.log1p(df[TARGET].to_numpy(dtype=float))
    return {
        name: build_pipeline(quantile, max_iter=max_iter).fit(X, y)
        for name, quantile in QUANTILES.items()
    }


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Quantile (pinball) loss in the units of `y_true` -- here, USD.

    Under-predicting is penalised `quantile` per dollar and
    over-predicting `1 - quantile` per dollar, which is exactly what
    makes a q10 model deliberately pessimistic and a q90 model
    deliberately optimistic.
    """
    delta = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1.0) * delta)))


def _predict_usd(models: dict[str, Pipeline], X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: np.expm1(pipe.predict(X)) for name, pipe in models.items()}


def evaluate(
    models: dict[str, Pipeline],
    test_df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict:
    """Score the three models on a held-out test frame, in USD.

    Returns MAE / MAPE for the median model, the pinball loss of each
    quantile model, and the empirical `coverage` of the [q10, q90]
    interval (which should land near 0.80 if the quantile models are
    calibrated).

    When `train_df` is supplied, a throwaway q50 model is additionally
    fit on a *random* split of `train_df + test_df` (same test size,
    `random_state=RANDOM_SEED`) purely so its MAE can be reported as
    `random_split_mae` next to `temporal_split_mae`. It is a diagnostic
    number for the notebooks, never saved as an artifact; without
    `train_df` the key is present but None.
    """
    X_test = test_df[FEATURES]
    y_true = test_df[TARGET].to_numpy(dtype=float)
    preds = _predict_usd(models, X_test)

    q50 = preds["q50"]
    mae = float(np.mean(np.abs(q50 - y_true)))
    mape = float(np.mean(np.abs(q50 - y_true) / y_true))

    # sort the interval ends before measuring coverage: independently
    # trained quantile models can cross on individual rows.
    low = np.minimum(preds["q10"], preds["q90"])
    high = np.maximum(preds["q10"], preds["q90"])
    coverage = float(np.mean((y_true >= low) & (y_true <= high)))

    metrics = {
        "n_test": int(len(test_df)),
        "mae": mae,
        "mape": mape,
        "coverage": coverage,
        "crossing_rate": float(np.mean(preds["q10"] > preds["q90"])),
        "pinball_q10": pinball_loss(y_true, preds["q10"], 0.10),
        "pinball_q50": pinball_loss(y_true, preds["q50"], 0.50),
        "pinball_q90": pinball_loss(y_true, preds["q90"], 0.90),
        "temporal_split_mae": mae,
        "random_split_mae": None,
    }

    if train_df is not None and len(test_df) > 0:
        full = pd.concat([train_df, test_df], ignore_index=True)
        rand_train, rand_test = train_test_split(
            full,
            test_size=len(test_df) / len(full),
            random_state=RANDOM_SEED,
            shuffle=True,
        )
        baseline = build_pipeline(QUANTILES["q50"], max_iter=max_iter).fit(
            rand_train[FEATURES], np.log1p(rand_train[TARGET].to_numpy(dtype=float))
        )
        rand_pred = np.expm1(baseline.predict(rand_test[FEATURES]))
        rand_true = rand_test[TARGET].to_numpy(dtype=float)
        metrics["random_split_mae"] = float(np.mean(np.abs(rand_pred - rand_true)))

    return metrics


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    """Current commit sha, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):  # pragma: no cover - defensive
        return None
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else None


def _as_date_string(value) -> str | None:
    if value is None:
        return None
    return str(pd.Timestamp(value).date())


def save_artifacts(
    models: dict[str, Pipeline],
    metrics: dict,
    feature_names: dict | list,
    dir: Path = MODELS_DIR,  # noqa: A002 - name fixed by the task interface
    n_train: int | None = None,
    n_test: int | None = None,
    data_end_date=None,
    max_days_since_release: int | None = None,
) -> Path:
    """Write `q10/q50/q90.joblib` plus `metadata.json` into `dir`.

    `metadata.json` is the contract between training and everything
    downstream (`predict.PricePredictor`, the API, the notebooks). Two
    fields exist purely for the time-handling story:

    - `data_end_date`: the last `order_date` in the data. The predictor
      uses it as the default "today", because the real wall clock is
      years past the end of the dataset.
    - `max_days_since_release`: the largest `days_since_release` seen in
      the *training* split. A request beyond it is still answered, but
      flagged `extrapolated=True`.

    `feature_names` may be a dict (stored as-is, e.g.
    `{"categorical": [...], "numeric": [...]}`) or a flat list (stored
    under an `"all"` key).
    """
    dest = Path(dir)
    dest.mkdir(parents=True, exist_ok=True)

    for name, pipe in models.items():
        joblib.dump(pipe, dest / MODEL_FILENAMES[name])

    trained_at = datetime.now(timezone.utc).replace(microsecond=0)
    git_sha = _git_sha()
    version_stamp = trained_at.strftime("%Y%m%dT%H%M%SZ")
    model_version = f"{version_stamp}-{git_sha[:7]}" if git_sha else version_stamp

    features = (
        dict(feature_names)
        if isinstance(feature_names, dict)
        else {"all": list(feature_names)}
    )

    metadata = {
        "model_version": model_version,
        "trained_at": trained_at.isoformat(),
        "git_sha": git_sha,
        "n_train": int(n_train) if n_train is not None else None,
        "n_test": int(n_test) if n_test is not None else None,
        "quantiles": QUANTILES,
        "target": TARGET,
        "target_transform": "log1p",
        "features": features,
        "metrics": metrics,
        "data_end_date": _as_date_string(data_end_date),
        "max_days_since_release": (
            int(max_days_since_release) if max_days_since_release is not None else None
        ),
    }
    (dest / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
    return dest


def load_artifacts(dir: Path = MODELS_DIR) -> tuple[dict[str, Pipeline], dict]:  # noqa: A002
    """Load the three saved pipelines and `metadata.json` back from `dir`."""
    src = Path(dir)
    missing = [f for f in (*MODEL_FILENAMES.values(), METADATA_FILENAME) if not (src / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing model artifacts in {src}: {', '.join(missing)} "
            "(run `python -m sneakerml.cli train` first)"
        )
    models = {name: joblib.load(src / fn) for name, fn in MODEL_FILENAMES.items()}
    metadata = json.loads((src / METADATA_FILENAME).read_text())
    return models, metadata
