"""SHAP explanations for the quantile price models, in original-column terms.

`shap.TreeExplainer` attributes a prediction over the *transformed*
matrix -- one column per one-hot level, e.g. 40-odd separate columns all
derived from `silhouette`. Nobody wants to read that. `Explainer` sums
the SHAP values of every transformed column back onto the original
feature it came from, so the output talks about `silhouette`,
`retail_price` and `days_since_release`.

**How the mapping is derived.** Not by string-splitting
`get_feature_names_out()` (category values legitimately contain spaces
and underscores -- "Air Jordan 1", "x_1" -- so the split is ambiguous).
Instead `_feature_owners` walks the fitted `ColumnTransformer`'s own
`transformers_` structure and counts how many output columns each input
column produces: `len(categories_[i])` for a one-hot column, minus the
categories folded into the `min_frequency` "infrequent" bucket, plus one
for that bucket when it exists; exactly one for a passthrough column.
The result is length-checked against `get_feature_names_out()`, so a
future change to the pipeline fails loudly instead of silently
mis-attributing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from sneakerml.config import MODELS_DIR, RANDOM_SEED

TOP_N_CONTRIBUTIONS = 8
GLOBAL_IMPORTANCE_SAMPLE = 2_000
DEFAULT_IMPORTANCE_PATH = MODELS_DIR / "global_importance.json"


def _feature_owners(preprocessor: ColumnTransformer) -> list[str]:
    """Original column name behind each column of the transformed matrix."""
    owners: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or transformer is None:
            continue
        columns = list(columns)
        # sklearn rewrites the literal "passthrough" into an identity
        # FunctionTransformer when the ColumnTransformer is fitted, so both
        # spellings have to be recognised here.
        if transformer == "passthrough" or (
            isinstance(transformer, FunctionTransformer) and transformer.func is None
        ):
            owners.extend(columns)
            continue
        if isinstance(transformer, OneHotEncoder):
            if transformer.drop is not None:  # pragma: no cover - not used here
                raise NotImplementedError(
                    "feature attribution does not support OneHotEncoder(drop=...)"
                )
            infrequent = transformer.infrequent_categories_
            for index, (column, categories) in enumerate(
                zip(columns, transformer.categories_)
            ):
                folded = infrequent[index] if infrequent is not None else None
                n_out = len(categories)
                if folded is not None and len(folded) > 0:
                    n_out = n_out - len(folded) + 1
                owners.extend([column] * n_out)
            continue
        raise NotImplementedError(  # pragma: no cover - defensive
            f"unsupported transformer {name!r} of type {type(transformer).__name__}"
        )

    expected = len(preprocessor.get_feature_names_out())
    if len(owners) != expected:
        raise RuntimeError(
            f"feature-owner mapping produced {len(owners)} entries for "
            f"{expected} transformed columns; the pipeline changed shape"
        )
    return owners


class Explainer:
    """SHAP wrapper around one fitted `train.build_pipeline` pipeline.

    Wraps `shap.TreeExplainer(pipeline[-1])` and handles the two things
    the raw explainer does not: running the input through the fitted
    preprocessor, and folding per-one-hot-level attributions back onto
    the original feature names.
    """

    def __init__(self, pipeline: Pipeline):
        if not isinstance(pipeline, Pipeline):
            raise TypeError(
                f"Explainer expects a fitted sklearn Pipeline, got {type(pipeline).__name__}"
            )
        self.pipeline = pipeline
        self.preprocessor: ColumnTransformer = pipeline[:-1].named_steps["preprocessor"]
        self.model = pipeline[-1]
        self.feature_owners: list[str] = _feature_owners(self.preprocessor)
        #: original input columns, in pipeline order
        self.input_features: list[str] = list(dict.fromkeys(self.feature_owners))
        self.explainer = shap.TreeExplainer(self.model)

    # -- core ---------------------------------------------------------------

    def shap_by_column(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values summed per original column, one row per row of `X`.

        Values are in the model's units -- log1p(USD) -- because that is
        the space the models are fit in.
        """
        transformed = self.preprocessor.transform(X)
        values = np.asarray(self.explainer.shap_values(transformed), dtype=float)
        if values.ndim == 3:  # pragma: no cover - single-output model here
            values = values[..., 0]
        frame = pd.DataFrame(values, columns=self.feature_owners)
        # columns repeat (one per one-hot level); summing like-named
        # columns is the aggregation step
        return frame.T.groupby(level=0).sum().T[self.input_features]

    # -- local --------------------------------------------------------------

    def contributions(
        self, X: pd.DataFrame, top_n: int = TOP_N_CONTRIBUTIONS
    ) -> list[dict]:
        """Explain one prediction: the `top_n` most influential features.

        `X` is a feature frame as handed to the pipeline; only its first
        row is explained (`PricePredictor` passes exactly one row).
        Returns `{"feature", "value", "effect_pct"}` dicts sorted by
        descending absolute effect, where `feature` is an *original*
        column name and `value` is that column's value in the row.

        `effect_pct = expm1(shap) * 100` reads as "this feature moved the
        price about N %". That reading is exact only for small effects:
        because the model is additive in log1p space, individual
        percentage effects compose multiplicatively and do not sum to the
        total change. It is an approximate, teaching-oriented decomposition,
        not an exact price attribution.
        """
        if len(X) == 0:
            raise ValueError("contributions() needs at least one row to explain")

        row = X.iloc[[0]]
        shap_row = self.shap_by_column(row).iloc[0]

        items = []
        for feature, shap_value in shap_row.items():
            value = row.iloc[0][feature]
            if isinstance(value, (bool, np.bool_)):
                value = bool(value)
            elif isinstance(value, (int, float, np.number)):
                value = float(value)
            else:
                value = str(value)
            items.append(
                {
                    "feature": str(feature),
                    "value": value,
                    "effect_pct": float(np.expm1(shap_value) * 100.0),
                }
            )

        items.sort(key=lambda item: abs(item["effect_pct"]), reverse=True)
        return items[:top_n]

    # -- global -------------------------------------------------------------

    def global_importance(
        self,
        X: pd.DataFrame,
        sample: int = GLOBAL_IMPORTANCE_SAMPLE,
        random_state: int = RANDOM_SEED,
    ) -> dict[str, float]:
        """Mean |SHAP| per original column over a random sample of `X`.

        Sampling keeps this cheap on the full ~90k-row dataset; the
        result is the "which features matter overall" chart in the
        notebooks. Values stay in log1p space (they are magnitudes, and
        only their relative size is meaningful).
        """
        if len(X) == 0:
            raise ValueError("global_importance() needs at least one row")
        if sample and len(X) > sample:
            X = X.sample(n=sample, random_state=random_state)
        mean_abs = self.shap_by_column(X).abs().mean()
        ordered = mean_abs.sort_values(ascending=False)
        return {str(k): float(v) for k, v in ordered.items()}


def save_global_importance(
    importance: dict[str, float], path: Path = DEFAULT_IMPORTANCE_PATH
) -> Path:
    """Write a `global_importance` dict to JSON alongside the models."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(importance, indent=2) + "\n")
    return dest
