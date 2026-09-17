"""Command-line entry points for sneakerml data acquisition tasks.

Usage:
    python -m sneakerml.cli download [--kaggle]
    python -m sneakerml.cli simulate [--n N] [--seed SEED]
    python -m sneakerml.cli clean
    python -m sneakerml.cli train [--max-iter N] [--split-date YYYY-MM-DD]
    python -m sneakerml.cli catalog
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sneakerml.catalog import DEFAULT_CATALOG_PATH, build_catalog, save_catalog
from sneakerml.config import MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, RAW_DIR, SPLIT_DATE
from sneakerml.data.clean import build_clean_dataset
from sneakerml.data.download import fetch_stockx
from sneakerml.data.simulate import make_listings
from sneakerml.explain import Explainer, save_global_importance
from sneakerml.features import CATEGORICAL, NUMERIC, TARGET, add_features
from sneakerml.train import (
    DEFAULT_MAX_ITER,
    FEATURES,
    evaluate,
    save_artifacts,
    temporal_split,
    train_models,
)

CLEAN_PATH = PROCESSED_DIR / "clean.parquet"


def _cmd_download(args: argparse.Namespace) -> None:
    dest = fetch_stockx(kaggle=args.kaggle)
    print(f"Wrote {dest}")


def _cmd_simulate(args: argparse.Namespace) -> None:
    stockx_path = RAW_DIR / "stockx_2019.csv"
    if not stockx_path.exists():
        raise SystemExit(f"{stockx_path} not found; run `sneakerml.cli download` first.")

    stockx = pd.read_csv(stockx_path, encoding="utf-8-sig")
    listings = make_listings(stockx, n=args.n, seed=args.seed)

    dest = RAW_DIR / "listings_sim.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    listings.to_csv(dest, index=False)
    print(f"Wrote {dest} ({len(listings)} rows)")


def _cmd_clean(args: argparse.Namespace) -> None:
    out = PROCESSED_DIR / "clean.parquet"
    df = build_clean_dataset(raw_dir=RAW_DIR, out=out)
    print(f"Wrote {out} ({len(df)} rows)")


def _load_clean() -> pd.DataFrame:
    if not CLEAN_PATH.exists():
        raise SystemExit(f"{CLEAN_PATH} not found; run `sneakerml.cli clean` first.")
    return pd.read_parquet(CLEAN_PATH)


def _cmd_train(args: argparse.Namespace) -> None:
    featured = add_features(_load_clean())
    train_df, test_df = temporal_split(featured, split_date=args.split_date)
    if train_df.empty or test_df.empty:
        raise SystemExit(
            f"split date {args.split_date} leaves an empty train or test set "
            f"({len(train_df)} train / {len(test_df)} test rows)."
        )

    print(f"Training on {len(train_df)} rows, testing on {len(test_df)} "
          f"(split at {args.split_date}, max_iter={args.max_iter}) ...")
    models = train_models(train_df, max_iter=args.max_iter)
    metrics = evaluate(models, test_df, train_df=train_df, max_iter=args.max_iter)

    max_days = int(train_df["days_since_release"].max())
    save_artifacts(
        models,
        metrics,
        {"categorical": CATEGORICAL, "numeric": NUMERIC, "target": TARGET},
        dir=args.models_dir,
        n_train=len(train_df),
        n_test=len(test_df),
        data_end_date=featured["order_date"].max(),
        max_days_since_release=max_days,
    )

    importance = Explainer(models["q50"]).global_importance(train_df[FEATURES])
    save_global_importance(importance, Path(args.models_dir) / "global_importance.json")

    print(f"Wrote artifacts to {args.models_dir}")
    print(f"  MAE (q50, temporal split)  ${metrics['mae']:,.2f}")
    print(f"  MAPE (q50)                 {metrics['mape'] * 100:.1f}%")
    print(f"  coverage of [q10, q90]     {metrics['coverage']:.3f}  (target ~0.80)")
    print(f"  quantile crossing rate     {metrics['crossing_rate']:.3f}")
    for q in ("q10", "q50", "q90"):
        print(f"  pinball {q}                ${metrics[f'pinball_{q}']:,.2f}")
    if metrics["random_split_mae"] is not None:
        print(f"  MAE, random split baseline ${metrics['random_split_mae']:,.2f} "
              "(optimistic -- it peeks across time)")
    print("  top features: " + ", ".join(list(importance)[:5]))


def _cmd_catalog(args: argparse.Namespace) -> None:
    catalog = build_catalog(_load_clean())
    dest = save_catalog(catalog, args.out)
    print(f"Wrote {dest} ({len(catalog)} sneakers)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sneakerml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser("download", help="Download the real StockX CSV")
    p_download.add_argument(
        "--kaggle",
        action="store_true",
        help="Use the Kaggle CLI if ~/.kaggle/kaggle.json exists",
    )
    p_download.set_defaults(func=_cmd_download)

    p_simulate = subparsers.add_parser(
        "simulate", help="Generate the simulated second-source ('goatish') listings CSV"
    )
    p_simulate.add_argument("--n", type=int, default=20_000)
    p_simulate.add_argument("--seed", type=int, default=RANDOM_SEED)
    p_simulate.set_defaults(func=_cmd_simulate)

    p_clean = subparsers.add_parser(
        "clean", help="Clean and merge the StockX and listings sources into clean.parquet"
    )
    p_clean.set_defaults(func=_cmd_clean)

    p_train = subparsers.add_parser(
        "train", help="Train, evaluate and save the q10/q50/q90 price models"
    )
    p_train.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    p_train.add_argument("--split-date", default=SPLIT_DATE)
    p_train.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    p_train.set_defaults(func=_cmd_train)

    p_catalog = subparsers.add_parser(
        "catalog", help="Build the searchable sneaker catalog JSON"
    )
    p_catalog.add_argument("--out", type=Path, default=DEFAULT_CATALOG_PATH)
    p_catalog.set_defaults(func=_cmd_catalog)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
