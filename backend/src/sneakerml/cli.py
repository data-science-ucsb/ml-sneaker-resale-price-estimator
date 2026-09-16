"""Command-line entry points for sneakerml data acquisition tasks.

Usage:
    python -m sneakerml.cli download [--kaggle]
    python -m sneakerml.cli simulate [--n N] [--seed SEED]
    python -m sneakerml.cli clean
"""

from __future__ import annotations

import argparse

import pandas as pd

from sneakerml.config import PROCESSED_DIR, RANDOM_SEED, RAW_DIR
from sneakerml.data.clean import build_clean_dataset
from sneakerml.data.download import fetch_stockx
from sneakerml.data.simulate import make_listings


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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
