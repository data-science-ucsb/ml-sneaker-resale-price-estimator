"""Tests for the `train` / `catalog` CLI subcommands (cli.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from sneakerml import cli
from sneakerml.data.clean import impute_size, load_stockx
from sneakerml.train import DEFAULT_MAX_ITER, load_artifacts


def test_parser_exposes_train_and_catalog():
    parser = cli.build_parser()
    args = parser.parse_args(["train"])
    assert args.func is cli._cmd_train
    assert args.max_iter == DEFAULT_MAX_ITER
    assert parser.parse_args(["catalog"]).func is cli._cmd_catalog


def test_train_overrides_are_parsed():
    args = cli.build_parser().parse_args(
        ["train", "--max-iter", "5", "--split-date", "2018-07-01"]
    )
    assert args.max_iter == 5
    assert args.split_date == "2018-07-01"


@pytest.fixture
def clean_parquet(fixture_csv, tmp_path, monkeypatch):
    """Point the CLI at a tiny clean.parquet built from the sample CSV."""
    path = tmp_path / "clean.parquet"
    impute_size(load_stockx(fixture_csv)).to_parquet(path, index=False)
    monkeypatch.setattr(cli, "CLEAN_PATH", path)
    return path


def test_train_command_writes_artifacts(clean_parquet, tmp_path, capsys):
    models_dir = tmp_path / "models"
    cli.main(
        [
            "train",
            "--max-iter",
            "5",
            "--split-date",
            "2018-07-01",
            "--models-dir",
            str(models_dir),
        ]
    )
    models, metadata = load_artifacts(models_dir)
    assert set(models) == {"q10", "q50", "q90"}
    assert metadata["data_end_date"] == "2019-02-13"
    assert metadata["max_days_since_release"] > 0
    assert (models_dir / "global_importance.json").exists()

    out = capsys.readouterr().out
    assert "coverage of [q10, q90]" in out
    assert "MAPE" in out


def test_catalog_command_writes_json(clean_parquet, tmp_path, capsys):
    out_path = tmp_path / "catalog.json"
    cli.main(["catalog", "--out", str(out_path)])
    records = json.loads(out_path.read_text())
    assert len(records) == 41
    assert "Wrote" in capsys.readouterr().out


def test_train_refuses_a_split_that_empties_a_side(clean_parquet, tmp_path):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "train",
                "--max-iter",
                "5",
                "--split-date",
                "2030-01-01",
                "--models-dir",
                str(tmp_path / "models"),
            ]
        )


def test_missing_clean_parquet_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "CLEAN_PATH", tmp_path / "nope.parquet")
    with pytest.raises(SystemExit, match="run `sneakerml.cli clean` first"):
        cli.main(["catalog", "--out", str(tmp_path / "catalog.json")])
