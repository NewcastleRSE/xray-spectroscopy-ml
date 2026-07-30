# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   ...

"""Parametrised dry-run tests: train then infer for every model pair."""


import logging
from pathlib import Path

import pytest

from xanesnet import infer as infer_cli
from xanesnet import train as train_cli

from .conftest import collect_model_pairs, find_checkpoint

MODEL_PAIRS = collect_model_pairs()
BASIC_PAIRS = [(t, i) for t, i in MODEL_PAIRS if "ensemble" not in t.stem]
ENSEMBLE_PAIRS = [(t, i) for t, i in MODEL_PAIRS if "ensemble" in t.stem]
BASIC_IDS = [p[0].stem for p in BASIC_PAIRS]
ENSEMBLE_IDS = [p[0].stem for p in ENSEMBLE_PAIRS]


def _run_train(train_path: Path, out_dir: Path) -> Path:
    """Run training via the CLI entry point; return path to the run directory.

    Args:
        train_path: Path to a train-mode YAML config.
        out_dir: Parent directory for the training run output.

    Returns:
        Path to the created run directory (inside *out_dir*).
    """
    train_cli.main([
        "-i", str(train_path),
        "-o", str(out_dir),
        "-n", "test",
        "--yes",
    ])
    run_dirs = sorted(out_dir.glob("train_test_*"))
    assert run_dirs, f"No run directory created under {out_dir}"
    return run_dirs[-1]


def _run_infer(infer_path: Path, checkpoint_path: Path, out_dir: Path) -> Path:
    """Run inference via the CLI entry point; return path to the run directory.

    Args:
        infer_path: Path to a user-facing infer-mode YAML config.
        checkpoint_path: Path to a deployment checkpoint (``final.pth``).
        out_dir: Parent directory for the inference run output.

    Returns:
        Path to the created run directory (inside *out_dir*).
    """
    infer_cli.main([
        "-i", str(infer_path),
        "-m", str(checkpoint_path),
        "-o", str(out_dir),
        "-n", "test",
        "--yes",
    ])
    run_dirs = sorted(out_dir.glob("infer_test_*"))
    assert run_dirs, f"No run directory created under {out_dir}"
    return run_dirs[-1]


@pytest.mark.slow
@pytest.mark.parametrize("train_path, infer_path", BASIC_PAIRS, ids=BASIC_IDS)
def test_train_and_infer(train_path: Path, infer_path: Path, tmp_path: Path) -> None:
    """Train a model and run basic inference with the resulting checkpoint.

    Args:
        train_path: Path to a train-mode YAML config.
        infer_path: Path to a user-facing infer-mode YAML config.
        tmp_path: Pytest temporary directory (auto-cleaned).
    """
    logging.info("Dry run: %s", train_path.stem)

    train_run_dir = _run_train(train_path, tmp_path / "train")
    ckpt_path = find_checkpoint(train_run_dir)

    infer_run_dir = _run_infer(infer_path, ckpt_path, tmp_path / "infer")
    predictions_dir = infer_run_dir / "predictions"
    assert predictions_dir.is_dir()
    assert (predictions_dir / "predictions.h5").exists()
    assert (predictions_dir / "WRITER_INFO.txt").exists()


@pytest.mark.slow
@pytest.mark.parametrize("train_path, infer_path", ENSEMBLE_PAIRS, ids=ENSEMBLE_IDS)
def test_train_and_ensemble_infer(train_path: Path, infer_path: Path, tmp_path: Path) -> None:
    """Train a deep ensemble and run ensemble inference.

    Args:
        train_path: Path to a train-mode YAML config with a deep-ensemble
            strategy.
        infer_path: Path to a user-facing infer-mode YAML config with an
            ensemble inferencer.
        tmp_path: Pytest temporary directory (auto-cleaned).
    """
    logging.info("Ensemble dry run: %s", train_path.stem)

    train_run_dir = _run_train(train_path, tmp_path / "train")
    ckpt_path = find_checkpoint(train_run_dir)

    infer_run_dir = _run_infer(infer_path, ckpt_path, tmp_path / "infer")
    predictions_dir = infer_run_dir / "predictions"
    assert predictions_dir.is_dir()
    assert (predictions_dir / "predictions.h5").exists()
    assert (predictions_dir / "WRITER_INFO.txt").exists()
