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

"""Shared fixtures and helpers for XANESNET dry-run tests."""


from pathlib import Path

from xanesnet.utils.prompts import set_auto_yes

# Suppress interactive prompts during automated testing.
set_auto_yes(True)

TESTS_DIR = Path(__file__).resolve().parent
TRAIN_DIR = TESTS_DIR / "train"
INFER_DIR = TESTS_DIR / "infer"
ANALYZE_DIR = TESTS_DIR / "analyze"


def collect_configs(subdir: str) -> list[Path]:
    """Collect all YAML files in a dry-runs subdirectory.

    Args:
        subdir: Subdirectory name (``"train"``, ``"infer"``, or ``"analyze"``).

    Returns:
        Sorted list of ``.yaml`` file paths found in the subdirectory.
    """
    directory = TESTS_DIR / subdir
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.yaml"))


def find_checkpoint(run_dir: Path) -> Path:
    """Locate the deployment checkpoint produced by a training run.

    Returns ``models/final.pth`` when it exists (the canonical deployment
    checkpoint carrying the full signature).  Falls back to the most recent
    epoch checkpoint under ``checkpoints/``.

    Args:
        run_dir: Root directory of a completed training run.

    Returns:
        Path to the checkpoint ``.pth`` file.

    Raises:
        FileNotFoundError: If no checkpoint file can be found.
    """
    final = run_dir / "models" / "final.pth"
    if final.exists():
        return final
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    return checkpoints[-1]


def collect_model_pairs() -> list[tuple[Path, Path]]:
    """Collect paired (train_config, infer_config) paths.

    A pair is formed when a train config in ``train/`` has a matching
    infer config in ``infer/`` with the same stem (e.g.
    ``test_schnet.yaml`` in both directories).

    Returns:
        List of ``(train_path, infer_path)`` tuples sorted by train stem.
    """
    train_configs = collect_configs("train")
    infer_configs = collect_configs("infer")
    infer_stems = {p.stem for p in infer_configs}

    pairs: list[tuple[Path, Path]] = []
    for train_path in train_configs:
        stem = train_path.stem
        if stem in infer_stems:
            infer_path = INFER_DIR / f"{stem}.yaml"
            pairs.append((train_path, infer_path))
    return sorted(pairs, key=lambda x: x[0].stem)
