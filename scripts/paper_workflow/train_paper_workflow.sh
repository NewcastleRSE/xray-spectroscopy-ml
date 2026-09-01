#!/usr/bin/env bash
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

# Override these values when running the dispatcher with a different output
# location or run naming scheme, for example: OUT_DIR=/path/to/runs bash ...
OUT_DIR="${OUT_DIR:-./runs/paper_workflow}"
MLP_RUN_NAME="${MLP_RUN_NAME:-paper_workflow_mlp}"
SCHNET_RUN_NAME="${SCHNET_RUN_NAME:-paper_workflow_schnet}"

echo "[1/2] train: configs/paper_workflow/mlp_train.yaml"
python -m xanesnet.cli train \
    -i ./configs/paper_workflow/mlp_train.yaml \
    -o "$OUT_DIR" \
    -n "$MLP_RUN_NAME" \
    --yes

echo "[2/2] train: configs/paper_workflow/schnet_train.yaml"
python -m xanesnet.cli train \
    -i ./configs/paper_workflow/schnet_train.yaml \
    -o "$OUT_DIR" \
    -n "$SCHNET_RUN_NAME" \
    --yes

echo "Training finished. Results are under: ${OUT_DIR}"
