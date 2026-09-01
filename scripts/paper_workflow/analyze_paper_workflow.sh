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
# location, run naming scheme, or explicit inference-run paths.
OUT_DIR="${OUT_DIR:-./runs/paper_workflow}"
MLP_RUN_NAME="${MLP_RUN_NAME:-paper_workflow_mlp}"
SCHNET_RUN_NAME="${SCHNET_RUN_NAME:-paper_workflow_schnet}"

ANALYSIS_COMPARE_NAME="${ANALYSIS_COMPARE_NAME:-paper_workflow_schnet_vs_mlp}"
ANALYSIS_ELEMENTS_NAME="${ANALYSIS_ELEMENTS_NAME:-paper_workflow_schnet_elements}"
ANALYSIS_CLUSTERS_NAME="${ANALYSIS_CLUSTERS_NAME:-paper_workflow_schnet_clusters}"

# Path to the newest inference run for a model run name.
latest_infer_run() {
    local run_name="$1"
    ls -dt "${OUT_DIR}"/infer_"${run_name}"_* 2>/dev/null | head -1 || true
}

if [[ -z "${MLP_RUN:-}" ]]; then
    MLP_RUN="$(latest_infer_run "$MLP_RUN_NAME")"
fi
if [[ -z "${SCHNET_RUN:-}" ]]; then
    SCHNET_RUN="$(latest_infer_run "$SCHNET_RUN_NAME")"
fi

if [[ ! -d "${MLP_RUN}/predictions" ]]; then
    echo "ERROR: no MLP inference run found. Set MLP_RUN or run infer_paper_workflow.sh first." >&2
    exit 1
fi
if [[ ! -d "${SCHNET_RUN}/predictions" ]]; then
    echo "ERROR: no SchNet inference run found. Set SCHNET_RUN or run infer_paper_workflow.sh first." >&2
    exit 1
fi

echo "[1/3] analyze: configs/paper_workflow/analyze_schnet_vs_mlp.yaml"
python -m xanesnet.cli analyze \
    -i ./configs/paper_workflow/analyze_schnet_vs_mlp.yaml \
    -r "$MLP_RUN" \
    -r "$SCHNET_RUN" \
    -d MLP SchNet \
    -o "$OUT_DIR" \
    -n "$ANALYSIS_COMPARE_NAME" \
    --yes

echo "[2/3] analyze: configs/paper_workflow/analyze_schnet_elements.yaml"
python -m xanesnet.cli analyze \
    -i ./configs/paper_workflow/analyze_schnet_elements.yaml \
    -r "$SCHNET_RUN" \
    -d SchNet \
    -o "$OUT_DIR" \
    -n "$ANALYSIS_ELEMENTS_NAME" \
    --yes

echo "[3/3] analyze: configs/paper_workflow/analyze_schnet_clusters.yaml"
python -m xanesnet.cli analyze \
    -i ./configs/paper_workflow/analyze_schnet_clusters.yaml \
    -r "$SCHNET_RUN" \
    -d SchNet \
    -o "$OUT_DIR" \
    -n "$ANALYSIS_CLUSTERS_NAME" \
    --yes

echo "Analysis finished. Results are under: ${OUT_DIR}"
