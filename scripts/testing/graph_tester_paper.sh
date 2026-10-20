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
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

# Override when the active shell does not provide the project environment.
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Required
JSON_DIR="./data/toy_data/"

# Exactly one of these should be used. If FILE_STEM is non-empty, it wins.
SAMPLE_INDEX=10
FILE_STEM=""

# Graph parameters
CUTOFF=10.0
MAX_NEIGHBORS=50
GRAPH_METHODS=(radius cov_radius voronoi)
MIN_FACET_AREA="1.0%"           # e.g. "0.25" or "1.0%" (voronoi only)
COV_RADII_SCALE=1.5              # cov_radius only

# Publication figure options
LABEL_ATOMS=false
MARK_TARGET_SITE=false
NO_AXIS_LABELS=true
NO_SHOW=true
VIEW_ELEV=22.0
VIEW_AZIM=-58.0
ZOOM=1.10
TARGET_SITE_IDX=0
STRUCTURE_SAVE_PATH="./runs/testing/graph_tester_structure.png"
EDGES_SAVE_DIR="./runs/testing"
EDGES_SAVE_PREFIX="graph_tester_edges"
LEGEND_MODE="standalone"
LEGEND_SAVE_PATH="./runs/testing/graph_tester_legend.png"
VORONOI_CELL_SCOPE="first-shell"
VORONOI_CELLS_SAVE_PATH="./runs/testing/graph_tester_voronoi_cells.png"

common_args=(
	"scripts/testing/src/graph_tester_paper.py"
	"--json-dir" "$JSON_DIR"
	"--cutoff" "$CUTOFF"
	"--max-neighbors" "$MAX_NEIGHBORS"
	"--cov-radii-scale" "$COV_RADII_SCALE"
	"--target-site-idx" "$TARGET_SITE_IDX"
	"--view-elev" "$VIEW_ELEV"
	"--view-azim" "$VIEW_AZIM"
	"--zoom" "$ZOOM"
	"--legend-mode" "$LEGEND_MODE"
	"--save-legend" "$LEGEND_SAVE_PATH"
)

if [[ -n "$FILE_STEM" ]]; then
	common_args+=("--file" "$FILE_STEM")
else
	common_args+=("--index" "$SAMPLE_INDEX")
fi

if [[ -n "$MIN_FACET_AREA" ]]; then
	common_args+=("--min-facet-area" "$MIN_FACET_AREA")
fi

if [[ "$LABEL_ATOMS" == "true" ]]; then
	common_args+=("--label-atoms")
fi

if [[ "$MARK_TARGET_SITE" == "false" ]]; then
	common_args+=("--no-target-marker")
fi

if [[ "$NO_AXIS_LABELS" == "true" ]]; then
	common_args+=("--no-axis-labels")
fi

if [[ "$NO_SHOW" == "true" ]]; then
	common_args+=("--no-show")
fi

for graph_method in "${GRAPH_METHODS[@]}"; do
	edges_save_path="${EDGES_SAVE_DIR}/${EDGES_SAVE_PREFIX}_${graph_method}.png"
	args=("${common_args[@]}" "--graph-method" "$graph_method" "--save-edges" "$edges_save_path")
	if [[ "$graph_method" == "${GRAPH_METHODS[0]}" && -n "$STRUCTURE_SAVE_PATH" ]]; then
		args+=("--save-structure" "$STRUCTURE_SAVE_PATH")
	fi
	if [[ "$graph_method" == "voronoi" && -n "$VORONOI_CELLS_SAVE_PATH" ]]; then
		args+=("--save-voronoi-cells" "$VORONOI_CELLS_SAVE_PATH" "--voronoi-cell-scope" "$VORONOI_CELL_SCOPE")
	fi
	echo "Running: ${PYTHON_BIN} ${args[*]}"
	"${PYTHON_BIN}" "${args[@]}"
done
