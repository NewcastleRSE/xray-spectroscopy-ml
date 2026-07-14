#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# encodings_tester dispatcher
# Edit values below, then run:
#   bash scripts/dispatchers/encodings_tester.sh
# -----------------------------------------------------------------------------

# Path to the training config whose encodings should be visualized.
# The datasource, dataset, and encoding pipeline are taken from this file.
CONFIG="./configs/in_mlp.yaml"

# Override the datasource json_path from the config (leave empty to use the
# value in the config file).
DATA_DIR="/media/hendrik/ExternalSSD/final_data/omnixas/"

# Maximum number of spectra drawn per panel (keep small for readability).
MAX_SAMPLES=12

# Optional output file (PDF with one page per encoding + overview).
# Leave empty to display interactively.
SAVE_PATH="./runs/encoding_check.pdf"   # e.g. "runs/encoding_check.pdf"

# Set to "true" to suppress plt.show() (useful for headless environments).
NO_SHOW=false

# ---------------------------------------------------------------------------

args=(
    "scripts/encodings_tester.py"
    "--config"    "$CONFIG"
    "--max-samples" "$MAX_SAMPLES"
)

[[ -n "$DATA_DIR"   ]] && args+=("--data-dir" "$DATA_DIR")
[[ -n "$SAVE_PATH"  ]] && args+=("--save" "$SAVE_PATH")
[[ "$NO_SHOW" == "true" ]] && args+=("--no-show")

python "${args[@]}"
