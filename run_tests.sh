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

# Run all project tests with pytest.
#
# Usage:
#   ./run_tests.sh              # run all tests (including slow dry-runs)
#   ./run_tests.sh -q           # quick: skip slow tests
#   ./run_tests.sh -v           # verbose output
#   ./run_tests.sh -x           # stop on first failure
#   ./run_tests.sh -s           # no capture (see print/log output)
#   ./run_tests.sh -k "schnet"  # only tests matching "schnet"
# 
# Extra pytest arguments can be appended directly:
#   ./run_tests.sh -- -m "not slow" --tb=long

set -euo pipefail

cd "$(dirname "$0")"

PYTEST_ARGS=()

while getopts "qvxsh" opt; do
    case "$opt" in
        q) PYTEST_ARGS+=("-m" "not slow") ;;
        v) PYTEST_ARGS+=("-v") ;;
        x) PYTEST_ARGS+=("-x") ;;
        s) PYTEST_ARGS+=("-s") ;;
        h)
            echo "Usage: $0 [-q] [-v] [-x] [-s] [-h] [-- <extra pytest args>]"
            echo ""
            echo "  -q   quick mode — skip slow tests (dry-run train/infer pipelines)"
            echo "  -v   verbose output"
            echo "  -x   stop on first failure"
            echo "  -s   disable output capture (see prints/logs in real time)"
            echo "  -h   show this help"
            echo ""
            echo "Everything after -- is forwarded to pytest directly."
            exit 0
            ;;
        *) echo "Unknown option: -$OPTARG"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# Collect any extra arguments
if [[ $# -gt 0 ]]; then
    if [[ "$1" == "--" ]]; then
        shift
    fi
    PYTEST_ARGS+=("$@")
fi

echo "Running: pytest tests/ ${PYTEST_ARGS[*]}"
echo ""

pytest tests/ "${PYTEST_ARGS[@]}"
