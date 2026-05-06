# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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

"""Reporter that writes scalar per-sample values to CSV files."""

import csv
import logging
from pathlib import Path
from typing import cast

from xanesnet.analysis.utils import ScalarValue, is_scalar_value
from xanesnet.serialization.jsonl_stream import JSONLStream

from ..result import AnalysisResults
from ..selectors import Selector
from .base import Reporter, selector_label
from .registry import ReporterRegistry


@ReporterRegistry.register("scalar")
class ScalarReporter(Reporter):
    """Write per-sample scalar values from selectors and collectors as CSV files.

    Args:
        reporter_type: Registered reporter name from the analysis configuration.
    """

    def __init__(self, reporter_type: str) -> None:
        """Initialize a scalar CSV reporter."""
        super().__init__(reporter_type)

    def report(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write scalar value CSV files grouped by prediction reader and selector.

        Args:
            results: Analysis pipeline outputs to report.
            output_dir: Directory where the ``scalar_values`` report tree should be written.
        """
        if not results.selectors and not results.collector_results:
            logging.info("    No data to report.")
            return

        root = output_dir / "scalar_values"

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                sel_label = selector_label(results.selectors_config, sel_idx)
                subdir = root / f"pred_{reader_idx:03d}__sel_{sel_idx:03d}_{sel_label}"
                subdir.mkdir(parents=True, exist_ok=True)

                stream: JSONLStream | None = None
                if reader_idx < len(results.collector_results) and sel_idx < len(results.collector_results[reader_idx]):
                    stream = results.collector_results[reader_idx][sel_idx]

                self._write_scalar_csvs(selector, stream, subdir)

    @staticmethod
    def _write_scalar_csvs(
        selector: Selector,
        stream: JSONLStream | None,
        output_dir: Path,
    ) -> None:
        """Write one CSV per scalar field found in selected samples and collector values.

        Each CSV uses ``file_name`` as the first column and one scalar field as the second column.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            stream: Optional collector result stream aligned with ``selector``.
            output_dir: Directory where CSV files should be written.
        """
        rows_by_key: dict[str, list[tuple[str, ScalarValue]]] = {}

        if stream is not None:
            for sel_sample, col_sample in zip(selector, stream):
                file_name = str(col_sample["file_name"])
                for key, value in sel_sample.items():
                    if key != "file_name" and is_scalar_value(value):
                        rows_by_key.setdefault(key, []).append((file_name, cast(float, value)))
                for key, value in col_sample.items():
                    if key != "file_name" and is_scalar_value(value):
                        rows_by_key.setdefault(key, []).append((file_name, cast(float, value)))
        else:
            for sel_sample in selector:
                file_name = str(sel_sample["file_name"])
                for key, value in sel_sample.items():
                    if key != "file_name" and is_scalar_value(value):
                        rows_by_key.setdefault(key, []).append((file_name, cast(float, value)))

        if not rows_by_key:
            logging.info("      No scalar data found, skipping.")
            return

        for key, rows in rows_by_key.items():
            filepath = output_dir / f"{key}.csv"
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["file_name", key])
                writer.writerows(rows)
