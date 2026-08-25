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

"""Plotter for best/worst spectra comparisons across methods."""

import logging
from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.serialization.prediction_readers import PredictionSample

from ..reporters.base import selector_label
from ..result import AnalysisResults
from ..selectors import Selector
from .base import Plotter
from .registry import PlotterRegistry
from .utils import (
    method_label_lines,
    spectra_page_figure,
    spectra_structure_page_figure,
    spectrum_error_value,
)

_Entry = tuple[PredictionSample, dict[str, Any], float]


@PlotterRegistry.register("spectra_comparison")
class SpectraComparisonPlotter(Plotter):
    """Plot the N best and N worst spectra per method.

    For every prediction-reader/selector pair the ``n`` best and ``n`` worst
    samples, ranked by a scalar error value, are written as multi-page PDFs.
    Structure-matched samples show their structure next to the spectra on the
    same page.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        n: Number of best and worst samples plotted per method.
        sort_key: Scalar key used to rank samples. When ``None``, the MSE
            between predicted and target spectra is used. Collector values take
            precedence over sample scalars.
    """

    DEFAULT_N: ClassVar[int] = 10

    def __init__(self, plotter_type: str, n: int | None = None, sort_key: str | None = None) -> None:
        """Initialize a best/worst spectra comparison plotter."""
        super().__init__(plotter_type)
        self.n = n if n is not None else self.DEFAULT_N
        self.sort_key = sort_key

    def plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write best/worst spectra PDFs with structure panels where available.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``spectra_comparison`` tree should be written.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        root = output_dir / "spectra_comparison"
        metric = self.sort_key if self.sort_key is not None else "mse"

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                sel_label_str = selector_label(results.selectors_config, sel_idx)
                sel_cfg = results.selectors_config[sel_idx]
                label_lines = method_label_lines(results.prediction_names[reader_idx], sel_label_str, sel_cfg)

                stream: JSONLStream | None = None
                if reader_idx < len(results.collector_results) and sel_idx < len(results.collector_results[reader_idx]):
                    stream = results.collector_results[reader_idx][sel_idx]

                entries = self._collect_entries(selector, stream)
                if not entries:
                    continue

                by_error = sorted(entries, key=lambda entry: entry[2])
                best = by_error[: self.n]
                worst = list(reversed(by_error[-self.n :]))

                combo_label = f"pred_{reader_idx:03d}__sel_{sel_idx:03d}_{sel_label_str}"
                combo_dir = root / combo_label
                combo_dir.mkdir(parents=True, exist_ok=True)

                self._write_pages(best, label_lines, combo_dir / "best.pdf", "best", metric)
                self._write_pages(worst, label_lines, combo_dir / "worst.pdf", "worst", metric)

    def _collect_entries(self, selector: Selector, stream: JSONLStream | None) -> list[_Entry]:
        """Collect per-sample entries with their ranking error value.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            stream: Optional collector result stream aligned with ``selector``.

        Returns:
            ``(sample, collector scalars, error)`` entries in selector order.
        """
        entries: list[_Entry] = []
        if stream is not None:
            for sel_sample, col_sample in zip(selector, stream):
                entries.append((sel_sample, col_sample, spectrum_error_value(sel_sample, col_sample, self.sort_key)))
        else:
            for sel_sample in selector:
                entries.append((sel_sample, {}, spectrum_error_value(sel_sample, {}, self.sort_key)))
        return entries

    @staticmethod
    def _write_pages(
        entries: list[_Entry],
        label_lines: list[str],
        pdf_path: Path,
        direction: str,
        metric: str,
    ) -> None:
        """Write one multi-page PDF for the best or worst entries of one method.

        Args:
            entries: Ranked ``(sample, collector scalars, error)`` entries, best or worst first.
            label_lines: Method label lines used for the page subtitle.
            pdf_path: Destination PDF path.
            direction: ``"best"`` or ``"worst"`` for the page subtitle.
            metric: Scalar key used for ranking.
        """
        total = len(entries)
        with PdfPages(pdf_path) as pdf:
            for rank, (sample, col_scalars, error) in enumerate(entries, start=1):
                subtitle = f"{direction} #{rank} of {total}  |  {metric}={error:.4g}  |  " + "  |  ".join(label_lines)
                if sample.get("structure") is not None:
                    fig = spectra_structure_page_figure(sample, col_scalars, subtitle)
                else:
                    fig = spectra_page_figure(sample, col_scalars, subtitle)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
