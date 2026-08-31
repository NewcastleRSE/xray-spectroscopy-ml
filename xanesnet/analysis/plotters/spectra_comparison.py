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
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from xanesnet.serialization.config import Config
from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.serialization.prediction_readers import PredictionSample
from xanesnet.utils.exceptions import ConfigError

from ..result import AnalysisResults
from ..sample_data import iter_aligned
from ..selectors import Selector
from ..utils import is_scalar_value
from .base import Plotter
from .common import (
    combined_spectra_page_figure,
    spectra_page_figure,
    spectra_structure_page_figure,
)
from .registry import PlotterRegistry

_Entry = tuple[PredictionSample, dict[str, Any], float]


@PlotterRegistry.register("spectra_comparison")
class SpectraComparisonPlotter(Plotter):
    """Plot the N best and N worst spectra per method.

    For every prediction-reader/selector pair the ``n_samples`` best and
    ``n_samples`` worst samples, ranked by a scalar error value, are written as
    multi-page PDFs. Structure-matched samples show their structure next to the
    spectra on the same page.

    When two or more prediction readers are configured, an additional
    ``combined`` PDF per shared selector index ranks the samples common to
    every reader by their mean error and overlays every reader's prediction
    against the shared target for the best and worst of them.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        n_samples: Number of best and worst samples plotted per method.
        sort_key: Scalar collector key used to rank samples. Must be produced
            by a configured collector.
    """

    def __init__(self, plotter_type: str, n_samples: int, sort_key: str) -> None:
        """Initialize a best/worst spectra comparison plotter."""
        super().__init__(plotter_type)
        self.n_samples = n_samples
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
        metric = self.sort_key

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                label = results.method_label(reader_idx, sel_idx)
                stream = results.collector_stream(reader_idx, sel_idx)

                entries = self._collect_entries(selector, stream)
                if not entries:
                    continue

                by_error = sorted(entries, key=lambda entry: entry[2])
                best = by_error[: self.n_samples]
                worst = list(reversed(by_error[-self.n_samples :]))

                combo_dir = root / label.dir_name
                combo_dir.mkdir(parents=True, exist_ok=True)

                self._write_pages(best, label.lines, combo_dir / "best.pdf", "best", metric)
                self._write_pages(worst, label.lines, combo_dir / "worst.pdf", "worst", metric)

        self._plot_combined(results, root)

    def _collect_entries(self, selector: Selector, stream: JSONLStream | None) -> list[_Entry]:
        """Collect per-sample entries with their ranking error value.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            stream: Optional collector result stream aligned with ``selector``.

        Returns:
            ``(sample, collector scalars, error)`` entries in selector order.
        """
        entries: list[_Entry] = []
        for sample, record in iter_aligned(selector, stream):
            value = record.get(self.sort_key)
            if not is_scalar_value(value):
                raise ConfigError(
                    f"Sort key '{self.sort_key}' is missing or not a scalar for sample "
                    f"'{sample['sample_id']}'. Configure a scalar collector that produces this key."
                )
            entries.append((sample, record, float(value)))
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

    def _plot_combined(self, results: AnalysisResults, root: Path) -> None:
        """Write combined best/worst pages ranking samples shared by every reader.

        For every selector index shared by all prediction readers, the mean
        ranking error across readers is computed for the samples common to
        all of them, and the best and worst ``n_samples`` overlay every
        reader's prediction against the shared target. Nothing is written
        with fewer than two prediction readers.

        Args:
            results: Analysis pipeline outputs to plot.
            root: Root ``spectra_comparison`` directory; combined PDFs are
                written under ``<root>/combined/``.
        """
        if len(results.selectors) < 2:
            return

        n_common_selectors = min(len(reader_selectors) for reader_selectors in results.selectors)
        combined_root = root / "combined"

        for sel_idx in range(n_common_selectors):
            by_reader = [
                {
                    str(sample["sample_id"]): (sample, record, error)
                    for sample, record, error in self._collect_entries(
                        results.selectors[reader_idx][sel_idx], results.collector_stream(reader_idx, sel_idx)
                    )
                }
                for reader_idx in range(len(results.selectors))
            ]
            common_ids = set.intersection(*(set(entries) for entries in by_reader))
            if len(common_ids) < 2:
                continue

            mean_error = {
                sample_id: sum(entries[sample_id][2] for entries in by_reader) / len(by_reader)
                for sample_id in common_ids
            }
            ranked = sorted(common_ids, key=lambda sample_id: mean_error[sample_id])
            n = min(self.n_samples, len(ranked))
            best = ranked[:n]
            worst = list(reversed(ranked[-n:]))

            selector_type = results.selectors[0][sel_idx].selector_type
            selector_str = str(results.selectors[0][sel_idx])
            combo_dir = combined_root / f"sel_{sel_idx:03d}_{selector_type}"
            combo_dir.mkdir(parents=True, exist_ok=True)

            self._write_combined_pages(
                by_reader,
                results.prediction_names,
                best,
                mean_error,
                self.sort_key,
                selector_str,
                combo_dir / "best.pdf",
                "best",
            )
            self._write_combined_pages(
                by_reader,
                results.prediction_names,
                worst,
                mean_error,
                self.sort_key,
                selector_str,
                combo_dir / "worst.pdf",
                "worst",
            )

    @staticmethod
    def _write_combined_pages(
        by_reader: list[dict[str, _Entry]],
        method_labels: list[str],
        sample_ids: list[str],
        mean_error: dict[str, float],
        metric: str,
        selector_str: str,
        pdf_path: Path,
        direction: str,
    ) -> None:
        """Write one combined multi-page PDF for shared best or worst samples.

        Args:
            by_reader: Per-reader mapping from sample id to its ``(sample, collector
                scalars, error)`` entry.
            method_labels: Display name per prediction reader.
            sample_ids: Ranked sample ids to render, best or worst first.
            mean_error: Mean ranking error across readers, keyed by sample id.
            metric: Scalar key used for ranking.
            selector_str: Shared selector description for the page subtitle.
            pdf_path: Destination PDF path.
            direction: ``"best"`` or ``"worst"`` for the page subtitle.
        """
        total = len(sample_ids)
        with PdfPages(pdf_path) as pdf:
            for rank, sample_id in enumerate(sample_ids, start=1):
                sample = by_reader[0][sample_id][0]
                target = np.asarray(sample["target"]).ravel()
                predictions = [np.asarray(entries[sample_id][0]["prediction"]).ravel() for entries in by_reader]
                subtitle = (
                    f"{direction} #{rank} of {total}  |  mean {metric}={mean_error[sample_id]:.4g}  |  {selector_str}"
                )
                fig = combined_spectra_page_figure(sample, method_labels, predictions, target, subtitle)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    @property
    def signature(self) -> Config:
        """Return the spectra comparison plotter signature."""
        signature = super().signature
        signature.update_with_dict({"n_samples": self.n_samples, "sort_key": self.sort_key})
        return signature
