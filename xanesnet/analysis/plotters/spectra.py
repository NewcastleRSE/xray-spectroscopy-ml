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

"""Plotter that writes predicted-target spectra comparisons for every selected sample."""

import logging
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from xanesnet.serialization.config import Config
from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.serialization.prediction_readers import PredictionSample

from ..result import AnalysisResults
from ..sample_data import iter_aligned
from ..selectors import Selector
from .base import Plotter
from .common import (
    combined_spectra_page_figure,
    spectra_page_figure,
    spectra_structure_page_figure,
)
from .registry import PlotterRegistry


@PlotterRegistry.register("spectra_all")
class AllSpectraPlotter(Plotter):
    """Plot predicted and target spectra for every selected sample.

    One multi-page PDF is written per prediction-reader/selector pair; each
    selected sample contributes one page. Structure-matched samples show their
    structure next to the spectra on the same page. When ``max_pages`` is
    smaller than the number of selected samples, a random subset is drawn.

    When two or more prediction readers are configured, an additional
    ``combined`` PDF per shared selector index overlays every reader's
    prediction against the shared target, for the samples common to all
    readers.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        max_pages: Maximum number of pages per PDF. ``None`` writes all selected samples.
    """

    def __init__(self, plotter_type: str, max_pages: int | None) -> None:
        """Initialize an all-spectra comparison plotter."""
        super().__init__(plotter_type)
        self.max_pages = max_pages

    def plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write one multi-page spectra PDF per prediction-reader/selector pair.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``spectra_plots`` tree should be written.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        root = output_dir / "spectra_plots"
        root.mkdir(parents=True, exist_ok=True)

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                label = results.method_label(reader_idx, sel_idx)
                stream = results.collector_stream(reader_idx, sel_idx)
                pdf_path = root / f"{label.dir_name}.pdf"

                self._plot_to_pdf(selector, stream, pdf_path, "\n".join(label.lines))

        self._plot_combined(results, root)

    def _plot_to_pdf(
        self,
        selector: Selector,
        stream: JSONLStream | None,
        pdf_path: Path,
        subtitle: str,
    ) -> None:
        """Write selected spectra comparisons into a multi-page PDF.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            stream: Optional collector result stream aligned with ``selector``.
            pdf_path: Destination PDF path.
            subtitle: Subtitle text describing prediction and selector context.
        """
        entries: list[tuple[PredictionSample, dict[str, Any]]] = list(iter_aligned(selector, stream))
        if not entries:
            return
        entries = _random_subset(entries, self.max_pages)

        with PdfPages(pdf_path) as pdf:
            for sample, col_scalars in entries:
                if sample.get("structure") is not None:
                    fig = spectra_structure_page_figure(sample, col_scalars, subtitle)
                else:
                    fig = spectra_page_figure(sample, col_scalars, subtitle)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    def _plot_combined(self, results: AnalysisResults, root: Path) -> None:
        """Write combined spectra pages comparing prediction readers on shared samples.

        For every selector index shared by all prediction readers, the samples
        common to every reader (matched by sample id) are collected and a
        random subset overlays every reader's prediction against the shared
        target. Nothing is written with fewer than two prediction readers.

        Args:
            results: Analysis pipeline outputs to plot.
            root: Root ``spectra_plots`` directory; combined PDFs are written
                under ``<root>/combined/``.
        """
        if len(results.selectors) < 2:
            return

        n_common_selectors = min(len(reader_selectors) for reader_selectors in results.selectors)
        combined_root = root / "combined"

        for sel_idx in range(n_common_selectors):
            by_reader = [
                {
                    str(sample["sample_id"]): (sample, col_scalars)
                    for sample, col_scalars in iter_aligned(
                        results.selectors[reader_idx][sel_idx], results.collector_stream(reader_idx, sel_idx)
                    )
                }
                for reader_idx in range(len(results.selectors))
            ]
            common_ids = set.intersection(*(set(entries) for entries in by_reader))
            if len(common_ids) < 2:
                continue

            selected_ids = _random_subset(sorted(common_ids), self.max_pages)
            selector_type = results.selectors[0][sel_idx].selector_type
            subtitle = str(results.selectors[0][sel_idx])

            combined_root.mkdir(parents=True, exist_ok=True)
            pdf_path = combined_root / f"sel_{sel_idx:03d}_{selector_type}.pdf"
            with PdfPages(pdf_path) as pdf:
                for sample_id in selected_ids:
                    sample = by_reader[0][sample_id][0]
                    target = np.asarray(sample["target"]).ravel()
                    predictions = [np.asarray(entries[sample_id][0]["prediction"]).ravel() for entries in by_reader]
                    fig = combined_spectra_page_figure(sample, results.prediction_names, predictions, target, subtitle)
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

    @property
    def signature(self) -> Config:
        """Return the all-spectra plotter signature."""
        signature = super().signature
        signature.update_with_dict({"max_pages": self.max_pages})
        return signature


def _random_subset(items: list[Any], max_count: int | None) -> list[Any]:
    """Return every item, or a random subset when ``max_count`` truncates them.

    Args:
        items: Candidate items in their natural order.
        max_count: Maximum number of items to keep, or ``None`` to keep all.

    Returns:
        All items when ``max_count`` is ``None`` or not smaller than
        ``len(items)``; otherwise a random sample of that size, drawn using
        the global seed.
    """
    if max_count is None or len(items) <= max_count:
        return list(items)
    return random.sample(list(items), max_count)
