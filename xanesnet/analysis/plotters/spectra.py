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
from itertools import repeat
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
from ..selectors import Selector
from ..utils import sample_key, sample_key_sort_key
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

    One multi-page PDF per method; each page shows one sample, with its
    structure when matched. ``max_pages`` draws a random subset. With two or
    more readers, a combined PDF overlays every reader's prediction for the
    samples common to all readers.

    Requires:
        Scalar collector output (optional): annotated on each page.
        Matched raw structures (optional): shown alongside the spectra.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        latex_font: Render figures in a LaTeX-style serif font when ``True``.
        max_pages: Maximum number of pages per PDF. ``None`` writes all selected samples.
    """

    def __init__(self, plotter_type: str, max_pages: int | None, latex_font: bool) -> None:
        """Initialize an all-spectra comparison plotter."""
        super().__init__(plotter_type, latex_font=latex_font)
        self.max_pages = max_pages

    def _plot(self, results: AnalysisResults, output_dir: Path) -> None:
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
        entries: list[tuple[PredictionSample, dict[str, Any]]] = list(
            zip(selector, stream if stream is not None else repeat({}))
        )
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

        For every selector index shared by all prediction readers, records
        common to every reader are matched by compound sample identity
        (sample ID and target-site index), and a random subset overlays every
        reader's prediction against the shared target. Nothing is written with
        fewer than two prediction readers.

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
            by_reader: list[dict[tuple[str, int | None], tuple[PredictionSample, dict[str, Any]]]] = []
            for reader_idx in range(len(results.selectors)):
                entries: dict[tuple[str, int | None], tuple[PredictionSample, dict[str, Any]]] = {}
                stream = results.collector_stream(reader_idx, sel_idx)
                aligned = zip(results.selectors[reader_idx][sel_idx], stream if stream is not None else repeat({}))
                for sample, col_scalars in aligned:
                    identity = sample_key(sample)
                    if identity in entries:
                        raise ConfigError(f"Duplicate prediction record for sample identity {identity!r}.")
                    entries[identity] = (sample, col_scalars)
                by_reader.append(entries)

            common_keys = set.intersection(*(set(entries) for entries in by_reader))
            if len(common_keys) < 2:
                continue

            selected_keys = _random_subset(sorted(common_keys, key=sample_key_sort_key), self.max_pages)
            selector_type = results.selectors[0][sel_idx].selector_type
            subtitle = str(results.selectors[0][sel_idx])

            combined_root.mkdir(parents=True, exist_ok=True)
            pdf_path = combined_root / f"sel_{sel_idx:03d}_{selector_type}.pdf"
            with PdfPages(pdf_path) as pdf:
                for identity in selected_keys:
                    sample = by_reader[0][identity][0]
                    target = np.asarray(sample["target"]).ravel()
                    predictions = [np.asarray(entries[identity][0]["prediction"]).ravel() for entries in by_reader]
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
