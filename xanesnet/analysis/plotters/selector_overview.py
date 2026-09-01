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

"""Plotter comparing a reader's selectors by error, with a mean spectrum and structure grid each."""

import logging
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from xanesnet.serialization.config import Config
from xanesnet.serialization.prediction_readers import PredictionSample
from xanesnet.utils.exceptions import ConfigError

from ..result import AnalysisResults
from ..selectors import Selector
from ..utils import is_scalar_value, sample_label
from .base import Plotter
from .common import (
    COLOUR_ACCENT_GREEN,
    COLOUR_ACCENT_RED,
    COLOUR_PREDICTION,
    COLOUR_TARGET,
    add_subtitle,
    compact_layout,
    draw_structure,
    style_axis,
)
from .registry import PlotterRegistry

# Number of representative structures drawn per selector detail page.
_N_REPS = 9


@dataclass(frozen=True)
class _SelectorEntry:
    """Per-selector summary used by the scoreboard and detail pages.

    Attributes:
        label: Selector description, as shown in the analysis configuration.
        mean_error: Mean of the configured error key over the selector's samples.
        n_samples: Number of samples the selector contributed.
        samples: ``(sample, error)`` pairs in selector order.
    """

    label: str
    mean_error: float
    n_samples: int
    samples: tuple[tuple[PredictionSample, float], ...]


@PlotterRegistry.register("selector_overview")
class SelectorOverviewPlotter(Plotter):
    """Compare a reader's selectors by error, with a mean spectrum and structures each.

    For every reader with two or more selectors, a scoreboard ranks the
    selectors by mean ``err_key``; a multi-page PDF gives each selector one
    page with its mean spectrum and representative structures.

    Requires:
        Per-sample error values: provided by a scalar collector emitting ``err_key``.
        Matched raw structures (optional): representative structures per selector.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        err_key: Scalar collector key used to rank and colour selectors. Must
            be produced by a configured collector.
    """

    def __init__(self, plotter_type: str, err_key: str) -> None:
        """Initialize a selector-overview plotter."""
        super().__init__(plotter_type)
        self.err_key = err_key

    def plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write a per-reader selector scoreboard and per-selector detail pages.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``selector_overview`` tree should be written.

        Raises:
            ConfigError: If ``err_key`` is missing or non-scalar for a selected sample.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        root = output_dir / "selector_overview"

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")
            if len(reader_selectors) < 2:
                logging.info("      Only one selector, nothing to compare, skipping.")
                continue

            entries = [
                self._selector_entry(results, reader_idx, sel_idx, selector)
                for sel_idx, selector in enumerate(reader_selectors)
            ]
            valid_entries = [entry for entry in entries if entry is not None]
            if len(valid_entries) < 2:
                continue

            reader_dir = root / f"pred_{reader_idx:03d}"
            reader_dir.mkdir(parents=True, exist_ok=True)
            subtitle = results.prediction_names[reader_idx]

            self._scoreboard(valid_entries, subtitle, reader_dir / "scoreboard.pdf")
            self._selector_pages(valid_entries, subtitle, reader_dir / "selector_pages.pdf")

    def _selector_entry(
        self, results: AnalysisResults, reader_idx: int, sel_idx: int, selector: Selector
    ) -> _SelectorEntry | None:
        """Summarize one selector's error and collect its samples.

        Args:
            results: Analysis pipeline outputs to plot.
            reader_idx: Zero-based prediction reader index.
            sel_idx: Zero-based selector index within the reader.
            selector: Selector to summarize.

        Returns:
            The selector's summary, or ``None`` when it selected no samples.

        Raises:
            ConfigError: If ``err_key`` is missing or non-scalar for a sample.
        """
        stream = results.collector_stream(reader_idx, sel_idx)
        samples: list[tuple[PredictionSample, float]] = []
        for sample, record in zip(selector, stream if stream is not None else repeat({})):
            value = record.get(self.err_key)
            if not is_scalar_value(value):
                raise ConfigError(
                    f"Key '{self.err_key}' is missing or not a scalar for sample '{sample['sample_id']}'. "
                    "Configure a scalar collector that produces this key."
                )
            samples.append((sample, float(value)))

        if not samples:
            return None

        errors = np.array([error for _, error in samples])
        return _SelectorEntry(
            label=str(selector),
            mean_error=float(errors.mean()),
            n_samples=len(samples),
            samples=tuple(samples),
        )

    def _scoreboard(self, entries: list[_SelectorEntry], subtitle: str, out: Path) -> None:
        """Write a horizontal bar chart comparing every selector's mean error.

        Args:
            entries: Per-selector summaries in ranking order.
            subtitle: Plot subtitle text describing the prediction reader.
            out: Destination PDF path.
        """
        rows = sorted(entries, key=lambda entry: entry.mean_error)
        global_mean = float(np.mean([entry.mean_error for entry in rows]))
        labels = [f"{entry.label}  (n={entry.n_samples})" for entry in rows]
        means = [entry.mean_error for entry in rows]
        colours = [COLOUR_ACCENT_GREEN if mean <= global_mean else COLOUR_ACCENT_RED for mean in means]

        fig, ax = plt.subplots(figsize=(6.5, max(2.4, 0.35 * len(rows) + 0.9)))
        ys = np.arange(len(rows))
        ax.barh(ys, means, color=colours, alpha=0.85)
        ax.axvline(global_mean, color="black", linewidth=1.0, linestyle="--", label="mean of selectors shown")
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(self.err_key)
        ax.legend(fontsize=8, framealpha=0.9)
        style_axis(ax)
        add_subtitle(fig, subtitle)
        compact_layout(fig)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    def _selector_pages(self, entries: list[_SelectorEntry], subtitle: str, out: Path) -> None:
        """Write one detail page per selector with structures.

        Selectors without any structure-matched sample are skipped; their mean
        spectrum alone would leave the structure grid empty. Pages follow the
        scoreboard order (best mean error first).

        Args:
            entries: Per-selector summaries.
            subtitle: Plot subtitle text describing the prediction reader.
            out: Destination PDF path.
        """
        pages = [entry for entry in entries if any(sample.get("structure") is not None for sample, _ in entry.samples)]
        if not pages:
            return
        pages = sorted(pages, key=lambda entry: entry.mean_error)

        with PdfPages(out) as pdf:
            for entry in pages:
                fig = self._selector_page(entry, subtitle)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    def _selector_page(self, entry: _SelectorEntry, subtitle: str) -> Figure:
        """Build one selector page: mean spectrum next to representative structures.

        The left panel shows the mean target and prediction spectra over all of
        the selector's samples; the right panel is a grid of the structures
        whose error is closest to the selector's median error.

        Args:
            entry: Selector summary carrying its samples.
            subtitle: Plot subtitle text describing the prediction reader.

        Returns:
            Matplotlib figure for one selector detail page.
        """
        samples = entry.samples
        preds = np.stack([np.asarray(sample["prediction"]).ravel() for sample, _ in samples])
        targets = np.stack([np.asarray(sample["target"]).ravel() for sample, _ in samples])
        target_mean = targets.mean(axis=0)
        pred_mean = preds.mean(axis=0)
        pred_std = preds.std(axis=0)
        median_error = float(np.median([error for _, error in samples]))

        structured = [(sample, error) for sample, error in samples if sample.get("structure") is not None]
        reps = sorted(structured, key=lambda pair: abs(pair[1] - median_error))[:_N_REPS]

        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.45])

        ax_spec = fig.add_subplot(gs[0, 0])
        x = np.arange(len(target_mean))
        ax_spec.plot(x, target_mean, color=COLOUR_TARGET, linewidth=1.6, label="Target mean")
        ax_spec.fill_between(
            x,
            pred_mean - pred_std,
            pred_mean + pred_std,
            color=COLOUR_PREDICTION,
            alpha=0.25,
            linewidth=0,
            label="Prediction +/- 1 std",
        )
        ax_spec.plot(x, pred_mean, color=COLOUR_PREDICTION, linewidth=2.0, label="Prediction mean")
        ax_spec.set_xlabel("Energy")
        ax_spec.set_ylabel("Intensity")
        ax_spec.set_title(
            f"{entry.label}: n={entry.n_samples}, mean {self.err_key}={entry.mean_error:.4g}, median={median_error:.4g}",
            loc="left",
        )
        ax_spec.legend(fontsize=8, framealpha=0.9)
        style_axis(ax_spec)

        gs_right = gs[0, 1].subgridspec(3, 3, wspace=0.04, hspace=0.14)
        for k, (sample, error) in enumerate(reps):
            ax = fig.add_subplot(gs_right[k // 3, k % 3])
            structure = sample.get("structure")
            if structure is not None:
                draw_structure(
                    ax,
                    structure,
                    str(sample["sample_id"]),
                    sample.get("target_site_index"),
                    legend_fontsize=5.0,
                    show_scale=False,
                )
            ax.set_title(f"{sample_label(sample)}  {self.err_key}={error:.3g}", fontsize=5.5)
        for k in range(len(reps), _N_REPS):
            fig.add_subplot(gs_right[k // 3, k % 3]).axis("off")

        add_subtitle(fig, subtitle)
        fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.08, wspace=0.06)
        return fig

    @property
    def signature(self) -> Config:
        """Return the selector-overview plotter signature."""
        signature = super().signature
        signature.update_with_dict({"err_key": self.err_key})
        return signature
