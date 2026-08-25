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

"""Plotter for energy-resolved loss curves."""

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from xanesnet.serialization.jsonl_stream import JSONLStream

from ..reporters.base import selector_label
from ..result import AnalysisResults
from .base import Plotter
from .registry import PlotterRegistry
from .utils import (
    add_subtitle,
    compact_layout,
    method_colour,
    method_label_lines,
    style_axis,
)

Curve = tuple[np.ndarray, np.ndarray]


@PlotterRegistry.register("energy_resolved_loss")
class EnergyResolvedLossPlotter(Plotter):
    """Plot average energy-resolved loss curves per method and combined.

    For each (prediction-reader, selector) pair and each loss key collected by
    an ``energy_resolved_loss`` collector, one figure plots the per-channel
    mean loss with its standard-deviation band. Two combined figures per loss
    key compare all methods: a per-method grid and an overlay of all mean
    curves, styled after the scalar distribution plots.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        y_scale: Y-axis scaling, either ``"linear"`` or ``"log"``.
        start_index: First channel index to plot; ``None`` starts at the first channel.
        end_index: One-past-the-end channel index to plot; ``None`` plots to the last channel.
        y_zoom: Optional upper y-axis limit for zooming in around zero. The
            lower limit stays at zero.
        keys: Energy-resolved loss keys to plot. ``None`` plots every
            collected key; a list restricts the plotter to those keys so
            different plotters can use different axis settings per key.
    """

    def __init__(
        self,
        plotter_type: str,
        y_scale: str = "linear",
        start_index: int | None = None,
        end_index: int | None = None,
        y_zoom: float | None = None,
        keys: list[str] | None = None,
    ) -> None:
        """Initialize an energy-resolved loss plotter.

        Raises:
            ValueError: If ``y_scale`` is neither ``"linear"`` nor ``"log"``.
        """
        super().__init__(plotter_type)
        if y_scale not in ("linear", "log"):
            raise ValueError(f"Unsupported y_scale: {y_scale!r}; expected 'linear' or 'log'.")
        self.y_scale = y_scale
        self.start_index = start_index
        self.end_index = end_index
        self.y_zoom = y_zoom
        self.keys = keys

    def plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write per-method and combined energy-resolved loss curve PDFs.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``energy_loss_plots`` tree should be written.
        """
        root = output_dir / "energy_loss_plots"

        series: list[tuple[int, int, str, list[str], str, dict[str, Curve]]] = []
        key_order: list[str] = []

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

                curves = self._collect_curves(stream)
                if not curves:
                    continue

                colour = method_colour(len(series))
                series.append((reader_idx, sel_idx, sel_label_str, label_lines, colour, curves))
                for key in curves:
                    if key not in key_order:
                        key_order.append(key)

        if not series:
            logging.info("    No energy-resolved loss data collected, skipping.")
            return

        for reader_idx, sel_idx, sel_label_str, label_lines, colour, curves in series:
            combo_label = f"pred_{reader_idx:03d}__sel_{sel_idx:03d}_{sel_label_str}"
            combo_dir = root / combo_label
            combo_dir.mkdir(parents=True, exist_ok=True)

            for key, (mean, std) in curves.items():
                self._curve_figure(mean, std, key, "\n".join(label_lines), colour, combo_dir / f"{key}.pdf")

        combined_dir = root / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        for key in key_order:
            key_series: list[tuple[str, list[str], Curve]] = [
                (colour, label_lines, curves[key]) for _, _, _, label_lines, colour, curves in series if key in curves
            ]
            self._combined_grid(key, key_series, combined_dir)
            self._combined_overlay(key, key_series, combined_dir)

    def _collect_curves(self, stream: JSONLStream | None) -> dict[str, Curve]:
        """Aggregate per-channel loss vectors into mean and standard-deviation curves.

        Only the configured channel range and, when ``keys`` is set, only the
        configured loss keys are retained.

        Args:
            stream: Collector result stream aligned with the selector.

        Returns:
            Mapping from loss key to ``(mean, std)`` curves with shape ``(N,)``.
        """
        channel_slice = slice(self.start_index, self.end_index)
        vectors: dict[str, list[np.ndarray]] = {}
        if stream is not None:
            for record in stream:
                for key, value in record.items():
                    if key == "sample_id" or not isinstance(value, (list, tuple)):
                        continue
                    vectors.setdefault(key, []).append(np.asarray(value, dtype=float)[channel_slice])

        curves: dict[str, Curve] = {}
        for key, values in vectors.items():
            stack = np.stack(values)
            curves[key] = (stack.mean(axis=0), stack.std(axis=0))
        if self.keys is not None:
            curves = {key: curve for key, curve in curves.items() if key in self.keys}
        return curves

    def _apply_axis(self, ax: Axes) -> None:
        """Apply the configured y-axis scaling and zoom to one axis.

        Args:
            ax: Matplotlib axis to configure.
        """
        ax.set_yscale(self.y_scale)
        if self.y_zoom is not None:
            ax.set_ylim(0, self.y_zoom)

    def _curve_figure(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        key: str,
        subtitle: str,
        colour: str,
        out: Path,
    ) -> None:
        """Write one mean-loss curve figure for a single method.

        Args:
            mean: Per-channel mean loss with shape ``(N,)``.
            std: Per-channel loss standard deviation with shape ``(N,)``.
            key: Loss key used for labels and output path context.
            subtitle: Plot subtitle text describing prediction and selector context.
            colour: Method colour used for the curve.
            out: Destination PDF path.
        """
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        x = np.arange(len(mean))
        ax.plot(x, mean, color=colour, linewidth=2.0, label="Mean")
        ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.25, linewidth=0, label="+/- 1 std")
        ax.set_xlabel("Energy")
        ax.set_ylabel(key)
        ax.legend(fontsize=8, framealpha=0.9)
        style_axis(ax)
        self._apply_axis(ax)
        add_subtitle(fig, subtitle)
        compact_layout(fig)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    def _combined_grid(
        self,
        key: str,
        series: list[tuple[str, list[str], Curve]],
        out: Path,
    ) -> None:
        """Write one combined figure with a per-method curve grid.

        All cells share one x range and one y range so the cells stay directly
        comparable.

        Args:
            key: Loss key used for labels and output path context.
            series: ``(colour, label lines, (mean, std))`` per method in first-seen order.
            out: Directory where the combined grid PDF should be written.
        """
        n = len(series)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), sharex=True, sharey=True, squeeze=False
        )

        for idx, (colour, label_lines, (mean, std)) in enumerate(series):
            ax = axes[idx // ncols][idx % ncols]
            x = np.arange(len(mean))
            ax.plot(x, mean, color=colour, linewidth=1.6)
            ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.25, linewidth=0)
            ax.set_title("\n".join(label_lines), fontsize=5.5)
            ax.tick_params(labelsize=5.5)

        for ax in axes.flat[n:]:
            ax.axis("off")
        for col in range(ncols):
            used_rows = [r for r in range(nrows) if r * ncols + col < n]
            if used_rows:
                axes[used_rows[-1]][col].tick_params(labelbottom=True)

        self._apply_axis(axes[0][0])

        for i in range(nrows):
            axes[i][0].set_ylabel(key, fontsize=8)
        for j in range(ncols):
            axes[nrows - 1][j].set_xlabel("Energy", fontsize=8)
        fig.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.16, wspace=0.08, hspace=0.18)
        fig.savefig(out / f"{key}_grid.pdf", bbox_inches="tight")
        plt.close(fig)

    def _combined_overlay(
        self,
        key: str,
        series: list[tuple[str, list[str], Curve]],
        out: Path,
    ) -> None:
        """Write one combined figure overlaying all mean-loss curves.

        Args:
            key: Loss key used for labels and output path context.
            series: ``(colour, label lines, (mean, std))`` per method in first-seen order.
            out: Directory where the combined overlay PDF should be written.
        """
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for colour, label_lines, (mean, std) in series:
            x = np.arange(len(mean))
            ax.plot(x, mean, color=colour, linewidth=1.8, label="\n".join(label_lines))
            ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.12, linewidth=0)
        ax.set_xlabel("Energy")
        ax.set_ylabel(key)
        ax.legend(fontsize=8, framealpha=0.9)
        style_axis(ax)
        self._apply_axis(ax)
        compact_layout(fig)
        fig.savefig(out / f"{key}_overlay.pdf", bbox_inches="tight")
        plt.close(fig)
