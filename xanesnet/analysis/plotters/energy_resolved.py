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
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from xanesnet.serialization.config import Config

from ..result import AnalysisResults
from .base import Plotter
from .common import (
    add_subtitle,
    adjust_grid,
    compact_layout,
    finish_grid,
    method_colour,
    method_grid,
    style_axis,
    style_grid_cell,
)
from .registry import PlotterRegistry

Curve = tuple[np.ndarray, np.ndarray]


@PlotterRegistry.register("energy_resolved_loss")
class EnergyResolvedLossPlotter(Plotter):
    """Plot average energy-resolved loss curves per method and combined.

    The per-channel mean and standard deviation come from the ``vector``
    aggregator, which must be present in the analysis configuration. For each
    (prediction-reader, selector) pair and each loss key collected by an
    ``energy_resolved_loss`` collector, one figure plots the mean loss with its
    standard-deviation band. Two combined figures per loss key compare all
    methods: a per-method grid and an overlay of all mean curves, styled after
    the scalar distribution plots.

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
        y_scale: str,
        start_index: int | None,
        end_index: int | None,
        y_zoom: float | None,
        keys: list[str] | None,
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
            output_dir: Directory where the ``energy_resolved_loss_plots`` tree should be written.

        Raises:
            ConfigError: If no ``vector`` aggregator is configured.
        """
        root = output_dir / "energy_resolved_loss_plots"

        series: list[tuple[str, list[str], str, dict[str, Curve]]] = []
        key_order: list[str] = []

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx in range(len(reader_selectors)):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                label = results.method_label(reader_idx, sel_idx)

                data = results.aggregation(reader_idx, sel_idx, "vector").data
                curves = self._select_curves(data)
                if not curves:
                    continue

                colour = method_colour(len(series))
                series.append((label.dir_name, label.lines, colour, curves))
                for key in curves:
                    if key not in key_order:
                        key_order.append(key)

        if not series:
            logging.info("    No energy-resolved loss data collected, skipping.")
            return

        for dir_name, label_lines, colour, curves in series:
            combo_dir = root / dir_name
            combo_dir.mkdir(parents=True, exist_ok=True)

            for key, (mean, std) in curves.items():
                self._curve_figure(mean, std, key, "\n".join(label_lines), colour, combo_dir / f"{key}.pdf")

        combined_dir = root / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        for key in key_order:
            key_series: list[tuple[str, list[str], Curve]] = [
                (colour, label_lines, curves[key]) for _, label_lines, colour, curves in series if key in curves
            ]
            self._combined_grid(key, key_series, combined_dir)
            self._combined_overlay(key, key_series, combined_dir)

    def _select_curves(self, data: dict[str, Any]) -> dict[str, Curve]:
        """Restrict aggregated per-channel curves to the configured keys and channels.

        Args:
            data: ``vector`` aggregator data mapping each key to its ``mean``
                and ``std`` curves.

        Returns:
            Mapping from loss key to ``(mean, std)`` curves with shape ``(N,)``.
        """
        channel_slice = slice(self.start_index, self.end_index)
        return {
            key: (entry["mean"][channel_slice], entry["std"][channel_slice])
            for key, entry in data.items()
            if self.keys is None or key in self.keys
        }

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

        fig, axes = method_grid(n, cell_width=3.4, cell_height=2.6)
        ncols = len(axes[0])

        for idx, (colour, label_lines, (mean, std)) in enumerate(series):
            ax = axes[idx // ncols][idx % ncols]
            x = np.arange(len(mean))
            ax.plot(x, mean, color=colour, linewidth=1.6)
            ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.25, linewidth=0)
            style_grid_cell(ax, label_lines)

        finish_grid(axes, n, "Energy", key)
        self._apply_axis(axes[0][0])
        adjust_grid(fig, left=0.13, right=0.99, top=0.95, bottom=0.16)
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

    @property
    def signature(self) -> Config:
        """Return the energy-resolved loss plotter signature."""
        signature = super().signature
        signature.update_with_dict(
            {
                "y_scale": self.y_scale,
                "start_index": self.start_index,
                "end_index": self.end_index,
                "y_zoom": self.y_zoom,
                "keys": self.keys,
            }
        )
        return signature
