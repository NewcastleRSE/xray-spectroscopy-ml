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

"""Plotter for the per-channel bias-variance error decomposition."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..result import AnalysisResults
from .base import Plotter
from .common import (
    COLOR_ACCENT_GREEN,
    COLOR_ACCENT_RED,
    add_subtitle,
    adjust_grid,
    compact_layout,
    finish_grid,
    method_color,
    method_grid,
    style_axis,
    style_grid_cell,
)
from .registry import PlotterRegistry

# Label lines, color, MSE, squared bias, variance.
_MethodStats = tuple[list[str], str, np.ndarray, np.ndarray, np.ndarray]

# The two decomposition terms keep fixed colors across every figure so they
# stay recognizable next to the per-method MSE color.
_BIAS_COLOR = COLOR_ACCENT_RED
_VARIANCE_COLOR = COLOR_ACCENT_GREEN


@PlotterRegistry.register("bias_variance")
class BiasVariancePlotter(Plotter):
    """Plot the per-channel bias-variance decomposition of the MSE per method.

    A combined grid compares all methods on shared axes.

    Requires:
        Per-channel bias/variance decomposition: provided by the ``bias_variance`` aggregator.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        latex_font: Render figures in a LaTeX-style serif font when ``True``.
    """

    def __init__(self, plotter_type: str, latex_font: bool) -> None:
        """Initialize a bias-variance plotter."""
        super().__init__(plotter_type, latex_font=latex_font)

    def _plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write per-method decomposition PDFs and a combined grid.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``bias_variance_plots`` tree should be written.

        Raises:
            ConfigError: If no ``bias_variance`` aggregator is configured.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        root = output_dir / "bias_variance_plots"
        root.mkdir(parents=True, exist_ok=True)

        methods: list[_MethodStats] = []

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx in range(len(reader_selectors)):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                label = results.method_label(reader_idx, sel_idx)

                data = results.aggregation(reader_idx, sel_idx, "bias_variance").data
                if not data:
                    continue
                stats = (data["mse"], data["bias2"], data["variance"])

                color = method_color(len(methods))
                methods.append((label.lines, color, *stats))

                combo_dir = root / label.dir_name
                combo_dir.mkdir(parents=True, exist_ok=True)
                self._stats_figure(*stats, "\n".join(label.lines), color, combo_dir / "bias_variance.pdf")

        if not methods:
            logging.info("    No samples selected, skipping.")
            return

        combined_dir = root / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        self._stats_grid(methods, combined_dir / "bias_variance_grid.pdf")

    @staticmethod
    def _draw_decomposition(ax: Axes, mse: np.ndarray, bias2: np.ndarray, variance: np.ndarray, color: str) -> None:
        """Draw the three decomposition curves into one axis.

        Args:
            ax: Matplotlib axis to draw on.
            mse: Per-channel mean squared error with shape ``(N,)``.
            bias2: Per-channel squared bias with shape ``(N,)``.
            variance: Per-channel variance with shape ``(N,)``.
            color: Method color used for the MSE curve.
        """
        x = np.arange(len(mse))
        ax.plot(x, mse, color=color, linewidth=2.0, label="MSE")
        ax.plot(x, bias2, color=_BIAS_COLOR, linewidth=1.6, linestyle="--", label="Bias^2")
        ax.plot(x, variance, color=_VARIANCE_COLOR, linewidth=1.6, linestyle=":", label="Variance")

    def _stats_figure(
        self,
        mse: np.ndarray,
        bias2: np.ndarray,
        variance: np.ndarray,
        subtitle: str,
        color: str,
        out: Path,
    ) -> None:
        """Write one decomposition figure for a single method.

        Args:
            mse: Per-channel mean squared error with shape ``(N,)``.
            bias2: Per-channel squared bias with shape ``(N,)``.
            variance: Per-channel variance with shape ``(N,)``.
            subtitle: Plot subtitle text describing prediction and selector context.
            color: Method color used for the MSE curve.
            out: Destination PDF path.
        """
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        self._draw_decomposition(ax, mse, bias2, variance, color)
        ax.set_xlabel("Energy")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8, framealpha=0.9)
        style_axis(ax)
        add_subtitle(fig, subtitle)
        compact_layout(fig)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _stats_grid(methods: list[_MethodStats], out: Path) -> None:
        """Write one combined figure with a per-method decomposition grid.

        All cells share one x range and one y range so the cells stay directly
        comparable.

        Args:
            methods: Method decomposition statistics in first-seen order.
            out: Destination PDF path.
        """
        n = len(methods)

        fig, axes = method_grid(n, cell_width=3.4, cell_height=2.6)
        ncols = len(axes[0])

        for idx, (label_lines, color, mse, bias2, variance) in enumerate(methods):
            ax = axes[idx // ncols][idx % ncols]
            BiasVariancePlotter._draw_decomposition(ax, mse, bias2, variance, color)
            style_grid_cell(ax, label_lines)
            ax.legend(fontsize=6, framealpha=0.9)

        finish_grid(axes, n, "Energy", "Loss")
        adjust_grid(fig, left=0.13, right=0.99, top=0.88, bottom=0.16)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
