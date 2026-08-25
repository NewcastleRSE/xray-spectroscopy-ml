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
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from ..reporters.base import selector_label
from ..result import AnalysisResults
from ..selectors import Selector
from .base import Plotter
from .registry import PlotterRegistry
from .utils import (
    add_subtitle,
    compact_layout,
    method_colour,
    method_label_lines,
    style_axis,
)

# Label lines, colour, MSE, squared bias, variance.
_MethodStats = tuple[list[str], str, np.ndarray, np.ndarray, np.ndarray]

_BIAS_COLOUR = "#e85651"
_VARIANCE_COLOUR = "#3f9e6e"


@PlotterRegistry.register("bias_variance")
class BiasVariancePlotter(Plotter):
    """Plot the per-channel bias-variance decomposition of the MSE per method.

    For every (prediction-reader, selector) pair the per-channel MSE is split
    into its squared bias and variance components using the exact identity
    ``MSE = bias^2 + variance`` with the mean taken over the selected samples.
    A dominant bias component indicates a systematic shift (for example a
    wrong edge position), while a dominant variance component indicates
    sample-to-sample noise. A combined grid compares all methods on shared
    axes.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
    """

    def __init__(self, plotter_type: str) -> None:
        """Initialize a bias-variance plotter."""
        super().__init__(plotter_type)

    def plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write per-method decomposition PDFs and a combined grid.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``bias_variance_plots`` tree should be written.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        root = output_dir / "bias_variance_plots"
        root.mkdir(parents=True, exist_ok=True)

        methods: list[_MethodStats] = []

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                sel_label_str = selector_label(results.selectors_config, sel_idx)
                sel_cfg = results.selectors_config[sel_idx]
                label_lines = method_label_lines(results.prediction_names[reader_idx], sel_label_str, sel_cfg)

                stats = self._collect_stats(selector)
                if stats is None:
                    continue

                colour = method_colour(len(methods))
                methods.append((label_lines, colour, *stats))

                combo_label = f"pred_{reader_idx:03d}__sel_{sel_idx:03d}_{sel_label_str}"
                combo_dir = root / combo_label
                combo_dir.mkdir(parents=True, exist_ok=True)
                self._stats_figure(*stats, "\n".join(label_lines), colour, combo_dir / "bias_variance.pdf")

        if not methods:
            logging.info("    No samples selected, skipping.")
            return

        combined_dir = root / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        self._stats_grid(methods, combined_dir / "bias_variance_grid.pdf")

    @staticmethod
    def _collect_stats(selector: Selector) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Compute per-channel MSE, squared bias, and variance for one method.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.

        Returns:
            ``(mse, bias2, variance)`` curves with shape ``(N,)``, or ``None``
            when no samples are selected.
        """
        preds_list: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []
        for sel_sample in selector:
            preds_list.append(np.asarray(sel_sample["prediction"]).ravel())
            targets_list.append(np.asarray(sel_sample["target"]).ravel())
        if not preds_list:
            return None
        preds = np.stack(preds_list)
        targets = np.stack(targets_list)
        pred_mean = preds.mean(axis=0)
        bias2 = (pred_mean - targets.mean(axis=0)) ** 2
        variance = ((preds - pred_mean) ** 2).mean(axis=0)
        return bias2 + variance, bias2, variance

    @staticmethod
    def _draw_decomposition(ax: Axes, mse: np.ndarray, bias2: np.ndarray, variance: np.ndarray, colour: str) -> None:
        """Draw the three decomposition curves into one axis.

        Args:
            ax: Matplotlib axis to draw on.
            mse: Per-channel mean squared error with shape ``(N,)``.
            bias2: Per-channel squared bias with shape ``(N,)``.
            variance: Per-channel variance with shape ``(N,)``.
            colour: Method colour used for the MSE curve.
        """
        x = np.arange(len(mse))
        ax.plot(x, mse, color=colour, linewidth=2.0, label="MSE")
        ax.plot(x, bias2, color=_BIAS_COLOUR, linewidth=1.6, linestyle="--", label="Bias^2")
        ax.plot(x, variance, color=_VARIANCE_COLOUR, linewidth=1.6, linestyle=":", label="Variance")

    def _stats_figure(
        self,
        mse: np.ndarray,
        bias2: np.ndarray,
        variance: np.ndarray,
        subtitle: str,
        colour: str,
        out: Path,
    ) -> None:
        """Write one decomposition figure for a single method.

        Args:
            mse: Per-channel mean squared error with shape ``(N,)``.
            bias2: Per-channel squared bias with shape ``(N,)``.
            variance: Per-channel variance with shape ``(N,)``.
            subtitle: Plot subtitle text describing prediction and selector context.
            colour: Method colour used for the MSE curve.
            out: Destination PDF path.
        """
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        self._draw_decomposition(ax, mse, bias2, variance, colour)
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
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), sharex=True, sharey=True, squeeze=False
        )

        for idx, (label_lines, colour, mse, bias2, variance) in enumerate(methods):
            ax = axes[idx // ncols][idx % ncols]
            BiasVariancePlotter._draw_decomposition(ax, mse, bias2, variance, colour)
            ax.set_title("\n".join(label_lines), fontsize=5.5)
            ax.tick_params(labelsize=5.5)

        for ax in axes.flat[n:]:
            ax.axis("off")
        for col in range(ncols):
            used_rows = [r for r in range(nrows) if r * ncols + col < n]
            if used_rows:
                axes[used_rows[-1]][col].tick_params(labelbottom=True)

        for i in range(nrows):
            axes[i][0].set_ylabel("Loss", fontsize=8)
        for j in range(ncols):
            axes[nrows - 1][j].set_xlabel("Energy", fontsize=8)
        fig.subplots_adjust(left=0.13, right=0.99, top=0.88, bottom=0.16, wspace=0.08, hspace=0.18)
        fig.legend(
            handles=[
                Line2D([0], [0], color="gray", linewidth=2.0, label="MSE"),
                Line2D([0], [0], color=_BIAS_COLOUR, linewidth=1.6, linestyle="--", label="Bias^2"),
                Line2D([0], [0], color=_VARIANCE_COLOUR, linewidth=1.6, linestyle=":", label="Variance"),
            ],
            loc="upper center",
            ncol=3,
            fontsize=8,
            framealpha=0.9,
        )
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
