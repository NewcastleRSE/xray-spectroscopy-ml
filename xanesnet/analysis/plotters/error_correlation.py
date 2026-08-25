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

"""Plotter for cross-method per-sample error correlation."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from xanesnet.serialization.jsonl_stream import JSONLStream

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
    spectrum_error_value,
    style_axis,
)

# Label lines, colour, sample id to per-sample error.
_MethodErrors = tuple[list[str], str, dict[str, float]]


@PlotterRegistry.register("error_correlation")
class ErrorCorrelationPlotter(Plotter):
    """Plot pairwise per-sample error correlations across methods.

    For each method the per-sample error is collected and one grid figure
    draws the pairwise scatter of errors on the samples shared by each method
    pair, annotated with the Pearson correlation coefficient and the number of
    common samples. In addition, one standalone figure per method pair
    (``pair_III__JJJ.pdf``) shows the same scatter with full axis labels. The
    identity line shows which method of a pair has the larger error. High
    correlation means both methods fail on the same samples, while low
    correlation means their errors complement each other.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        sort_key: Scalar key used as the per-sample error. When ``None``, the
            MSE between predicted and target spectra is used. Collector values
            take precedence over sample scalars.
    """

    def __init__(self, plotter_type: str, sort_key: str | None = None) -> None:
        """Initialize an error correlation plotter."""
        super().__init__(plotter_type)
        self.sort_key = sort_key

    def plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write the pairwise error correlation grid and per-pair figures.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``error_correlation`` tree should be written.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        methods: list[_MethodErrors] = []

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

                errors = self._collect_errors(selector, stream)
                if not errors:
                    continue

                methods.append((label_lines, method_colour(len(methods)), errors))

        if len(methods) < 2:
            logging.info("    Need at least two methods for error correlation, skipping.")
            return

        root = output_dir / "error_correlation"
        root.mkdir(parents=True, exist_ok=True)
        self._correlation_grid(methods, root / "correlation_grid.pdf")

        for i in range(len(methods)):
            for j in range(i):
                pair = self._pair_values(methods[i][2], methods[j][2])
                if pair is None:
                    continue
                xs, ys = pair
                self._pair_figure(xs, ys, methods[i][0], methods[j][0], root / f"pair_{i:03d}__{j:03d}.pdf")

    def _collect_errors(self, selector: Selector, stream: JSONLStream | None) -> dict[str, float]:
        """Collect the per-sample error of one method keyed by sample id.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            stream: Optional collector result stream aligned with ``selector``.

        Returns:
            Mapping from sample id to the per-sample error value.
        """
        errors: dict[str, float] = {}
        if stream is not None:
            for sel_sample, col_sample in zip(selector, stream):
                errors[str(sel_sample["sample_id"])] = spectrum_error_value(sel_sample, col_sample, self.sort_key)
        else:
            for sel_sample in selector:
                errors[str(sel_sample["sample_id"])] = spectrum_error_value(sel_sample, {}, self.sort_key)
        return errors

    @staticmethod
    def _correlation_grid(methods: list[_MethodErrors], out: Path) -> None:
        """Write one grid figure with pairwise error correlation panels.

        Diagonal cells carry the method labels; lower-triangle cells compare
        the per-sample errors of the corresponding method pair.

        Args:
            methods: Method errors in first-seen order.
            out: Destination PDF path.
        """
        n = len(methods)

        fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.4 * n), sharex=True, sharey=True, squeeze=False)

        for i in range(n):
            for j in range(n):
                ax = axes[i][j]
                if i == j:
                    ax.axis("off")
                    ax.text(
                        0.5,
                        0.5,
                        "\n".join(methods[i][0]),
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=6,
                    )
                    continue
                if j > i:
                    ax.axis("off")
                    continue

                pair = ErrorCorrelationPlotter._pair_values(methods[i][2], methods[j][2])
                if pair is None:
                    ax.axis("off")
                    ax.text(
                        0.5,
                        0.5,
                        "no common samples",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="gray",
                    )
                    continue

                xs, ys = pair
                ErrorCorrelationPlotter._draw_pair_panel(ax, xs, ys, fontsize=5.5)
                ax.tick_params(labelsize=5.5)

        for j in range(n):
            axes[n - 1][j].set_xlabel("\n".join(methods[j][0]), fontsize=5.5)
        for i in range(n):
            axes[i][0].set_ylabel("\n".join(methods[i][0]), fontsize=5.5)

        compact_layout(fig, rect=(0.03, 0.04, 1.0, 0.99))
        add_subtitle(fig, "lower panels: error of row method vs error of column method on common samples")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _pair_values(errors_i: dict[str, float], errors_j: dict[str, float]) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the error arrays for the samples shared by two methods.

        Args:
            errors_i: Per-sample errors of the row method.
            errors_j: Per-sample errors of the column method.

        Returns:
            ``(xs, ys)`` error arrays for the common samples, or ``None`` when
            fewer than two samples are shared.
        """
        common = sorted(set(errors_i) & set(errors_j))
        if len(common) < 2:
            return None
        return np.array([errors_i[s] for s in common]), np.array([errors_j[s] for s in common])

    @staticmethod
    def _draw_pair_panel(ax: Axes, xs: np.ndarray, ys: np.ndarray, fontsize: float) -> None:
        """Draw one pairwise error scatter with identity line and annotation.

        Args:
            ax: Matplotlib axis to draw on.
            xs: Error values of the row method with shape ``(M,)``.
            ys: Error values of the column method with shape ``(M,)``.
            fontsize: Font size used for the annotation text.
        """
        lo = min(float(xs.min()), float(ys.min()))
        hi = max(float(xs.max()), float(ys.max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, linestyle="--")
        ax.scatter(xs, ys, s=6, alpha=0.5, color="#005186")
        r = _pearson(xs, ys)
        note = f"r={r:.2f}" if r is not None else "r=n/a"
        ax.text(
            0.02,
            0.98,
            f"{note}\nn={len(xs)}",
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment="top",
        )

    @staticmethod
    def _pair_figure(
        xs: np.ndarray,
        ys: np.ndarray,
        row_label_lines: list[str],
        col_label_lines: list[str],
        out: Path,
    ) -> None:
        """Write one standalone correlation figure for a single method pair.

        Args:
            xs: Error values of the row method with shape ``(M,)``.
            ys: Error values of the column method with shape ``(M,)``.
            row_label_lines: Label lines of the row method.
            col_label_lines: Label lines of the column method.
            out: Destination PDF path.
        """
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        ErrorCorrelationPlotter._draw_pair_panel(ax, xs, ys, fontsize=8)
        ax.set_xlabel("\n".join(col_label_lines))
        ax.set_ylabel("\n".join(row_label_lines))
        style_axis(ax)
        add_subtitle(fig, f"{row_label_lines[0]} (y) vs {col_label_lines[0]} (x) on common samples")
        compact_layout(fig)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)


def _pearson(xs: np.ndarray, ys: np.ndarray) -> float | None:
    """Compute the Pearson correlation coefficient of two error vectors.

    Args:
        xs: Error values of the first method with shape ``(M,)``.
        ys: Error values of the second method with shape ``(M,)``.

    Returns:
        Correlation coefficient in ``[-1, 1]``, or ``None`` when either
        vector is constant.
    """
    xc = xs - xs.mean()
    yc = ys - ys.mean()
    denom = float(np.sqrt((xc**2).sum() * (yc**2).sum()))
    if denom == 0:
        return None
    return float((xc * yc).sum() / denom)
