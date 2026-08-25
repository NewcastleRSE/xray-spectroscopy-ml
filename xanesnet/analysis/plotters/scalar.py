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

"""Plotter for scalar value distributions."""

import logging
import math
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from xanesnet.analysis.utils import ScalarValue
from xanesnet.serialization.jsonl_stream import JSONLStream

from ..reporters.base import selector_label
from ..result import AnalysisResults
from .base import Plotter
from .registry import PlotterRegistry
from .utils import (
    add_subtitle,
    collect_scalar_values,
    compact_layout,
    method_colour,
    method_label_lines,
    style_axis,
)

_CLIP_IQR_SCALE: float = 1.5
_CLIP_MIN_SAMPLES: int = 20


@PlotterRegistry.register("scalar")
class ScalarPlotter(Plotter):
    """Create distribution plots and combined figures for scalar value keys.

    For each (prediction-reader, selector) combination a per-key directory is
    created containing a histogram, box plot, and violin plot. In addition,
    four combined figures per scalar key compare all prediction-reader/selector
    combinations, styled after ``StatTablePlotter``: a per-method histogram
    grid, a grouped histogram whose bars sit side by side per bin, a combined
    box plot, and a combined violin plot.

    Histogram bars report the percentage share of in-range samples per bin so
    distributions stay comparable across differently sized samples. Histograms
    display a robust Tukey-fence range by default: values outside
    ``Q1 - 1.5 * IQR`` to ``Q3 + 1.5 * IQR`` are excluded from the plotted
    range and the number of excluded values is annotated on the figure. Box
    and violin plots always display the full value range. Summary statistics
    are always computed over the full sample.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        bins: Number of histogram bins. Defaults to ``DEFAULT_BINS``.
    """

    DEFAULT_BINS: ClassVar[int] = 50

    def __init__(self, plotter_type: str, bins: int | None = None) -> None:
        """Initialize a scalar distribution plotter."""
        super().__init__(plotter_type)
        self.bins = bins if bins is not None else self.DEFAULT_BINS

    def plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Create per-method distribution plots and combined per-key figures.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``scalar_plots`` tree should be written.
        """
        root = output_dir / "scalar_plots"

        reader_names = results.prediction_names
        series: list[tuple[int, int, str, list[str], str, dict[str, list[ScalarValue]]]] = []
        key_order: list[str] = []

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                sel_label_str = selector_label(results.selectors_config, sel_idx)
                sel_cfg = results.selectors_config[sel_idx]
                label_lines = method_label_lines(reader_names[reader_idx], sel_label_str, sel_cfg)

                stream: JSONLStream | None = None
                if reader_idx < len(results.collector_results) and sel_idx < len(results.collector_results[reader_idx]):
                    stream = results.collector_results[reader_idx][sel_idx]

                values_by_key = collect_scalar_values(selector, stream)
                if not values_by_key:
                    continue

                colour = method_colour(len(series))
                series.append((reader_idx, sel_idx, sel_label_str, label_lines, colour, values_by_key))
                for key in values_by_key:
                    if key not in key_order:
                        key_order.append(key)

        if not series:
            logging.info("    No scalar values collected, skipping.")
            return

        for reader_idx, sel_idx, sel_label_str, label_lines, colour, values_by_key in series:
            combo_label = f"pred_{reader_idx:03d}__sel_{sel_idx:03d}_{sel_label_str}"
            combo_dir = root / combo_label
            subtitle = "\n".join(label_lines)

            for key, vals in values_by_key.items():
                key_dir = combo_dir / key
                key_dir.mkdir(parents=True, exist_ok=True)
                arr = np.array(vals)

                self._histogram(arr, key, subtitle, colour, key_dir)
                self._boxplot(arr, key, subtitle, colour, key_dir)
                self._violin(arr, key, subtitle, colour, key_dir)

        combined_dir = root / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        for key in key_order:
            key_series: list[tuple[str, list[str], np.ndarray]] = []
            for _, _, _, label_lines, colour, values_by_key in series:
                if key in values_by_key:
                    key_series.append((colour, label_lines, np.array(values_by_key[key])))
            self._combined_grid(key, key_series, combined_dir)
            self._combined_grouped(key, key_series, combined_dir)
            self._combined_boxplot(key, key_series, combined_dir)
            self._combined_violin(key, key_series, combined_dir)

    def _histogram(self, arr: np.ndarray, key: str, subtitle: str, colour: str, out: Path) -> None:
        """Write a histogram PDF for one scalar key.

        Bar heights report the percentage share of in-range samples per bin.

        Args:
            arr: One-dimensional scalar values with shape ``(N,)``.
            key: Scalar value key used for labels and output path context.
            subtitle: Plot subtitle text describing prediction and selector context.
            colour: Method colour used for the histogram bars.
            out: Directory where ``histogram.pdf`` should be written.
        """
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        clip = _clip_limits(arr)
        lo, hi = _bin_range(arr, clip)
        heights, edges = _share_heights(arr, self.bins, lo, hi)
        width = edges[1] - edges[0]
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, heights, width=width, color=colour, alpha=0.55)
        ax.set_xlabel(key)
        ax.set_ylabel("Share (%)")
        style_axis(ax)
        add_subtitle(fig, subtitle)
        _add_clip_note(ax, clip)
        _add_stats_text(ax, arr)
        compact_layout(fig)
        fig.savefig(out / "histogram.pdf", bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _boxplot(arr: np.ndarray, key: str, subtitle: str, colour: str, out: Path) -> None:
        """Write a box plot PDF for one scalar key.

        The full value range is always displayed; unlike histograms, box plots
        do not clip outliers because the box itself is quartile-based.

        Args:
            arr: One-dimensional scalar values with shape ``(N,)``.
            key: Scalar value key used for labels and output path context.
            subtitle: Plot subtitle text describing prediction and selector context.
            colour: Method colour used for the box fill.
            out: Directory where ``boxplot.pdf`` should be written.
        """
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        bp = ax.boxplot(arr, vert=True, patch_artist=True)
        bp["boxes"][0].set_facecolor(colour)
        bp["boxes"][0].set_alpha(0.7)
        ax.set_ylabel(key)
        ax.set_xticklabels([""])
        style_axis(ax)
        add_subtitle(fig, subtitle)
        compact_layout(fig)
        fig.savefig(out / "boxplot.pdf", bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _violin(arr: np.ndarray, key: str, subtitle: str, colour: str, out: Path) -> None:
        """Write a violin plot PDF for one scalar key.

        The full value range is always displayed so the complete distribution
        shape, including outliers, remains visible.

        Args:
            arr: One-dimensional scalar values with shape ``(N,)``.
            key: Scalar value key used for labels and output path context.
            subtitle: Plot subtitle text describing prediction and selector context.
            colour: Method colour used for the violin bodies.
            out: Directory where ``violin.pdf`` should be written.
        """
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        vp = ax.violinplot(arr, showmedians=True, showextrema=True)
        bodies = vp["bodies"]
        assert isinstance(bodies, list)
        for body in bodies:
            body.set_facecolor(colour)
            body.set_alpha(0.7)
        ax.set_ylabel(key)
        ax.set_xticks([1])
        ax.set_xticklabels([""])
        style_axis(ax)
        add_subtitle(fig, subtitle)
        compact_layout(fig)
        fig.savefig(out / "violin.pdf", bbox_inches="tight")
        plt.close(fig)

    def _combined_grid(
        self,
        key: str,
        series: list[tuple[str, list[str], np.ndarray]],
        out: Path,
    ) -> None:
        """Write one combined figure with a per-method histogram grid.

        All cells share one x range, one y range, and one clip range so the
        cells stay directly comparable. Each cell annotates how many of its
        own values fall outside the shared display range.

        Args:
            key: Scalar value key used for labels and output path context.
            series: ``(colour, label lines, values)`` per method in first-seen order.
            out: Directory where the combined grid PDF should be written.
        """
        n = len(series)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

        all_values = np.concatenate([arr for _, _, arr in series])
        clip = _clip_limits(all_values)
        lo, hi = _bin_range(all_values, clip)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.0 * ncols, 2.2 * nrows), sharex=True, sharey=True, squeeze=False
        )

        for idx, (colour, label_lines, arr) in enumerate(series):
            ax = axes[idx // ncols][idx % ncols]
            heights, edges = _share_heights(arr, self.bins, lo, hi)
            width = edges[1] - edges[0]
            centers = (edges[:-1] + edges[1:]) / 2
            ax.bar(centers, heights, width=width, color=colour, alpha=0.8)
            ax.set_title("\n".join(label_lines), fontsize=5.5)
            ax.tick_params(labelsize=5.5)
            if clip is not None:
                clipped_i = int(np.count_nonzero((arr < lo) | (arr > hi)))
                _add_cell_clip_note(ax, (lo, hi, clipped_i))

        for ax in axes.flat[n:]:
            ax.axis("off")
        for col in range(ncols):
            used_rows = [r for r in range(nrows) if r * ncols + col < n]
            if used_rows:
                axes[used_rows[-1]][col].tick_params(labelbottom=True)

        if clip is not None:
            ymin, ymax = axes[0][0].get_ylim()
            axes[0][0].set_ylim(ymin, ymax * 1.2)  # headroom for the per-cell notes

        for i in range(nrows):
            axes[i][0].set_ylabel("Share (%)", fontsize=8)
        for j in range(ncols):
            axes[nrows - 1][j].set_xlabel(key, fontsize=8)
        fig.subplots_adjust(left=0.16, right=0.99, top=0.95, bottom=0.20, wspace=0.08, hspace=0.18)
        fig.savefig(out / f"{key}_grid.pdf", bbox_inches="tight")
        plt.close(fig)

    def _combined_grouped(
        self,
        key: str,
        series: list[tuple[str, list[str], np.ndarray]],
        out: Path,
    ) -> None:
        """Write one combined figure with side-by-side per-method share bars.

        One bar per method sits next to the others inside each bin so the
        distributions stay comparable without stacking.

        Args:
            key: Scalar value key used for labels and output path context.
            series: ``(colour, label lines, values)`` per method in first-seen order.
            out: Directory where the combined grouped PDF should be written.
        """
        all_values = np.concatenate([arr for _, _, arr in series])
        clip = _clip_limits(all_values)
        lo, hi = _bin_range(all_values, clip)
        bins = np.linspace(lo, hi, self.bins + 1)
        width = bins[1] - bins[0]
        centers = (bins[:-1] + bins[1:]) / 2

        n_methods = len(series)
        sub_width = width / n_methods
        offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * sub_width

        fig, ax = plt.subplots(figsize=(7, 4.2))
        for idx, (colour, label_lines, arr) in enumerate(series):
            heights, _ = _share_heights(arr, self.bins, lo, hi)
            ax.bar(
                centers + offsets[idx],
                heights,
                width=sub_width,
                color=colour,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.4,
                label="\n".join(label_lines),
            )
        ax.set_xlabel(key)
        ax.set_ylabel("Share (%)")
        _add_clip_note(ax, clip)
        ax.legend(fontsize=8, framealpha=0.9)
        style_axis(ax)
        compact_layout(fig)
        fig.savefig(out / f"{key}_grouped.pdf", bbox_inches="tight")
        plt.close(fig)

    def _combined_boxplot(
        self,
        key: str,
        series: list[tuple[str, list[str], np.ndarray]],
        out: Path,
    ) -> None:
        """Write one combined figure with side-by-side per-method box plots.

        All boxes share one axis so they stay directly comparable. The full
        value range is always displayed.

        Args:
            key: Scalar value key used for labels and output path context.
            series: ``(colour, label lines, values)`` per method in first-seen order.
            out: Directory where the combined box plot PDF should be written.
        """
        n = len(series)

        fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * n), 4.2))
        bp = ax.boxplot([arr for _, _, arr in series], positions=range(n), vert=True, patch_artist=True)
        for patch, (colour, _, _) in zip(bp["boxes"], series):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)
        ax.set_xticks(range(n))
        _set_method_xticklabels(ax, [label_lines for _, label_lines, _ in series])
        ax.set_ylabel(key)
        style_axis(ax)
        compact_layout(fig)
        fig.savefig(out / f"{key}_boxplot.pdf", bbox_inches="tight")
        plt.close(fig)

    def _combined_violin(
        self,
        key: str,
        series: list[tuple[str, list[str], np.ndarray]],
        out: Path,
    ) -> None:
        """Write one combined figure with side-by-side per-method violin plots.

        All violins share one axis so they stay directly comparable. The full
        value range is always displayed.

        Args:
            key: Scalar value key used for labels and output path context.
            series: ``(colour, label lines, values)`` per method in first-seen order.
            out: Directory where the combined violin PDF should be written.
        """
        n = len(series)

        fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * n), 4.2))
        vp = ax.violinplot([arr for _, _, arr in series], positions=range(n), showmedians=True, showextrema=True)
        bodies = vp["bodies"]
        assert isinstance(bodies, list)
        for body, (colour, _, _) in zip(bodies, series):
            body.set_facecolor(colour)
            body.set_alpha(0.7)
        ax.set_xticks(range(n))
        _set_method_xticklabels(ax, [label_lines for _, label_lines, _ in series])
        ax.set_ylabel(key)
        style_axis(ax)
        compact_layout(fig)
        fig.savefig(out / f"{key}_violin.pdf", bbox_inches="tight")
        plt.close(fig)


def _bin_range(arr: np.ndarray, clip: tuple[float, float, int] | None) -> tuple[float, float]:
    """Return lower and upper bin edges for one sample.

    Uses the clipped fence bounds when available and otherwise the data range,
    padded for degenerate data.

    Args:
        arr: One-dimensional scalar values with shape ``(N,)``.
        clip: Clip bounds from :func:`_clip_limits` or ``None``.

    Returns:
        ``(lo, hi)`` bin edges with ``lo < hi``.
    """
    if clip is not None:
        return clip[0], clip[1]
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if not lo < hi:
        pad = 0.5 * max(abs(lo), 1.0)
        lo -= pad
        hi += pad
    return lo, hi


def _share_heights(arr: np.ndarray, n_bins: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bin percentage shares for one sample.

    Args:
        arr: One-dimensional scalar values with shape ``(N,)``.
        n_bins: Number of histogram bins.
        lo: Lower bin edge.
        hi: Upper bin edge.

    Returns:
        ``(heights, edges)`` where ``heights`` holds the percentage share of
        in-range samples per bin and ``edges`` the ``n_bins + 1`` bin edges.
    """
    counts, edges = np.histogram(arr, bins=n_bins, range=(lo, hi))
    n_in = int(counts.sum())
    heights = counts / n_in * 100.0 if n_in > 0 else np.zeros_like(counts, dtype=float)
    return heights, edges


def _clip_limits(arr: np.ndarray) -> tuple[float, float, int] | None:
    """Compute robust display limits and the number of values outside them.

    Uses Tukey inner fences (``Q1 - 1.5 * IQR`` to ``Q3 + 1.5 * IQR``) so that
    outliers do not compress the interesting range; robust even when outliers
    make up a few percent of the sample. No values are removed; the clipped
    count is only used for on-figure annotations and summary statistics always
    use all values.

    Args:
        arr: One-dimensional scalar values with shape ``(N,)``.

    Returns:
        ``(lo, hi, clipped)`` fence bounds plus the number of values outside
        them, or ``None`` when clipping is not useful (too few samples or a
        degenerate interquartile range).
    """
    if len(arr) < _CLIP_MIN_SAMPLES:
        return None
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    spread = _CLIP_IQR_SCALE * (q3 - q1)
    lo = float(q1 - spread)
    hi = float(q3 + spread)
    if not lo < hi:
        return None
    clipped = int(np.count_nonzero((arr < lo) | (arr > hi)))
    return lo, hi, clipped


def _clip_note(clip: tuple[float, float, int] | None) -> str:
    """Describe how many values fall outside the clipped display range.

    Args:
        clip: Clip bounds from :func:`_clip_limits` or ``None``.

    Returns:
        Annotation text, or an empty string when no clipping applies.
    """
    if clip is None or clip[2] == 0:
        return ""
    lo, hi, clipped = clip
    return f"{clipped} values outside [{lo:.4g}, {hi:.4g}] not shown"


def _add_clip_note(ax: Axes, clip: tuple[float, float, int] | None) -> None:
    """Annotate an axis with the number of clipped values, when any.

    The note is anchored to the right end of the title line. The main titles
    are left-aligned so the note and the title never overlap each other or the
    plotted data.

    Args:
        ax: Matplotlib axis to annotate.
        clip: Clip bounds from :func:`_clip_limits` or ``None``.
    """
    note = _clip_note(clip)
    if not note:
        return
    ax.set_title(note, loc="right", fontsize=6.5, color="#e85651")


def _add_cell_clip_note(ax: Axes, clip: tuple[float, float, int] | None) -> None:
    """Annotate one grid cell with the number of clipped values, when any.

    The note sits in the top-left corner of the cell; callers reserve headroom
    above the bars so it never overlaps them.

    Args:
        ax: Matplotlib axis to annotate.
        clip: Clip bounds from :func:`_clip_limits` or ``None``.
    """
    note = _clip_note(clip)
    if not note:
        return
    ax.text(
        0.02,
        0.99,
        note,
        transform=ax.transAxes,
        fontsize=5,
        verticalalignment="top",
        horizontalalignment="left",
        color="#e85651",
    )


def _set_method_xticklabels(ax: Axes, label_lines: list[list[str]]) -> None:
    """Label method tick positions with stacked label lines.

    Short label sets are horizontal; long sets are rotated so they do not
    overlap.

    Args:
        ax: Matplotlib axis to annotate.
        label_lines: Label lines per method in tick order.
    """
    labels = ["\n".join(lines) for lines in label_lines]
    if len(labels) <= 5:
        ax.set_xticklabels(labels, fontsize=7)
    else:
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)


def _add_stats_text(ax: Axes, arr: np.ndarray) -> None:
    """Add sample count and summary statistics to a plot axis.

    Args:
        ax: Matplotlib axis to annotate.
        arr: One-dimensional scalar values with shape ``(N,)``.
    """
    text = f"n={len(arr)}\n" f"mean={np.mean(arr):.4g}\n" f"std={np.std(arr):.4g}\n" f"median={np.median(arr):.4g}"
    ax.text(
        0.97,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#bbbbbb"),
    )
