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

"""Plotter for mean-spectrum regression diagnostics."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from ..result import AnalysisResults
from .base import Plotter
from .common import (
    COLOUR_ACCENT_GREEN,
    COLOUR_ACCENT_RED,
    COLOUR_TARGET,
    add_subtitle,
    compact_layout,
    method_colour,
    style_axis,
)
from .registry import PlotterRegistry

# Method mean statistics: label lines, colour, mean target, mean prediction,
# target std, prediction std, best-percent prediction mean/std, worst-percent
# prediction mean/std.
_MethodMeans = tuple[
    list[str], str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


@PlotterRegistry.register("mean_spectrum")
class MeanSpectrumPlotter(Plotter):
    """Plot mean predicted and target spectra plus best/worst tail means.

    The averaged curves come from the ``spectrum`` aggregator, while the best
    and worst tail means and spreads come from the ``ranking`` aggregator. Both
    must be present in the analysis configuration; the ``ranking`` aggregator
    also owns the ``percent`` and ``sort_key`` options that define the tails.

    For every (prediction-reader, selector) pair this plotter writes three
    figures. ``mean_spectrum.pdf`` shows the mean predicted and mean target
    spectrum with the per-channel prediction standard deviation as a shaded
    band. ``tail_means.pdf`` compares the mean target and mean prediction with
    the mean prediction of the best and worst samples ranked by error.
    ``spread.pdf`` compares the per-channel standard deviation of the
    predictions with that of the targets.

    When the best and worst tail means both collapse onto the mean prediction,
    or the prediction spread stays well below the target spread, the model
    only predicts the mean spectrum.

    Three combined figures compare all methods: an overlay of the mean
    spectra plus separate overlays for the best and worst tail means. Each
    combined figure also shows the per-method prediction and target means,
    with the line styles explained in the legends.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
    """

    def __init__(self, plotter_type: str) -> None:
        """Initialize a mean-spectrum plotter."""
        super().__init__(plotter_type)

    def plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write per-method mean-spectrum PDFs and combined overlays.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``mean_spectra`` tree should be written.

        Raises:
            ConfigError: If no ``spectrum`` or ``ranking`` aggregator is configured.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        root = output_dir / "mean_spectra"
        root.mkdir(parents=True, exist_ok=True)

        methods: list[_MethodMeans] = []
        percent = 0.0

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx in range(len(reader_selectors)):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                label = results.method_label(reader_idx, sel_idx)

                spectrum_data = results.aggregation(reader_idx, sel_idx, "spectrum").data
                ranking_data = results.aggregation(reader_idx, sel_idx, "ranking").data
                if not spectrum_data or not ranking_data:
                    continue

                target_mean = spectrum_data["target_mean"]
                pred_mean = spectrum_data["prediction_mean"]
                target_std = spectrum_data["target_std"]
                pred_std = spectrum_data["prediction_std"]
                best_mean = ranking_data["best_prediction_mean"]
                best_std = ranking_data["best_prediction_std"]
                worst_mean = ranking_data["worst_prediction_mean"]
                worst_std = ranking_data["worst_prediction_std"]
                percent = float(ranking_data["percent"])

                colour = method_colour(len(methods))
                methods.append(
                    (
                        label.lines,
                        colour,
                        target_mean,
                        pred_mean,
                        target_std,
                        pred_std,
                        best_mean,
                        best_std,
                        worst_mean,
                        worst_std,
                    )
                )

                combo_dir = root / label.dir_name
                combo_dir.mkdir(parents=True, exist_ok=True)
                subtitle = "\n".join(label.lines)

                self._mean_figure(target_mean, pred_mean, pred_std, subtitle, colour, combo_dir / "mean_spectrum.pdf")
                self._tail_figure(
                    target_mean,
                    pred_mean,
                    best_mean,
                    best_std,
                    worst_mean,
                    worst_std,
                    percent,
                    subtitle,
                    colour,
                    combo_dir / "tail_means.pdf",
                )
                self._spread_figure(target_std, pred_std, subtitle, colour, combo_dir / "spread.pdf")

        if not methods:
            logging.info("    No samples selected, skipping.")
            return

        combined_dir = root / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        self._combined_mean_overlay(methods, combined_dir / "mean_spectrum_overlay.pdf")
        self._combined_tail_overlay(methods, percent, combined_dir / "tail_best_overlay.pdf", "best")
        self._combined_tail_overlay(methods, percent, combined_dir / "tail_worst_overlay.pdf", "worst")

    @staticmethod
    def _tail_figure(
        target_mean: np.ndarray,
        pred_mean: np.ndarray,
        best_mean: np.ndarray,
        best_std: np.ndarray,
        worst_mean: np.ndarray,
        worst_std: np.ndarray,
        percent: float,
        subtitle: str,
        colour: str,
        out: Path,
    ) -> None:
        """Write one figure comparing the mean prediction with its best/worst tails.

        Args:
            target_mean: Per-channel mean target spectrum.
            pred_mean: Per-channel mean predicted spectrum.
            best_mean: Per-channel mean prediction of the best ``percent`` samples.
            best_std: Per-channel prediction std of the best ``percent`` samples.
            worst_mean: Per-channel mean prediction of the worst ``percent`` samples.
            worst_std: Per-channel prediction std of the worst ``percent`` samples.
            percent: Tail size in percent, as configured on the aggregator.
            subtitle: Plot subtitle text describing prediction and selector context.
            colour: Method colour used for the mean prediction curve.
            out: Destination PDF path.
        """
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        x = np.arange(len(pred_mean))
        ax.plot(x, target_mean, color=COLOUR_TARGET, linewidth=1.4, linestyle=":", label="Target mean")
        ax.plot(x, pred_mean, color=colour, linewidth=2.0, label="Total prediction mean")
        ax.fill_between(
            x,
            best_mean - best_std,
            best_mean + best_std,
            color=COLOUR_ACCENT_GREEN,
            alpha=0.15,
            linewidth=0,
            label=f"Best {percent:g}% +/- 1 std",
        )
        ax.plot(x, best_mean, color=COLOUR_ACCENT_GREEN, linewidth=1.8, label=f"Best {percent:g}% mean")
        ax.fill_between(
            x,
            worst_mean - worst_std,
            worst_mean + worst_std,
            color=COLOUR_ACCENT_RED,
            alpha=0.15,
            linewidth=0,
            label=f"Worst {percent:g}% +/- 1 std",
        )
        ax.plot(x, worst_mean, color=COLOUR_ACCENT_RED, linewidth=1.8, label=f"Worst {percent:g}% mean")
        ax.set_xlabel("Energy")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=8, framealpha=0.9)
        style_axis(ax)
        add_subtitle(fig, subtitle)
        compact_layout(fig)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _mean_figure(
        target_mean: np.ndarray,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        subtitle: str,
        colour: str,
        out: Path,
    ) -> None:
        """Write one figure comparing the mean prediction and mean target spectra.

        Args:
            target_mean: Per-channel mean target spectrum.
            pred_mean: Per-channel mean predicted spectrum.
            pred_std: Per-channel prediction standard deviation.
            subtitle: Plot subtitle text describing prediction and selector context.
            colour: Method colour used for the prediction curve.
            out: Destination PDF path.
        """
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        x = np.arange(len(pred_mean))
        ax.plot(x, target_mean, color=COLOUR_TARGET, linewidth=1.6, label="Target mean")
        ax.fill_between(
            x,
            pred_mean - pred_std,
            pred_mean + pred_std,
            color=colour,
            alpha=0.25,
            linewidth=0,
            label="Prediction +/- 1 std",
        )
        ax.plot(x, pred_mean, color=colour, linewidth=2.0, label="Prediction mean")
        ax.set_xlabel("Energy")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=8, framealpha=0.9)
        style_axis(ax)
        add_subtitle(fig, subtitle)
        compact_layout(fig)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _spread_figure(
        target_std: np.ndarray,
        pred_std: np.ndarray,
        subtitle: str,
        colour: str,
        out: Path,
    ) -> None:
        """Write one figure comparing per-channel prediction and target spread.

        A prediction spread consistently below the target spread indicates
        that the model predicts the mean spectrum.

        Args:
            target_std: Per-channel target standard deviation.
            pred_std: Per-channel prediction standard deviation.
            subtitle: Plot subtitle text describing prediction and selector context.
            colour: Method colour used for the prediction spread curve.
            out: Destination PDF path.
        """
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        x = np.arange(len(pred_std))
        ax.plot(x, target_std, color=COLOUR_TARGET, linewidth=1.8, label="Target std")
        ax.plot(x, pred_std, color=colour, linewidth=1.8, label="Prediction std")
        ax.set_xlabel("Energy")
        ax.set_ylabel("Standard deviation")
        ax.legend(fontsize=8, framealpha=0.9)
        style_axis(ax)
        add_subtitle(fig, subtitle)
        compact_layout(fig)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _legend_with_styles(ax: Axes, extra: list[tuple[str, str]]) -> None:
        """Rebuild the axis legend with grey style entries appended.

        Args:
            ax: Matplotlib axis whose legend should be extended.
            extra: ``(label, linestyle)`` pairs to append as grey proxy entries.
        """
        handles, labels = ax.get_legend_handles_labels()
        existing = set(labels)
        for label, linestyle in extra:
            if label in existing:
                continue
            handles.append(Line2D([0], [0], color="gray", linewidth=1.4, linestyle=linestyle, label=label))
            labels.append(label)
        ax.legend(handles=handles, labels=labels, fontsize=8, framealpha=0.9)

    @staticmethod
    def _shared_target_mean(methods: list[_MethodMeans]) -> np.ndarray | None:
        """Return the common target mean when all methods share one.

        Args:
            methods: Method mean statistics in first-seen order.

        Returns:
            The shared per-channel target mean, or ``None`` when the methods
            have different target means.
        """
        first = methods[0][2]
        if all(np.allclose(first, method[2]) for method in methods[1:]):
            return first
        return None

    @staticmethod
    def _combined_mean_overlay(methods: list[_MethodMeans], out: Path) -> None:
        """Write one combined figure overlaying all mean prediction/target spectra.

        When all methods share the same target mean it is drawn once as a
        grey dotted line; otherwise one dotted target curve per method is
        drawn in the method colour.

        Args:
            methods: Method mean statistics in first-seen order.
            out: Destination PDF path.
        """
        fig, ax = plt.subplots(figsize=(7, 4.2))
        shared_target = MeanSpectrumPlotter._shared_target_mean(methods)
        if shared_target is not None:
            ax.plot(
                np.arange(len(shared_target)),
                shared_target,
                color=COLOUR_TARGET,
                linewidth=1.6,
                linestyle=":",
                label="Target mean",
            )
        for label_lines, colour, target_mean, pred_mean, *_ in methods:
            x = np.arange(len(pred_mean))
            if shared_target is None:
                ax.plot(x, target_mean, color=colour, linewidth=1.4, linestyle=":")
            ax.plot(x, pred_mean, color=colour, linewidth=2.0, label="\n".join(label_lines))
        ax.set_xlabel("Energy")
        ax.set_ylabel("Intensity")
        MeanSpectrumPlotter._legend_with_styles(ax, [("Prediction mean", "-"), ("Target mean", ":")])
        style_axis(ax)
        compact_layout(fig)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _combined_tail_overlay(methods: list[_MethodMeans], percent: float, out: Path, direction: str) -> None:
        """Write one combined figure overlaying per-method tail, prediction, and target means.

        When all methods share the same target mean it is drawn once as a
        grey dotted line; otherwise one dotted target curve per method is
        drawn in the method colour.

        Args:
            methods: Method mean statistics in first-seen order.
            percent: Tail size in percent, as configured on the aggregator.
            out: Destination PDF path.
            direction: ``"best"`` or ``"worst"``, selecting which tail mean is
                drawn as the solid per-method curve.
        """
        fig, ax = plt.subplots(figsize=(7, 4.2))
        shared_target = MeanSpectrumPlotter._shared_target_mean(methods)
        if shared_target is not None:
            ax.plot(
                np.arange(len(shared_target)),
                shared_target,
                color=COLOUR_TARGET,
                linewidth=1.6,
                linestyle=":",
                label="Target mean",
            )
        for label_lines, colour, target_mean, pred_mean, _, _, best_mean, _, worst_mean, _ in methods:
            tail_mean = best_mean if direction == "best" else worst_mean
            x = np.arange(len(tail_mean))
            if shared_target is None:
                ax.plot(x, target_mean, color=colour, linewidth=1.4, linestyle=":")
            ax.plot(x, pred_mean, color=colour, linewidth=1.6, linestyle="--")
            ax.plot(x, tail_mean, color=colour, linewidth=2.0, label="\n".join(label_lines))
        ax.set_xlabel("Energy")
        ax.set_ylabel("Intensity")
        MeanSpectrumPlotter._legend_with_styles(
            ax,
            [
                (f"{direction.capitalize()} {percent:g}% mean", "-"),
                ("Total prediction mean", "--"),
                ("Target mean", ":"),
            ],
        )
        style_axis(ax)
        compact_layout(fig)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
