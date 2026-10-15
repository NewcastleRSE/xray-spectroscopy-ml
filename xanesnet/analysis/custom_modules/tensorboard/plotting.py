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
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Render TensorBoard training histories with XANESNET plot styling."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...plotters.common.layout import save_figure
from ...plotters.common.style import (
    PlotStyle,
    add_legend,
    method_color,
    set_context_title,
    style_axis,
)
from .events import TrainingHistory, minimum_validation_point


def _create_axes(
    style: PlotStyle,
    title: str,
    y_scale: str,
    double_width: bool,
) -> tuple[Figure, Axes, Axes]:
    """Create a styled two-panel training-curve figure.

    Args:
        style: Shared plotter style profile.
        title: Context title shown above the loss axis.
        y_scale: Vertical-axis scale, either ``"linear"`` or ``"log"``.
        double_width: Whether to double the base figure width.

    Returns:
        Figure, loss axis, and learning-rate axis.
    """
    fig, axes = plt.subplots(
        2,
        1,
        figsize=style.figure_size("energy", width=12.8 if double_width else None, height=6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.0]},
    )
    loss_axis, learning_rate_axis = axes
    fig.subplots_adjust(left=0.15, right=0.96, top=0.92, bottom=0.13, hspace=0.08)
    style_axis(loss_axis, style)
    style_axis(learning_rate_axis, style)
    loss_axis.set_yscale(y_scale)
    loss_axis.set_ylabel("Loss")
    learning_rate_axis.set_ylabel("Learning rate")
    learning_rate_axis.set_xlabel("Epoch")
    set_context_title(loss_axis, title, style)
    return fig, loss_axis, learning_rate_axis


def _draw_history(
    loss_axis: Axes,
    learning_rate_axis: Axes,
    history: TrainingHistory,
    name: str,
    color: str,
    style: PlotStyle,
    combined: bool,
) -> None:
    """Draw one run's loss and learning-rate series.

    Args:
        loss_axis: Axis for training and validation loss.
        learning_rate_axis: Axis for learning rate.
        history: Metric series to draw.
        name: Prediction name used in the combined-plot legend.
        color: Method color used for this run.
        style: Shared plotter style profile.
        combined: Whether labels and minimum lines are for the combined plot.
    """
    training_label = f"{name} training" if combined else "Training loss"
    validation_label = f"{name} validation" if combined else "Validation loss"
    learning_rate_label = name if combined else "Learning rate"
    loss_axis.plot(
        history.training_loss.epochs,
        history.training_loss.values,
        color=color,
        linewidth=style.linewidth("main"),
        label=training_label,
    )
    loss_axis.plot(
        history.validation_loss.epochs,
        history.validation_loss.values,
        color=color,
        linestyle="--",
        linewidth=style.linewidth("secondary"),
        label=validation_label,
    )
    learning_rate_axis.plot(
        history.learning_rate.epochs,
        history.learning_rate.values,
        color=color,
        linewidth=style.linewidth("main"),
        label=learning_rate_label,
    )

    best_epoch, best_loss = minimum_validation_point(history)
    reference_color = color if combined else "#000000"
    reference_label = "_nolegend_" if combined else None
    loss_axis.axvline(
        best_epoch,
        color=reference_color,
        linestyle=":",
        linewidth=style.linewidth("reference"),
        label=f"Best validation epoch: {best_epoch}" if not combined else reference_label,
    )
    loss_axis.axhline(
        best_loss,
        color=reference_color,
        linestyle="--",
        linewidth=style.linewidth("reference"),
        label=f"Minimum validation loss: {best_loss:.4g}" if not combined else reference_label,
    )
    loss_axis.plot(
        best_epoch,
        best_loss,
        color=color,
        marker="o",
        markersize=5.0 * style.marker_scale,
        markeredgecolor="white",
        markeredgewidth=0.6,
        linestyle="None",
        label="_nolegend_",
    )


def plot_single_run(
    name: str,
    history: TrainingHistory,
    style: PlotStyle,
    output_path: Path,
    y_scale: str,
    double_width: bool,
) -> None:
    """Save one inference run's loss and learning-rate curves.

    Args:
        name: Prediction name shown above the figure.
        history: Training metrics read from TensorBoard.
        style: Shared plotter style profile.
        output_path: Destination PDF path.
        y_scale: Vertical-axis scale, either ``"linear"`` or ``"log"``.
        double_width: Whether to double the base figure width.
    """
    fig, loss_axis, learning_rate_axis = _create_axes(style, name, y_scale, double_width)
    _draw_history(loss_axis, learning_rate_axis, history, name, "#000000", style, combined=False)
    add_legend(loss_axis, style)
    add_legend(learning_rate_axis, style)
    save_figure(fig, output_path, style)


def plot_combined_runs(
    runs: list[tuple[str, TrainingHistory]],
    style: PlotStyle,
    output_path: Path,
    y_scale: str,
    double_width: bool,
) -> None:
    """Save a combined comparison of all inference-run histories.

    Args:
        runs: Prediction names and their TensorBoard histories.
        style: Shared plotter style profile.
        output_path: Destination PDF path.
        y_scale: Vertical-axis scale, either ``"linear"`` or ``"log"``.
        double_width: Whether to double the base figure width.
    """
    fig, loss_axis, learning_rate_axis = _create_axes(style, "Combined training runs", y_scale, double_width)
    for index, (name, history) in enumerate(runs):
        _draw_history(
            loss_axis,
            learning_rate_axis,
            history,
            name,
            method_color(index),
            style,
            combined=True,
        )
    add_legend(loss_axis, style, ncol=2)
    add_legend(learning_rate_axis, style, ncol=2)
    save_figure(fig, output_path, style)
