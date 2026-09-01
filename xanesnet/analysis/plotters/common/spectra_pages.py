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

"""Single-sample spectra pages shared by the spectra plotters.

A page shows the target and predicted spectrum of one sample above their
residual, annotated with every scalar value known for that sample. When the
sample carries a matched structure, the structure is drawn alongside.
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

from xanesnet.serialization.prediction_readers import PredictionSample

from ...sample_data import merged_scalars
from ...utils import ScalarValue, sample_label
from .layout import compact_layout
from .structures import draw_structure
from .style import (
    COLOUR_ACCENT_RED,
    COLOUR_PREDICTION,
    COLOUR_TARGET,
    add_subtitle,
    annotate_box,
    method_colour,
    style_axis,
)

# Outer margins of the structure page. The mosaic layout is incompatible with
# tight_layout, so the margins are set explicitly.
_STRUCTURE_PAGE_ADJUST = dict(left=0.04, right=0.99, top=0.95, bottom=0.10, wspace=0.10, hspace=0.16)


def spectra_page_figure(
    sample: PredictionSample,
    col_scalars: dict[str, Any],
    subtitle: str,
) -> Figure:
    """Create a spectra comparison figure for one prediction sample.

    Args:
        sample: Prediction sample containing ``prediction`` and ``target`` spectra, and
            ``sample_id`` for the title. It may also contain ``prediction_std`` uncertainty.
            Spectra values are flattened to one-dimensional arrays with shape ``(N,)``.
        col_scalars: Collector record aligned with ``sample``.
        subtitle: Subtitle text describing prediction and selector context.

    Returns:
        Matplotlib figure with spectra and residual panels.
    """
    fig, (ax_spec, ax_res) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 5),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    box_text = _draw_sample_page(sample, col_scalars, ax_spec, ax_res)
    add_subtitle(fig, subtitle)
    compact_layout(fig)

    # The headroom must be reserved after the final layout pass, because the
    # measured box and legend fractions only match once the axes stop moving.
    _reserve_spectra_headroom(ax_spec, box_text)
    return fig


def spectra_structure_page_figure(
    sample: PredictionSample,
    col_scalars: dict[str, Any],
    subtitle: str,
) -> Figure:
    """Create a spectra comparison figure with the matched structure alongside.

    The structure is drawn in the left panel and the spectra and residual
    panels sit on the right, so the molecular context and the prediction
    quality are visible together.

    Args:
        sample: Prediction sample containing ``prediction``, ``target``, ``sample_id``,
            and a matched ``structure``.
        col_scalars: Collector record aligned with ``sample``.
        subtitle: Subtitle text describing prediction and selector context.

    Returns:
        Matplotlib figure with structure, spectra, and residual panels.
    """
    fig, axes = plt.subplot_mosaic(
        [["structure", "spectra"], ["structure", "residual"]],
        figsize=(12, 5),
        width_ratios=[1.0, 2.4],
        height_ratios=[3, 1],
        gridspec_kw={"wspace": _STRUCTURE_PAGE_ADJUST["wspace"], "hspace": _STRUCTURE_PAGE_ADJUST["hspace"]},
    )
    ax_struct = axes["structure"]
    ax_spec = axes["spectra"]
    ax_res = axes["residual"]
    ax_res.sharex(ax_spec)

    box_text = _draw_sample_page(sample, col_scalars, ax_spec, ax_res)
    add_subtitle(fig, subtitle)
    fig.subplots_adjust(**_STRUCTURE_PAGE_ADJUST)

    structure = sample.get("structure")
    if structure is not None:
        # The structure frame must match the rendered cell shape, which is only
        # known after the margins above have been applied.
        fig.canvas.draw()
        bbox = ax_struct.get_position()
        frame_ratio = (bbox.height * fig.get_figheight()) / (bbox.width * fig.get_figwidth())
        draw_structure(
            ax_struct,
            structure,
            str(sample["sample_id"]),
            sample.get("target_site_index"),
            frame_ratio=frame_ratio,
            legend_fontsize=8.0,
        )
        ax_struct.set_title("Structure", loc="left")
    else:
        ax_struct.axis("off")

    _reserve_spectra_headroom(ax_spec, box_text)
    return fig


def combined_spectra_page_figure(
    sample: PredictionSample,
    method_labels: list[str],
    predictions: list[np.ndarray],
    target: np.ndarray,
    subtitle: str,
) -> Figure:
    """Create a spectra figure overlaying several methods' predictions on one sample.

    Used to compare prediction readers on a sample they share, against one
    target curve. Mirrors the single-method pages: when the sample carries a
    matched structure it is drawn in a left panel, otherwise the spectra and
    residual panels fill the figure.

    Args:
        sample: Prediction sample providing ``sample_id`` and, optionally, a
            matched ``structure`` with a ``target_site_index``.
        method_labels: Display label per method, aligned with ``predictions``.
        predictions: One-dimensional predicted spectrum per method, aligned with
            ``method_labels``.
        target: One-dimensional target spectrum shared by every method.
        subtitle: Subtitle text describing the shared selector context.

    Returns:
        Matplotlib figure with spectra and residual panels for every method,
        plus the structure panel when available.
    """
    sample_id = str(sample["sample_id"])
    structure = sample.get("structure")

    if structure is not None:
        fig, axes = plt.subplot_mosaic(
            [["structure", "spectra"], ["structure", "residual"]],
            figsize=(12, 5),
            width_ratios=[1.0, 2.4],
            height_ratios=[3, 1],
            gridspec_kw={"wspace": _STRUCTURE_PAGE_ADJUST["wspace"], "hspace": _STRUCTURE_PAGE_ADJUST["hspace"]},
        )
        ax_struct = axes["structure"]
        ax_spec = axes["spectra"]
        ax_res = axes["residual"]
        ax_res.sharex(ax_spec)
    else:
        fig, (ax_spec, ax_res) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(10, 5),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        ax_struct = None

    _draw_combined_spectra_panels(
        ax_spec,
        ax_res,
        method_labels,
        predictions,
        target,
        sample_label(sample),
    )
    add_subtitle(fig, subtitle)

    if ax_struct is not None:
        fig.subplots_adjust(**_STRUCTURE_PAGE_ADJUST)
        # The structure frame must match the rendered cell shape, which is only
        # known after the margins above have been applied.
        fig.canvas.draw()
        bbox = ax_struct.get_position()
        frame_ratio = (bbox.height * fig.get_figheight()) / (bbox.width * fig.get_figwidth())
        draw_structure(
            ax_struct,
            structure,
            sample_id,
            sample.get("target_site_index"),
            frame_ratio=frame_ratio,
            legend_fontsize=8.0,
        )
        ax_struct.set_title("Structure", loc="left")
    else:
        compact_layout(fig)

    return fig


def _draw_combined_spectra_panels(
    ax_spec: Axes,
    ax_res: Axes,
    method_labels: list[str],
    predictions: list[np.ndarray],
    target: np.ndarray,
    sample_id: str,
) -> None:
    """Draw the shared target and every method's prediction into two panels.

    Args:
        ax_spec: Top panel axis for the spectra.
        ax_res: Bottom panel axis for the residuals.
        method_labels: Display label per method, aligned with ``predictions``.
        predictions: One-dimensional predicted spectrum per method.
        target: One-dimensional target spectrum shared by every method.
        sample_id: Identifier shown in the page title.
    """
    x = np.arange(len(target))
    ax_spec.plot(x, target, label="Target", linewidth=2.0, color=COLOUR_TARGET)
    for index, (label, pred) in enumerate(zip(method_labels, predictions)):
        colour = method_colour(index)
        ax_spec.plot(x, pred, label=label, linewidth=1.6, linestyle="--", color=colour)
        ax_res.plot(x, pred - target, linewidth=1.6, color=colour)

    ax_res.axhline(0, color="black", linewidth=1.0, linestyle=":")
    ax_res.set_xlabel("Energy")
    ax_spec.set_title(f"Sample: {sample_id}", loc="left")
    ax_spec.legend(fontsize=7, loc="upper right")
    style_axis(ax_spec)
    style_axis(ax_res)


def _draw_sample_page(
    sample: PredictionSample,
    col_scalars: dict[str, Any],
    ax_spec: Axes,
    ax_res: Axes,
) -> Text | None:
    """Draw the spectra, residual, title, and scalar box of one sample page.

    Args:
        sample: Prediction sample providing the spectra and scalar fields.
        col_scalars: Collector record aligned with ``sample``.
        ax_spec: Top panel axis for the spectra.
        ax_res: Bottom panel axis for the residual.

    Returns:
        The rendered scalar-box text, or ``None`` when the sample has no scalars.
    """
    pred = np.asarray(sample["prediction"]).ravel()
    target = np.asarray(sample["target"]).ravel()
    pred_std_value = sample.get("prediction_std")
    pred_std = np.asarray(pred_std_value).ravel() if pred_std_value is not None else None

    _draw_spectra_panels(ax_spec, ax_res, pred, target, pred_std)
    ax_spec.set_title(f"Sample: {sample_label(sample)}", loc="left")
    return _add_spectra_scalar_box(ax_spec, merged_scalars(sample, col_scalars, include_metadata=True))


def _draw_spectra_panels(
    ax_spec: Axes,
    ax_res: Axes,
    pred: np.ndarray,
    target: np.ndarray,
    pred_std: np.ndarray | None,
) -> None:
    """Draw target, prediction, and residual curves into the two panels.

    Args:
        ax_spec: Top panel axis for the spectra.
        ax_res: Bottom panel axis for the residual.
        pred: One-dimensional prediction spectrum.
        target: One-dimensional target spectrum.
        pred_std: Optional one-dimensional standard deviation per channel.
    """
    x = np.arange(len(pred))
    ax_spec.plot(x, target, label="Target", linewidth=2.0, color=COLOUR_TARGET)
    if pred_std is not None:
        if pred_std.shape == pred.shape:
            ax_spec.fill_between(
                x,
                pred - pred_std,
                pred + pred_std,
                label="Prediction +/- 1 std",
                color=COLOUR_PREDICTION,
                alpha=0.18,
                linewidth=0,
            )
        else:
            logging.warning(
                "Skipping prediction_std shading because shape %s does not match prediction shape %s.",
                pred_std.shape,
                pred.shape,
            )
    ax_spec.plot(x, pred, label="Prediction", linewidth=2.0, color=COLOUR_PREDICTION, linestyle="--")
    ax_spec.legend(fontsize=8, loc="upper right")
    ax_res.plot(x, pred - target, color=COLOUR_ACCENT_RED, linewidth=2.0)
    ax_res.axhline(0, color="black", linewidth=1.0, linestyle=":")
    ax_res.set_xlabel("Energy")
    style_axis(ax_spec)
    style_axis(ax_res)


def _add_spectra_scalar_box(ax_spec: Axes, scalars: dict[str, ScalarValue]) -> Text | None:
    """Annotate a spectra panel with all scalar values of one sample.

    The box sits in the reserved whitespace above the spectra, aligned to the
    top right. The spectra legend stays in the upper right of the axes
    directly beneath the box.

    Args:
        ax_spec: Spectra panel axis to annotate.
        scalars: Scalar values of the sample and its collector record.

    Returns:
        The rendered text object, or ``None`` when there is nothing to show.
    """
    if not scalars:
        return None
    text = "\n".join(f"{key}: {value:.4g}" for key, value in scalars.items())
    return annotate_box(ax_spec, text, 0.99, 0.98, "right", "top", fontsize=8, alpha=0.7)


def _reserve_spectra_headroom(ax_spec: Axes, box_text: Text | None) -> None:
    """Reserve whitespace above the spectra for the scalar box and legend.

    The rendered extents of the already-drawn box and legend are measured.
    The legend is re-anchored directly beneath the box and the upper y limit
    is raised until the highest curve point sits below both. The caller must
    have applied the final figure layout first, otherwise the measured
    fractions no longer match the axes.

    Args:
        ax_spec: Spectra panel axis whose upper limit should be extended.
        box_text: Rendered scalar-box text, or ``None`` when no box exists.
    """
    if box_text is None:
        return
    fig = ax_spec.figure
    if fig is None:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    axes_win = ax_spec.get_window_extent()
    if axes_win.height <= 0:
        return
    box_height = (axes_win.y1 - box_text.get_window_extent(renderer).y0) / axes_win.height
    box_bottom = 0.98 - box_height

    free_bottom = box_bottom
    legend = ax_spec.get_legend()
    if legend is not None:
        legend.set_loc("upper right")
        legend.set_bbox_to_anchor((0.99, box_bottom - 0.01), transform=ax_spec.transAxes)
        fig.canvas.draw()
        legend_height = legend.get_window_extent(renderer).height / axes_win.height
        free_bottom = box_bottom - 0.01 - legend_height

    lo, hi = ax_spec.get_ylim()
    if not hi > lo:
        return
    headroom = min(1.08 * (1.0 / max(free_bottom - 0.01, 0.05) - 1.0), 3.5)
    ax_spec.set_ylim(lo, hi + headroom * (hi - lo))
