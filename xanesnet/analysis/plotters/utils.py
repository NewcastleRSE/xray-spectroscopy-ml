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

"""Shared helpers for analysis plotters."""

import logging
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.text import Text
from pymatgen.core import Molecule, Structure

from xanesnet.analysis.utils import ScalarValue, is_scalar_value
from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.serialization.prediction_readers import PredictionSample

from ..selectors import Selector

METHOD_COLOURS: list[str] = [
    "#005186",
    "#019cd8",
    "#e85651",
    "#d5a21e",
    "#7a5195",
    "#3f9e6e",
    "#c0504d",
    "#4f81bd",
]

_COMPUTABLE_METRICS: tuple[str, ...] = ("mse", "mae", "max_abs_error")

_TARGET_SITE_RADIUS: float = 0.75
_DEFAULT_SITE_RADIUS: float = 0.45
_RING_COLOUR: str = "#111111"

_STRUCTURE_FRAME_HALF: float = 0.5
_STRUCTURE_FILL: float = 0.85
_RING_SCALE: float = 1.6

_X_LABELPAD: float = 7.0
_Y_LABELPAD: float = 2.5


def collect_scalar_values(selector: Selector, stream: JSONLStream | None) -> dict[str, list[ScalarValue]]:
    """Collect scalar values from selected samples and optional collector outputs.

    Args:
        selector: Selector over prediction samples for one prediction reader and selector pair.
        stream: Optional collector result stream aligned with ``selector``.

    Returns:
        Mapping from scalar value key to values observed for that key.
    """
    values: dict[str, list[ScalarValue]] = {}
    if stream is not None:
        for sel_sample, col_sample in zip(selector, stream):
            for key, val in sel_sample.items():
                if key != "sample_id" and is_scalar_value(val):
                    values.setdefault(key, []).append(cast(float, val))
            for key, val in col_sample.items():
                if key != "sample_id" and is_scalar_value(val):
                    values.setdefault(key, []).append(cast(float, val))
    else:
        for sel_sample in selector:
            for key, val in sel_sample.items():
                if key != "sample_id" and is_scalar_value(val):
                    values.setdefault(key, []).append(cast(float, val))
    return values


def method_label_lines(reader_name: str, sel_label_str: str, sel_cfg: dict[str, Any]) -> list[str]:
    """Build the stacked label lines for one method group.

    The first line carries the prediction reader display name and the second
    carries the selector description, so long names can be stacked instead of
    sharing one line.

    Args:
        reader_name: Display name for the prediction reader.
        sel_label_str: Selector label derived from configuration.
        sel_cfg: Selector configuration dictionary for this selector index.

    Returns:
        Two label lines: the reader name and the selector description.
    """
    selector_parts = [f"sel={sel_label_str}"]
    extras = {k: v for k, v in sel_cfg.items() if k != "selector_type"}
    if extras:
        selector_parts.append(" ".join(f"{k}={v}" for k, v in extras.items()))
    return [reader_name, " ".join(selector_parts)]


def method_colour(index: int) -> str:
    """Return the palette colour for a method index (cycling).

    Args:
        index: Zero-based method index in first-seen order.

    Returns:
        Matplotlib-compatible colour string.
    """
    return METHOD_COLOURS[index % len(METHOD_COLOURS)]


def spectrum_metric(sample: PredictionSample, key: str) -> float:
    """Compute one per-sample error metric from prediction and target spectra.

    Args:
        sample: Prediction sample containing ``prediction`` and ``target`` arrays.
        key: Metric name, one of ``mse``, ``mae``, or ``max_abs_error``.

    Returns:
        Computed metric value.
    """
    pred = np.asarray(sample["prediction"]).ravel()
    target = np.asarray(sample["target"]).ravel()
    diff = pred - target
    if key == "mse":
        return float(np.mean(diff**2))
    if key == "mae":
        return float(np.mean(np.abs(diff)))
    if key == "max_abs_error":
        return float(np.max(np.abs(diff)))
    raise ValueError(f"Unknown spectrum metric: {key}")


def spectrum_error_value(
    sample: PredictionSample,
    col_scalars: dict[str, Any],
    sort_key: str | None,
) -> float:
    """Return the scalar value used to rank one spectra sample.

    Collector values take precedence over sample scalars; the configured key
    is tried first and the computable error metrics second. When nothing is
    available, the MSE between the predicted and target spectra is computed.

    Args:
        sample: Prediction sample containing spectra arrays and optional scalars.
        col_scalars: Collector scalar values aligned with ``sample``.
        sort_key: Preferred scalar key; ``None`` falls back to computed MSE.

    Returns:
        Ranking value for the sample.
    """
    keys = [sort_key] if sort_key is not None else []
    keys.extend(_COMPUTABLE_METRICS)
    for key in keys:
        value = col_scalars.get(key, sample.get(key))
        if is_scalar_value(value):
            return cast(float, value)
    return spectrum_metric(sample, "mse")


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
    ax_spec.plot(x, target, label="Target", linewidth=2.0, color="#019cd8")
    if pred_std is not None:
        if pred_std.shape == pred.shape:
            ax_spec.fill_between(
                x,
                pred - pred_std,
                pred + pred_std,
                label="Prediction +/- 1 std",
                color="#005186",
                alpha=0.18,
                linewidth=0,
            )
        else:
            logging.warning(
                "Skipping prediction_std shading because shape %s does not match prediction shape %s.",
                pred_std.shape,
                pred.shape,
            )
    ax_spec.plot(x, pred, label="Prediction", linewidth=2.0, color="#005186", linestyle="--")
    ax_spec.legend(fontsize=8, loc="upper right")
    ax_res.plot(x, pred - target, color="#e85651", linewidth=2.0)
    ax_res.axhline(0, color="black", linewidth=1.0, linestyle=":")
    ax_res.set_xlabel("Energy")
    style_axis(ax_spec)
    style_axis(ax_res)


def _spectra_scalars(sample: PredictionSample, col_scalars: dict[str, Any]) -> dict[str, ScalarValue]:
    """Gather all scalar values of one sample.

    Sample fields and collector values are merged, excluding the spectra and
    the sample identifier itself.

    Args:
        sample: Prediction sample providing scalar fields.
        col_scalars: Collector scalar values aligned with ``sample``.

    Returns:
        Mapping from scalar key to value.
    """
    scalars: dict[str, ScalarValue] = {}
    for key, value in sample.items():
        if key not in ("prediction", "target", "sample_id") and is_scalar_value(value):
            scalars[key] = cast(ScalarValue, value)
    for key, value in col_scalars.items():
        if key != "sample_id" and is_scalar_value(value):
            scalars[key] = cast(ScalarValue, value)
    return scalars


def _reserve_spectra_headroom(ax_spec: Axes, box_text: Text | None) -> None:
    """Reserve whitespace above the spectra for the scalar box and legend.

    The rendered extents of the already-drawn box and legend are measured.
    The legend is re-anchored directly beneath the box and the upper y limit
    is raised until the highest curve point sits below both.

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


def _add_spectra_scalar_box(ax_spec: Axes, scalars: dict[str, ScalarValue]) -> Text | None:
    """Annotate a spectra panel with all scalar values of one sample.

    The box sits in the reserved whitespace above the spectra, aligned to the
    top right, with a semi-transparent white background. The spectra legend
    stays in the upper right of the axes directly beneath the box.

    Args:
        ax_spec: Spectra panel axis to annotate.
        scalars: Scalar values gathered by :func:`_spectra_scalars`.

    Returns:
        The rendered text object, or ``None`` when there is nothing to show.
    """
    if not scalars:
        return None

    text = "\n".join(f"{k}: {v:.4g}" for k, v in scalars.items())
    return ax_spec.text(
        0.99,
        0.98,
        text,
        transform=ax_spec.transAxes,
        fontsize=8,
        horizontalalignment="right",
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="#bbbbbb"),
    )


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
        col_scalars: Collector scalar values aligned with ``sample``.
        subtitle: Subtitle text describing prediction and selector context.

    Returns:
        Matplotlib figure with spectra and residual panels.
    """
    pred = np.asarray(sample["prediction"]).ravel()
    target = np.asarray(sample["target"]).ravel()
    pred_std_value = sample.get("prediction_std")
    pred_std = np.asarray(pred_std_value).ravel() if pred_std_value is not None else None

    fig, (ax_spec, ax_res) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 5),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    _draw_spectra_panels(ax_spec, ax_res, pred, target, pred_std)
    ax_spec.set_title(f"Sample: {sample['sample_id']}", loc="left")
    box_text = _add_spectra_scalar_box(ax_spec, _spectra_scalars(sample, col_scalars))

    fig.text(0.5, -0.01, subtitle, ha="center", va="top", fontsize=7, color="gray")
    compact_layout(fig)
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
        col_scalars: Collector scalar values aligned with ``sample``.
        subtitle: Subtitle text describing prediction and selector context.

    Returns:
        Matplotlib figure with structure, spectra, and residual panels.
    """
    pred = np.asarray(sample["prediction"]).ravel()
    target = np.asarray(sample["target"]).ravel()
    pred_std_value = sample.get("prediction_std")
    pred_std = np.asarray(pred_std_value).ravel() if pred_std_value is not None else None
    sample_id = sample["sample_id"]
    structure = sample.get("structure")

    fig, axes = plt.subplot_mosaic(
        [["structure", "spectra"], ["structure", "residual"]],
        figsize=(12, 5),
        width_ratios=[1.0, 2.4],
        height_ratios=[3, 1],
        gridspec_kw={"wspace": 0.10, "hspace": 0.16},
    )
    ax_struct = axes["structure"]
    ax_spec = axes["spectra"]
    ax_res = axes["residual"]
    ax_res.sharex(ax_spec)

    _draw_spectra_panels(ax_spec, ax_res, pred, target, pred_std)
    ax_spec.set_title(f"Sample: {sample_id}", loc="left")
    box_text = _add_spectra_scalar_box(ax_spec, _spectra_scalars(sample, col_scalars))

    fig.text(0.5, -0.01, subtitle, ha="center", va="top", fontsize=7, color="gray")
    fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.10, wspace=0.10, hspace=0.16)

    if structure is not None:
        fig.canvas.draw()
        bbox = ax_struct.get_position()
        frame_ratio = (bbox.height * fig.get_figheight()) / (bbox.width * fig.get_figwidth())
        _draw_structure(
            ax_struct,
            structure,
            str(sample_id),
            sample.get("target_site_index"),
            frame_ratio=frame_ratio,
            legend_fontsize=8.0,
        )
        ax_struct.set_title("Structure", loc="left")
    else:
        ax_struct.axis("off")

    _reserve_spectra_headroom(ax_spec, box_text)
    return fig


def _draw_structure(
    ax: Axes,
    structure: Molecule | Structure,
    sample_id: str,
    target_site_index: int | None = None,
    frame_ratio: float = 1.0,
    legend_fontsize: float = 5.0,
) -> None:
    """Draw one structure as a ball projection into a fixed-size frame.

    The structure is projected with ``ase`` and then normalized so every
    structure occupies the same constant drawing frame regardless of its
    physical extent. A compact legend lists the element colours (plus the
    target-site marker when present). When ``target_site_index`` is given,
    that atom is enlarged and circled with a bold ring.

    Args:
        ax: Matplotlib axis to draw on.
        structure: Pymatgen structure or molecule.
        sample_id: Identifier used in the fallback warning.
        target_site_index: Index of the target site within ``structure``, or
            ``None`` when unknown.
        frame_ratio: Height/width ratio of the drawing frame. ``1.0`` draws a
            square frame (and enforces equal aspect); other values fill a
            rectangular frame with that ratio while keeping the structure
            undistorted.
        legend_fontsize: Font size of the compact element legend.
    """
    plot_vars_cls = None
    adaptor_cls = None
    try:
        from ase.io.utils import PlottingVariables as plot_vars_cls
        from pymatgen.io.ase import AseAtomsAdaptor as adaptor_cls
    except ImportError:
        pass
    if plot_vars_cls is not None and adaptor_cls is not None:
        try:
            atoms = adaptor_cls().get_atoms(structure)
            radii, highlight = _site_style(len(atoms), target_site_index)
            writer = plot_vars_cls(atoms, rotation="10x,20y,0z", radii=radii, scale=1, show_unit_cell=0)
            positions = np.asarray(writer.positions[:, :2], dtype=float)
            colours = np.asarray(writer.colors, dtype=float)
            atom_radii = np.asarray(writer.d, dtype=float) / 2.0
            fit_radii = atom_radii.copy()
            if highlight:
                assert target_site_index is not None
                fit_radii[target_site_index] *= _RING_SCALE
            pos, scale = _fit_frame(positions, fit_radii, frame_ratio)
            radii_scaled = atom_radii * scale
            for (x, y), radius, colour in zip(pos, radii_scaled, colours):
                ax.add_patch(
                    Circle(
                        (float(x), float(y)),
                        float(radius),
                        facecolor=tuple(float(v) for v in colour),
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=2,
                    )
                )
            if highlight:
                assert target_site_index is not None
                _add_target_site_ring(ax, pos[target_site_index], float(radii_scaled[target_site_index] * _RING_SCALE))
            half = _STRUCTURE_FRAME_HALF
            ax.set_xlim(-half, half)
            ax.set_ylim(-half * frame_ratio, half * frame_ratio)
            if frame_ratio == 1.0:
                ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color("black")
            _add_element_legend(ax, atoms, colours, highlight, fontsize=legend_fontsize)
            return
        except Exception:
            logging.warning(
                "Structure rendering failed for %s, drawing position scatter instead.",
                sample_id,
            )
    _structure_scatter(ax, structure, target_site_index)


def _site_style(n_atoms: int, target_site_index: int | None) -> tuple[list[float], bool]:
    """Build per-atom radii and highlight flags for one structure.

    Args:
        n_atoms: Number of atoms in the structure.
        target_site_index: Index of the target site, or ``None`` when unknown.

    Returns:
        ``(radii, highlight)`` where ``highlight`` is ``True`` when the
        target site index is valid and should be marked.
    """
    radii = [_DEFAULT_SITE_RADIUS] * n_atoms
    highlight = target_site_index is not None and 0 <= target_site_index < n_atoms
    if highlight:
        assert target_site_index is not None
        radii[target_site_index] = _TARGET_SITE_RADIUS
    return radii, highlight


def _fit_frame(positions: np.ndarray, radii: np.ndarray, frame_ratio: float = 1.0) -> tuple[np.ndarray, float]:
    """Center and scale projected positions into a constant drawing frame.

    Args:
        positions: Projected atom positions with shape ``(N, 2)``.
        radii: Per-atom radii, including any ring allowance, with shape ``(N,)``.
        frame_ratio: Height/width ratio of the frame to fill.

    Returns:
        ``(scaled_positions, scale)`` where the union of atom extents fills a
        fixed fraction of the frame, preserving the aspect ratio.
    """
    lo = (positions - radii[:, None]).min(axis=0)
    hi = (positions + radii[:, None]).max(axis=0)
    center = (lo + hi) / 2.0
    span = hi - lo
    frame_w = 2.0 * _STRUCTURE_FRAME_HALF * _STRUCTURE_FILL
    frame_h = frame_w * frame_ratio
    scale_w = frame_w / span[0] if span[0] > 0 else np.inf
    scale_h = frame_h / span[1] if span[1] > 0 else np.inf
    scale = float(min(scale_w, scale_h))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return (positions - center) * scale, scale


def _add_element_legend(ax: Axes, atoms: Any, colours: np.ndarray, has_target: bool, fontsize: float = 5.0) -> None:
    """Add a compact legend explaining the element colours.

    Args:
        ax: Matplotlib axis holding the structure.
        atoms: ASE ``Atoms`` object whose elements should be listed.
        colours: Per-atom RGB colour triples with shape ``(N, 3)``.
        has_target: Whether to append a target-site marker entry.
        fontsize: Legend font size.
    """
    try:
        from ase.data import chemical_symbols
    except ImportError:
        return
    handles: list[Any] = []
    labels: list[str] = []
    seen: set[str] = set()
    for number, colour in zip(atoms.get_atomic_numbers(), colours):
        symbol = chemical_symbols[int(number)]
        if symbol in seen:
            continue
        seen.add(symbol)
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=tuple(float(v) for v in colour),
                markeredgecolor="black",
                markersize=0.9 * fontsize,
                linestyle="",
            )
        )
        labels.append(symbol)
    if has_target:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor=_RING_COLOUR,
                markersize=1.2 * fontsize,
                linestyle="",
            )
        )
        labels.append("target site")
    if handles:
        ax.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            fontsize=fontsize,
            framealpha=0.8,
            borderpad=0.3,
            labelspacing=0.35,
            handlelength=0.7,
            handletextpad=0.35,
            borderaxespad=0.35,
        )


def _structure_scatter(
    ax: Axes,
    structure: Molecule | Structure,
    target_site_index: int | None = None,
) -> None:
    """Draw a fallback atomic-position scatter projection for one structure.

    When ``target_site_index`` is given, that site is marked with an open
    ring so it is clear which site the plotted spectrum belongs to.

    Args:
        ax: Matplotlib axis to draw on.
        structure: Pymatgen structure or molecule.
        target_site_index: Index of the target site within ``structure``, or
            ``None`` when unknown.
    """
    species = [str(site.specie) for site in structure]
    for spec in dict.fromkeys(species):
        xs = [site.coords[0] for site, sp in zip(structure, species) if sp == spec]
        ys = [site.coords[1] for site, sp in zip(structure, species) if sp == spec]
        ax.scatter(xs, ys, s=140, label=spec)
    if target_site_index is not None and 0 <= target_site_index < len(structure):
        assert target_site_index is not None
        site = structure[target_site_index]
        coords = getattr(site, "coords")
        ax.scatter(
            [coords[0]],
            [coords[1]],
            s=420,
            facecolors="none",
            edgecolors=_RING_COLOUR,
            linewidths=2.0,
            zorder=5,
        )
        _add_target_site_legend(ax)
    else:
        ax.legend(fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")


def _add_target_site_ring(ax: Axes, xy: np.ndarray, ring_radius: float) -> None:
    """Draw a bold ring around the rendered target-site atom.

    Args:
        ax: Matplotlib axis to draw on.
        xy: Projected image-plane position of the target site.
        ring_radius: Radius of the ring in frame units.
    """
    ring = Circle(
        (float(xy[0]), float(xy[1])),
        radius=ring_radius,
        fill=False,
        edgecolor=_RING_COLOUR,
        linewidth=2.0,
        zorder=10,
    )
    ax.add_patch(ring)


def _add_target_site_legend(ax: Axes) -> None:
    """Add a legend entry explaining the target-site highlight marker.

    Args:
        ax: Matplotlib axis that contains the target-site marker.
    """
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor=_RING_COLOUR,
            markersize=8,
        )
    )
    labels.append("target site")
    ax.legend(handles=handles, labels=labels, fontsize=8, framealpha=0.9)


def add_subtitle(fig: Figure, text: str) -> None:
    """Add centered subtitle text below a figure.

    Args:
        fig: Matplotlib figure to annotate.
        text: Subtitle text.
    """
    fig.text(0.5, -0.01, text, ha="center", va="top", fontsize=7, color="gray")


def compact_layout(
    fig: Figure,
    rect: tuple[float, float, float, float] | None = None,
    pad: float = 0.3,
) -> None:
    """Apply a compact tight layout to one figure.

    Uses smaller padding than the default ``tight_layout`` while still
    keeping labels, ticks, and titles from overlapping.

    Args:
        fig: Matplotlib figure to lay out.
        rect: Optional (left, bottom, right, top) figure region reserved for
            the axes, as fractions of the figure. Figures with a supertitle
            should reserve the top of the figure.
        pad: Padding between the figure edge and the axes, as a fraction of
            the font size.
    """
    fig.tight_layout(pad=pad, rect=rect)


def style_axis(ax: Axes, xlabelpad: float = _X_LABELPAD, ylabelpad: float = _Y_LABELPAD) -> None:
    """Apply uniform axis-label spacing to one axes.

    The y label usually sits further from its tick values than the x label,
    so this sets a larger gap for the x label and a smaller gap for the y
    label.

    Args:
        ax: Matplotlib axis to style.
        xlabelpad: Gap between the x axis label and its tick labels.
        ylabelpad: Gap between the y axis label and its tick labels.
    """
    ax.xaxis.labelpad = xlabelpad
    ax.yaxis.labelpad = ylabelpad
