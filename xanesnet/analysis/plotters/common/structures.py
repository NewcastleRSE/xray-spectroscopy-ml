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

"""Rendering of atomic structures into a constant-size drawing frame.

Structures are projected with ``ase`` and then centered and scaled so every
structure fills the same frame regardless of its physical extent. This keeps
atom sizes comparable between figures and between the cells of a grid. When
the projection is unavailable, a plain coordinate scatter is drawn instead.
"""

import logging
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from pymatgen.core import Molecule, Structure

from .formatting import format_decimal

# Absolute ase radii used before the frame fit; the target site is drawn
# larger so it stands out among its neighbors.
_TARGET_SITE_RADIUS: float = 0.75
_DEFAULT_SITE_RADIUS: float = 0.45
_RING_COLOR: str = "#111111"

# The drawing frame spans [-0.5, 0.5] and the structure fills this fraction
# of it, leaving room for the element legend.
_STRUCTURE_FRAME_HALF: float = 0.5
_STRUCTURE_FILL: float = 0.85

# Length of the scale bar as a fraction of the frame width, and its font size.
_SCALE_BAR_LENGTH: float = 0.25
_SCALE_BAR_FONTSIZE: float = 5.0

# Radius of the target-site ring relative to the atom it encircles.
_RING_SCALE: float = 1.6

# Failures of the optional ase projection path that are recoverable by falling
# back to the coordinate scatter. Anything else is a programming error and
# must surface.
_RENDER_ERRORS = (ValueError, TypeError, IndexError, KeyError, AttributeError)


def draw_structure(
    ax: Axes,
    structure: Molecule | Structure,
    sample_id: str,
    target_site_index: int | None = None,
    frame_ratio: float = 1.0,
    legend_fontsize: float = 5.0,
    show_scale: bool = True,
) -> None:
    """Draw one structure as a ball projection into a fixed-size frame.

    A compact legend lists the element colors (plus the target-site marker
    when present). When ``target_site_index`` is given, that atom is enlarged
    and circled with a bold ring. A scale bar at the bottom right shows the
    physical size of the frame in Angstrom.

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
        show_scale: Whether to draw the Angstrom scale bar at the bottom
            right of the frame.
    """
    plot_vars_cls = None
    adaptor_cls = None
    try:
        from ase.io.utils import PlottingVariables as plot_vars_cls
        from pymatgen.io.ase import AseAtomsAdaptor as adaptor_cls
    except ImportError:
        # The scatter fallback below makes the missing projection obvious in
        # the figure itself, so this stays silent to avoid one log line per
        # rendered sample.
        pass
    if plot_vars_cls is not None and adaptor_cls is not None:
        try:
            atoms = adaptor_cls().get_atoms(structure)
            radii, highlight = _site_style(len(atoms), target_site_index)
            writer = plot_vars_cls(atoms, rotation="10x,20y,0z", radii=radii, scale=1, show_unit_cell=0)
            positions = np.asarray(writer.positions[:, :2], dtype=float)
            colors = np.asarray(writer.colors, dtype=float)
            atom_radii = np.asarray(writer.d, dtype=float) / 2.0
            fit_radii = atom_radii.copy()
            if highlight:
                assert target_site_index is not None
                fit_radii[target_site_index] *= _RING_SCALE
            pos, scale = _fit_frame(positions, fit_radii, frame_ratio)
            radii_scaled = atom_radii * scale
            for (x, y), radius, color in zip(pos, radii_scaled, colors):
                ax.add_patch(
                    Circle(
                        (float(x), float(y)),
                        float(radius),
                        facecolor=tuple(float(v) for v in color),
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=2,
                    )
                )
            if highlight:
                assert target_site_index is not None
                _add_target_site_ring(ax, pos[target_site_index], float(radii_scaled[target_site_index] * _RING_SCALE))
            _frame_axes(ax, frame_ratio)
            _add_element_legend(ax, atoms, colors, highlight, fontsize=legend_fontsize)
            if show_scale:
                _add_scale_bar(ax, scale, frame_ratio)
            return
        except _RENDER_ERRORS:
            logging.warning(
                "Structure rendering failed for %s, drawing position scatter instead.",
                sample_id,
                exc_info=True,
            )
    _structure_scatter(ax, structure, target_site_index, frame_ratio)


def _frame_axes(ax: Axes, frame_ratio: float) -> None:
    """Set the constant drawing frame and its visible box on one axes.

    Args:
        ax: Matplotlib axis holding the structure.
        frame_ratio: Height/width ratio of the drawing frame.
    """
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


def _add_scale_bar(ax: Axes, scale: float, frame_ratio: float) -> None:
    """Draw a length annotation in Angstrom at the bottom right of a frame.

    Projected positions are in Angstrom and scaled into the fixed frame by
    ``scale``, so one frame unit corresponds to ``1 / scale`` Angstrom. The
    bar spans a fixed fraction of the frame width and is labelled with the
    corresponding physical length. Both sit in the bottom margin that the
    frame fit leaves empty, so they never cover an atom.

    Args:
        ax: Matplotlib axis holding the structure.
        scale: Frame units per Angstrom returned by :func:`_fit_frame`.
        frame_ratio: Height/width ratio of the drawing frame.
    """
    if not np.isfinite(scale) or scale <= 0.0:
        return
    label = f"{format_decimal(_SCALE_BAR_LENGTH / scale, 3)} Angstrom"
    half = _STRUCTURE_FRAME_HALF
    x1 = half - 0.05
    y = -half * frame_ratio + (1.0 - _STRUCTURE_FILL) * half * frame_ratio / 2.0
    ax.plot(
        [x1 - _SCALE_BAR_LENGTH, x1],
        [y, y],
        color="black",
        linewidth=1.2,
        clip_on=False,
        zorder=20,
    )
    ax.text(
        x1 - _SCALE_BAR_LENGTH - 0.02,
        y,
        label,
        ha="right",
        va="center",
        fontsize=_SCALE_BAR_FONTSIZE,
        zorder=20,
    )


def _add_element_legend(ax: Axes, atoms: Any, colors: np.ndarray, has_target: bool, fontsize: float = 5.0) -> None:
    """Add a compact legend explaining the element colors.

    Args:
        ax: Matplotlib axis holding the structure.
        atoms: ASE ``Atoms`` object whose elements should be listed.
        colors: Per-atom RGB color triples with shape ``(N, 3)``.
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
    for number, color in zip(atoms.get_atomic_numbers(), colors):
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
                markerfacecolor=tuple(float(v) for v in color),
                markeredgecolor="black",
                markersize=0.9 * fontsize,
                linestyle="",
            )
        )
        labels.append(symbol)
    if has_target:
        handles.append(_target_site_handle(markersize=1.2 * fontsize))
        labels.append(" target site")
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
    frame_ratio: float = 1.0,
) -> None:
    """Draw a fallback atomic-position scatter projection for one structure.

    When ``target_site_index`` is given, that site is marked with an open
    ring so it is clear which site the plotted spectrum belongs to.

    Args:
        ax: Matplotlib axis to draw on.
        structure: Pymatgen structure or molecule.
        target_site_index: Index of the target site within ``structure``, or
            ``None`` when unknown.
        frame_ratio: Height/width ratio of the fixed drawing frame.
    """
    species = [str(site.specie) for site in structure]
    positions = np.asarray([site.coords[:2] for site in structure], dtype=float)
    _, scale = _fit_frame(positions, np.full(len(positions), _DEFAULT_SITE_RADIUS), frame_ratio)
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    center = (lo + hi) / 2.0
    projected = (positions - center) * scale
    for spec in dict.fromkeys(species):
        selected = [index for index, sp in enumerate(species) if sp == spec]
        ax.scatter(projected[selected, 0], projected[selected, 1], s=140, label=spec)
    if target_site_index is not None and 0 <= target_site_index < len(structure):
        ax.scatter(
            [projected[target_site_index, 0]],
            [projected[target_site_index, 1]],
            s=420,
            facecolors="none",
            edgecolors=_RING_COLOR,
            linewidths=2.0,
            zorder=5,
        )
        _add_target_site_legend(ax)
    else:
        ax.legend(fontsize=8, framealpha=0.9)
    _frame_axes(ax, frame_ratio)


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
        edgecolor=_RING_COLOR,
        linewidth=2.0,
        zorder=10,
    )
    ax.add_patch(ring)


def _target_site_handle(markersize: float) -> Line2D:
    """Build the open-ring legend proxy for the target-site marker.

    Args:
        markersize: Marker size of the proxy.

    Returns:
        Legend handle matching the drawn target-site ring.
    """
    return Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="none",
        markeredgecolor=_RING_COLOR,
        markersize=markersize,
        linestyle="",
    )


def _add_target_site_legend(ax: Axes) -> None:
    """Add a legend entry explaining the target-site highlight marker.

    Args:
        ax: Matplotlib axis that contains the target-site marker.
    """
    handles, labels = ax.get_legend_handles_labels()
    handles.append(_target_site_handle(markersize=8))
    labels.append(" target site")
    ax.legend(handles=handles, labels=labels, fontsize=8, framealpha=0.9)
