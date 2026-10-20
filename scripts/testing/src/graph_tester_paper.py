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

"""Create publication-ready structure and graph-edge figures for a PMGJSON sample."""

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any, TypeAlias

import ase.data.colors as ase_colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pymatgen.core import Element, Molecule, Structure
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError, Voronoi

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xanesnet.datasources.pmgjson import PMGJSONSource
from xanesnet.graphs import GraphBuilderRegistry
from xanesnet.graphs.voronoi import _build_periodic_supercell

PMGObject: TypeAlias = Structure | Molecule
RGBColor: TypeAlias = tuple[float, float, float]
EdgeKey: TypeAlias = tuple[Any, ...]
VoronoiCell: TypeAlias = tuple[int, np.ndarray]
VoronoiFace: TypeAlias = tuple[np.ndarray, int, int]

EDGE_COLOR = (0.12, 0.42, 0.92, 0.95)
EDGE_FAR_COLOR = (0.005, 0.025, 0.16, 0.95)
PBC_EDGE_COLOR = (0.95, 0.30, 0.04, 0.96)
PBC_EDGE_FAR_COLOR = (0.28, 0.018, 0.004, 0.96)
EDGE_DEPTH_CONTRAST = 1.35
TARGET_COLOR = (0.82, 0.08, 0.12, 1.0)
CELL_COLOR = (0.18, 0.18, 0.18, 0.55)
VORONOI_CELL_FACE_COLOR = (0.16, 0.40, 0.86, 0.09)
VORONOI_CELL_EDGE_COLOR = (0.08, 0.22, 0.52, 0.48)
VORONOI_TARGET_FACE_COLOR = (0.82, 0.08, 0.12, 0.18)
VORONOI_TARGET_EDGE_COLOR = (0.72, 0.03, 0.08, 0.88)

EDGE_LINE_WIDTH = 1.5
PBC_EDGE_LINE_WIDTH = 1.5
CELL_LINE_WIDTH = 1.5
VORONOI_CELL_LINE_WIDTH = 0.65
VORONOI_TARGET_CELL_LINE_WIDTH = 1.15
EDGE_ATOM_SIZE_SCALE = 100.0
EDGE_IMAGE_MARKER_SIZE = 40.0
EDGE_TARGET_RING_MIN_SIZE = 330.0
EDGE_FRAME_PADDING = 1.0
DEPTH_ZORDER_BASE = 100.0
DEPTH_ZORDER_SCALE = 10.0
DEPTH_OVERLAY_ZORDER = 1000.0

STRUCTURE_DEFAULT_RADIUS = 0.35
STRUCTURE_TARGET_RADIUS = 0.64
STRUCTURE_RING_SCALE = 1.6
STRUCTURE_RING_COLOR = "#111111"
STRUCTURE_FRAME_HALF = 0.5
STRUCTURE_FILL = 0.80
STRUCTURE_MAX_FILL = 0.96
STRUCTURE_SCALE_BAR_LENGTH = 0.25

PUBLICATION_SERIF_FONTS = ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"]
LATIN_MODERN_FONT_DIRECTORIES = (
    Path("/usr/share/texmf/fonts/opentype/public/lm"),
    Path("/usr/share/texlive/texmf-dist/fonts/opentype/public/lm"),
)
PUBLICATION_FONT_SIZE = 18
PUBLICATION_TITLE_SIZE = 11
PUBLICATION_LABEL_SIZE = 18
PUBLICATION_TICK_SIZE = 18
PUBLICATION_LEGEND_SIZE = 18
PUBLICATION_ANNOTATION_SIZE = 18
PUBLICATION_LEGEND_MARKER_SIZE = 18
PUBLICATION_DPI = 600

LEGEND_MODES = ("per-figure", "standalone", "both", "none")
VORONOI_CELL_SCOPES = ("target", "first-shell", "all")
# Change this single value to scale axes, structure frames, graph edges, cell
# outlines, atom outlines, and other publication line weights together.
LINE_WIDTH_SCALE = 2.5


def _line_width(base_width: float) -> float:
    """Scale a publication line width by the global figure setting."""

    return float(base_width) * LINE_WIDTH_SCALE


def load_sample(json_dir: Path, index: int | None, file_stem: str | None) -> PMGObject:
    """Load one PMGJSON object by sorted index or file stem.

    Args:
        json_dir: Directory containing one PMGJSON file per sample.
        index: Zero-based sample index used when ``file_stem`` is not supplied.
        file_stem: Sample file stem without the ``.json`` suffix.

    Returns:
        Loaded pymatgen ``Structure`` or ``Molecule``.

    Raises:
        SystemExit: If the requested sample is not available.
    """

    ds = PMGJSONSource(datasource_type="pmgjson", json_path=str(json_dir), spectrum_key="spectrum")
    if file_stem is not None:
        if file_stem not in ds.sample_ids:
            raise SystemExit(f"file stem {file_stem!r} not found in {json_dir}")
        target_idx = ds.sample_ids.index(file_stem)
    else:
        target_idx = 0 if index is None else int(index)
        if not (0 <= target_idx < len(ds)):
            raise SystemExit(f"--index {target_idx} out of range [0, {len(ds)})")
    for i, obj in enumerate(ds):
        if i == target_idx:
            return obj
    raise SystemExit("Failed to iterate datasource")


def element_symbol(atomic_number: int) -> str:
    """Return an element symbol, falling back to the atomic number if needed."""

    try:
        return Element.from_Z(int(atomic_number)).symbol
    except Exception:
        return f"Z{atomic_number}"


def element_colors(atomic_numbers: np.ndarray) -> list[RGBColor]:
    """Return ASE Jmol element colors in Matplotlib's normalized RGB format."""

    colors: list[RGBColor] = []
    for z in atomic_numbers.tolist():
        try:
            rgb = np.asarray(getattr(ase_colors, "jmol_colors")[int(z)], dtype=np.float64)
            if rgb.shape != (3,):
                raise ValueError("unexpected Jmol color shape")
            colors.append((float(rgb[0]), float(rgb[1]), float(rgb[2])))
        except Exception:
            colors.append((0.5, 0.5, 0.5))
    return colors


def _project_structure(
    pmg_obj: PMGObject,
    atomic_numbers: np.ndarray,
    target_site_idx: int,
    zoom: float = 1.0,
    mark_target_site: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Project a structure into the fixed 2D ball-rendering frame used here.

    The projection is generated through ASE's plotting data, while the frame
    fitting and drawing remain local to this tool.  This keeps the structure
    panel independent of the analysis plotter implementation while retaining
    its compact, axis-free presentation.
    """

    n_atoms = len(pmg_obj)
    requested_radii = np.full(n_atoms, STRUCTURE_DEFAULT_RADIUS, dtype=np.float64)
    if mark_target_site:
        requested_radii[target_site_idx] = STRUCTURE_TARGET_RADIUS
    try:
        from ase.io.utils import PlottingVariables
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = AseAtomsAdaptor().get_atoms(pmg_obj)
        plot_vars = PlottingVariables(
            atoms,
            rotation="10x,20y,0z",
            radii=requested_radii.tolist(),
            scale=1,
            show_unit_cell=0,
        )
        positions = np.asarray(plot_vars.positions, dtype=np.float64)[:, :2]
        colors = np.asarray(plot_vars.colors, dtype=np.float64)
        rendered_radii = np.asarray(plot_vars.d, dtype=np.float64) / 2.0
        if positions.shape != (n_atoms, 2) or colors.shape != (n_atoms, 3) or rendered_radii.shape != (n_atoms,):
            raise ValueError("ASE returned unexpected structure-rendering shapes")
    except (ImportError, ValueError, TypeError, IndexError, KeyError, AttributeError):
        positions = np.asarray(pmg_obj.cart_coords, dtype=np.float64)[:, :2]
        colors = np.asarray(element_colors(atomic_numbers), dtype=np.float64)
        rendered_radii = requested_radii

    fit_radii = rendered_radii.copy()
    if mark_target_site:
        fit_radii[target_site_idx] *= STRUCTURE_RING_SCALE
    lower = (positions - fit_radii[:, None]).min(axis=0)
    upper = (positions + fit_radii[:, None]).max(axis=0)
    center = (lower + upper) / 2.0
    span = upper - lower
    zoom_factor = float(zoom) if np.isfinite(zoom) and zoom > 0.0 else 1.0
    frame_fill = min(STRUCTURE_MAX_FILL, max(0.50, STRUCTURE_FILL * zoom_factor))
    frame_width = 2.0 * STRUCTURE_FRAME_HALF * frame_fill
    scale_x = frame_width / span[0] if span[0] > 0.0 else np.inf
    scale_y = frame_width / span[1] if span[1] > 0.0 else np.inf
    scale = float(min(scale_x, scale_y))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    projected = (positions - center) * scale
    return projected, colors, rendered_radii * scale, scale


def _configure_structure_axes(ax: Any) -> None:
    """Configure the square, axis-free frame used for structure-only figures."""

    ax.set_xlim(-STRUCTURE_FRAME_HALF, STRUCTURE_FRAME_HALF)
    ax.set_ylim(-STRUCTURE_FRAME_HALF, STRUCTURE_FRAME_HALF)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(_line_width(0.8))
        spine.set_color("black")


def _add_structure_scale_bar(ax: Any, scale: float) -> None:
    """Add a compact angstrom scale bar inside the structure frame."""

    if not np.isfinite(scale) or scale <= 0.0:
        return
    x_right = STRUCTURE_FRAME_HALF - 0.05
    x_left = x_right - STRUCTURE_SCALE_BAR_LENGTH
    y = -STRUCTURE_FRAME_HALF + (1.0 - STRUCTURE_FILL) * STRUCTURE_FRAME_HALF / 2.0
    ax.plot(
        [x_left, x_right],
        [y, y],
        color="black",
        linewidth=_line_width(1.2),
        clip_on=False,
        zorder=20,
    )
    ax.text(
        x_left - 0.02,
        y,
        f"{STRUCTURE_SCALE_BAR_LENGTH / scale:.3g} Å",
        ha="right",
        va="center",
        fontsize=PUBLICATION_ANNOTATION_SIZE,
        zorder=20,
    )


def _draw_structure_projection(
    ax: Any,
    pmg_obj: PMGObject,
    atomic_numbers: np.ndarray,
    target_site_idx: int,
    label_atoms: bool,
    show_legend: bool,
    zoom: float,
    mark_target_site: bool = True,
) -> None:
    """Draw colored atoms and optionally the target ring in the 2D projection."""

    projected, colors, radii, scale = _project_structure(
        pmg_obj,
        atomic_numbers,
        target_site_idx,
        zoom,
        mark_target_site,
    )
    for (x, y), radius, color in zip(projected, radii, colors):
        ax.add_patch(
            Circle(
                (float(x), float(y)),
                float(radius),
                facecolor=tuple(float(value) for value in color),
                edgecolor="black",
                linewidth=_line_width(0.5),
                zorder=2,
            )
        )

    if mark_target_site:
        ax.add_patch(
            Circle(
                (float(projected[target_site_idx, 0]), float(projected[target_site_idx, 1])),
                float(radii[target_site_idx] * STRUCTURE_RING_SCALE),
                facecolor="none",
                edgecolor=STRUCTURE_RING_COLOR,
                linewidth=_line_width(2.0),
                zorder=10,
            )
        )
    if label_atoms:
        for i, (x, y) in enumerate(projected):
            ax.text(
                float(x),
                float(y),
                f" {element_symbol(int(atomic_numbers[i]))}{i}",
                fontsize=PUBLICATION_ANNOTATION_SIZE,
                color="black",
                zorder=20,
            )

    _configure_structure_axes(ax)
    if show_legend:
        handles = element_legend_handles(atomic_numbers)
        if mark_target_site:
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=PUBLICATION_LEGEND_MARKER_SIZE,
                    markerfacecolor="none",
                    markeredgecolor=STRUCTURE_RING_COLOR,
                    markeredgewidth=_line_width(1.8),
                    label=f"target site ({target_site_idx})",
                )
            )
        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=PUBLICATION_LEGEND_SIZE,
            framealpha=0.8,
            borderpad=0.3,
            labelspacing=0.35,
            handlelength=0.7,
            handletextpad=0.35,
            borderaxespad=0.35,
        )
    _add_structure_scale_bar(ax, scale)


def _cell_corner_coords(structure: Structure) -> np.ndarray:
    """Return Cartesian coordinates of the eight unit-cell corners in angstroms."""

    lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
    corners_frac = np.asarray(list(itertools.product([0.0, 1.0], repeat=3)), dtype=np.float64)
    return corners_frac @ lattice


def _voronoi_points(pmg_obj: PMGObject, cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the point set used to render central Voronoi cells."""

    effective_cutoff = float(cutoff) if np.isfinite(cutoff) and cutoff > 0.0 else 1.0
    if isinstance(pmg_obj, Structure):
        return _build_periodic_supercell(pmg_obj, effective_cutoff)
    coords = np.asarray(pmg_obj.cart_coords, dtype=np.float64)
    n_atoms = coords.shape[0]
    return coords, np.arange(n_atoms, dtype=np.int64), np.ones(n_atoms, dtype=bool)


def _molecule_voronoi_bounds(coords: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a generous display box for otherwise unbounded molecular cells."""

    if coords.size == 0:
        return np.full(3, -1.0, dtype=np.float64), np.full(3, 1.0, dtype=np.float64)
    span = coords.max(axis=0) - coords.min(axis=0)
    max_span = float(np.max(span)) if span.size > 0 else 1.0
    cutoff_span = float(cutoff) if np.isfinite(cutoff) and cutoff > 0.0 else 1.0
    margin = max(1.0, 0.5 * max(max_span, cutoff_span))
    return coords.min(axis=0) - margin, coords.max(axis=0) + margin


def _clipped_molecule_cell(
    points: np.ndarray,
    point_index: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray | None:
    """Return one molecular Voronoi cell clipped by a distant display box."""

    point = np.asarray(points[point_index], dtype=np.float64)
    halfspaces: list[np.ndarray] = []
    for other_index, other in enumerate(points):
        if other_index == point_index:
            continue
        displacement = np.asarray(other, dtype=np.float64) - point
        if float(np.linalg.norm(displacement)) < 1e-10:
            continue
        # ||x - point||^2 <= ||x - other||^2, in scipy's a.x + b <= 0 form.
        coefficient = 2.0 * displacement
        constant = float(np.dot(point, point) - np.dot(other, other))
        halfspaces.append(np.concatenate([coefficient, np.asarray([constant])]))
    for axis in range(3):
        upper_coefficient = np.zeros(3, dtype=np.float64)
        upper_coefficient[axis] = 1.0
        halfspaces.append(np.concatenate([upper_coefficient, np.asarray([-upper[axis]])]))
        lower_coefficient = np.zeros(3, dtype=np.float64)
        lower_coefficient[axis] = -1.0
        halfspaces.append(np.concatenate([lower_coefficient, np.asarray([lower[axis]])]))
    try:
        vertices = np.asarray(
            HalfspaceIntersection(np.asarray(halfspaces, dtype=np.float64), point).intersections,
            dtype=np.float64,
        )
        if vertices.shape[0] < 4:
            return None
        ConvexHull(vertices)
    except (QhullError, ValueError, np.linalg.LinAlgError):
        return None
    return vertices


def compute_voronoi_geometry(
    pmg_obj: PMGObject,
    cutoff: float,
    site_indices: list[int],
) -> tuple[list[VoronoiCell], list[VoronoiFace]]:
    """Return closed central-site Voronoi cells and their shared faces.

    Periodic cells are obtained from a replicated structure. Molecular cells
    with an unbounded region are clipped only as a rendering fallback by a
    distant box; the finite interior regions remain exact Voronoi cells.
    """

    points, orig_idx, is_center = _voronoi_points(pmg_obj, cutoff)
    if points.shape[0] == 0:
        return [], []
    is_periodic = isinstance(pmg_obj, Structure)
    lower, upper = _molecule_voronoi_bounds(points, cutoff) if not is_periodic else (None, None)
    vor: Voronoi | None = None
    if points.shape[0] >= 5:
        try:
            vor = Voronoi(points)
        except (QhullError, ValueError):
            vor = None

    central_points: dict[int, int] = {}
    for point_index in np.flatnonzero(is_center):
        central_points.setdefault(int(orig_idx[point_index]), int(point_index))

    cells: list[VoronoiCell] = []
    for site_index in sorted(set(int(index) for index in site_indices)):
        point_index = central_points.get(site_index)
        if point_index is None:
            continue
        vertices: np.ndarray | None = None
        if vor is not None:
            region = vor.regions[vor.point_region[point_index]]
            if region and -1 not in region:
                candidate = np.asarray(vor.vertices[np.asarray(region, dtype=np.int64)], dtype=np.float64)
                if candidate.shape[0] >= 4:
                    try:
                        ConvexHull(candidate)
                        vertices = candidate
                    except (QhullError, ValueError):
                        vertices = None
        if vertices is None and not is_periodic:
            assert lower is not None and upper is not None
            vertices = _clipped_molecule_cell(points, point_index, lower, upper)
        if vertices is not None:
            cells.append((site_index, vertices))

    valid_sites = {site_index for site_index, _ in cells}
    if vor is None or not valid_sites:
        return cells, []

    faces: dict[tuple[int, ...], VoronoiFace] = {}
    for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        if len(ridge_vertices) < 3 or -1 in ridge_vertices:
            continue
        p0, p1 = int(ridge_points[0]), int(ridge_points[1])
        central_selected = [
            int(orig_idx[point_index])
            for point_index in (p0, p1)
            if bool(is_center[point_index]) and int(orig_idx[point_index]) in valid_sites
        ]
        if not central_selected:
            continue
        face_vertices = np.asarray(
            vor.vertices[np.asarray(ridge_vertices, dtype=np.int64)],
            dtype=np.float64,
        )
        if face_vertices.shape[0] < 3:
            continue
        key = tuple(sorted(int(vertex_index) for vertex_index in ridge_vertices))
        owner = central_selected[0]
        p0_site = int(orig_idx[p0])
        p1_site = int(orig_idx[p1])
        other = p1_site if owner == p0_site else p0_site
        faces.setdefault(key, (face_vertices, owner, other))
    return cells, list(faces.values())


def select_voronoi_cell_sites(
    scope: str,
    target_site_idx: int,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    n_atoms: int,
) -> list[int]:
    """Select target, first-shell, or all atom indices for cell rendering."""

    if scope == "target":
        return [target_site_idx]
    if scope == "all":
        return list(range(n_atoms))
    sites = {target_site_idx}
    for source, destination in zip(edge_src.tolist(), edge_dst.tolist()):
        if int(source) == target_site_idx:
            sites.add(int(destination))
        if int(destination) == target_site_idx:
            sites.add(int(source))
    return sorted(index for index in sites if 0 <= index < n_atoms)


def _cell_points(cells: list[VoronoiCell]) -> np.ndarray:
    """Flatten cell vertices for axis-limit calculation."""

    if not cells:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate([vertices for _, vertices in cells], axis=0)


def draw_unit_cell(ax: Any, structure: Structure) -> None:
    """Draw a unit-cell wireframe, including for non-orthogonal lattices."""

    corners_frac = np.asarray(list(itertools.product([0.0, 1.0], repeat=3)), dtype=np.float64)
    corners = _cell_corner_coords(structure)
    segments: list[np.ndarray] = []
    for i, j in itertools.combinations(range(8), 2):
        if np.sum(np.abs(corners_frac[i] - corners_frac[j])) == 1.0:
            segments.append(np.stack([corners[i], corners[j]], axis=0))
    for segment in segments:
        (line,) = ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=CELL_COLOR,
            linewidth=_line_width(CELL_LINE_WIDTH),
            linestyle="-",
        )
        line.set_zorder(_depth_zorder(ax, segment.mean(axis=0)))


def _projected_depth(ax: Any, point: np.ndarray) -> float:
    """Return the camera-projected depth of one 3D point."""

    point_np = np.asarray(point, dtype=np.float64)
    _, _, projected_depth = proj3d.proj_transform(
        float(point_np[0]),
        float(point_np[1]),
        float(point_np[2]),
        ax.get_proj(),
    )
    return float(projected_depth)


def _depth_zorder(ax: Any, point: np.ndarray) -> float:
    """Return a painter's z-order for a point in the current 3D camera view."""

    depth = _projected_depth(ax, point)
    if not np.isfinite(depth):
        return DEPTH_ZORDER_BASE
    # Matplotlib's projected depth is smaller for points nearer the camera.
    return DEPTH_ZORDER_BASE - DEPTH_ZORDER_SCALE * depth


def _projected_depth_range(ax: Any, points: np.ndarray) -> tuple[float, float]:
    """Return the finite projected-depth range used for edge color shading."""

    points_np = np.asarray(points, dtype=np.float64)
    if points_np.size == 0:
        return (0.0, 1.0)
    _, _, depths = proj3d.proj_transform(
        points_np[:, 0],
        points_np[:, 1],
        points_np[:, 2],
        ax.get_proj(),
    )
    finite_depths = np.asarray(depths, dtype=np.float64)
    finite_depths = finite_depths[np.isfinite(finite_depths)]
    if finite_depths.size == 0:
        return (0.0, 1.0)
    return float(finite_depths.min()), float(finite_depths.max())


def _depth_shaded_color(
    ax: Any,
    point: np.ndarray,
    depth_range: tuple[float, float],
    near_color: tuple[float, float, float, float],
    far_color: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Shade an edge from its near color to a darker far color."""

    depth = _projected_depth(ax, point)
    near_depth, far_depth = depth_range
    if not np.isfinite(depth) or not np.isfinite(near_depth) or not np.isfinite(far_depth):
        far_fraction = 0.0
    elif far_depth - near_depth < 1e-12:
        far_fraction = 0.0
    else:
        # Projected depth increases away from the camera.
        far_fraction = float(np.clip((depth - near_depth) / (far_depth - near_depth), 0.0, 1.0))
        far_fraction = float(np.clip(0.5 + EDGE_DEPTH_CONTRAST * (far_fraction - 0.5), 0.0, 1.0))
    near_rgb = np.asarray(near_color[:3], dtype=np.float64)
    far_rgb = np.asarray(far_color[:3], dtype=np.float64)
    rgb = (1.0 - far_fraction) * near_rgb + far_fraction * far_rgb
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(near_color[3]))


def _voronoi_cell_color(
    site_index: int,
    atomic_numbers: np.ndarray,
    target_site_idx: int,
    mark_target_site: bool,
) -> tuple[float, float, float, float]:
    """Return a soft fill color for one Voronoi cell."""

    if mark_target_site and site_index == target_site_idx:
        return VORONOI_TARGET_FACE_COLOR
    element_rgb = np.asarray(
        element_colors(np.asarray([atomic_numbers[site_index]], dtype=np.int64))[0],
        dtype=np.float64,
    )
    # Keep the Jmol identity visible while lifting dark elements into a
    # light, publication-friendly cell tint.
    rgb = 0.62 * np.ones(3, dtype=np.float64) + 0.38 * element_rgb
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), VORONOI_CELL_FACE_COLOR[3])


def _shade_voronoi_face(
    color: tuple[float, float, float, float],
    triangle: np.ndarray,
) -> tuple[float, float, float, float]:
    """Apply restrained directional shading to one triangular cell face."""

    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-12:
        return color
    normal /= normal_norm
    light_direction = np.asarray((0.38, 0.48, 0.79), dtype=np.float64)
    light_direction /= np.linalg.norm(light_direction)
    brightness = 0.78 + 0.22 * abs(float(np.dot(normal, light_direction)))
    rgb = np.clip(np.asarray(color[:3], dtype=np.float64) * brightness, 0.0, 1.0)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(color[3]))


def plot_voronoi_cells(
    ax: Any,
    cells: list[VoronoiCell],
    faces: list[VoronoiFace],
    atomic_numbers: np.ndarray,
    target_site_idx: int,
    mark_target_site: bool,
) -> None:
    """Draw translucent closed Voronoi cells and clean shared-face outlines."""

    regular_triangles: list[np.ndarray] = []
    regular_colors: list[tuple[float, float, float, float]] = []
    target_triangles: list[np.ndarray] = []
    target_colors: list[tuple[float, float, float, float]] = []
    for site_index, vertices in cells:
        try:
            hull = ConvexHull(vertices)
        except (QhullError, ValueError):
            continue
        cell_color = _voronoi_cell_color(
            site_index,
            atomic_numbers,
            target_site_idx,
            mark_target_site,
        )
        triangles = target_triangles if mark_target_site and site_index == target_site_idx else regular_triangles
        colors = target_colors if mark_target_site and site_index == target_site_idx else regular_colors
        for simplex in hull.simplices:
            triangle = np.asarray(vertices[np.asarray(simplex, dtype=np.int64)], dtype=np.float64)
            if triangle.shape != (3, 3):
                continue
            triangles.append(triangle)
            colors.append(_shade_voronoi_face(cell_color, triangle))

    if regular_triangles:
        regular_collection = Poly3DCollection(
            regular_triangles,
            facecolors=regular_colors,
            edgecolors="none",
            linewidths=0.0,
            zsort="average",
        )
        regular_collection.set_zorder(-20.0)
        ax.add_collection3d(regular_collection)
    if target_triangles:
        target_collection = Poly3DCollection(
            target_triangles,
            facecolors=target_colors,
            edgecolors="none",
            linewidths=0.0,
            zsort="average",
        )
        target_collection.set_zorder(-15.0)
        ax.add_collection3d(target_collection)

    for face_vertices, owner, other in faces:
        is_target_face = mark_target_site and (owner == target_site_idx or other == target_site_idx)
        color = VORONOI_TARGET_EDGE_COLOR if is_target_face else VORONOI_CELL_EDGE_COLOR
        linewidth = _line_width(VORONOI_TARGET_CELL_LINE_WIDTH if is_target_face else VORONOI_CELL_LINE_WIDTH)
        for start, end in zip(face_vertices, np.roll(face_vertices, -1, axis=0)):
            (line,) = ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                linewidth=linewidth,
                solid_capstyle="round",
            )
            line.set_zorder(-10.0)


def plot_atoms(
    ax: Any,
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    target_site_idx: int,
    label_atoms: bool,
    mark_target_site: bool = True,
) -> None:
    """Draw color-coded atoms and optionally highlight the target site."""

    colors = element_colors(atomic_numbers)
    marker_sizes = np.clip(
        EDGE_ATOM_SIZE_SCALE * (55.0 + 2.2 * atomic_numbers.astype(np.float64)),
        80.0,
        240.0,
    )
    for index, color in enumerate(colors):
        atom = ax.scatter(
            [coords[index, 0]],
            [coords[index, 1]],
            [coords[index, 2]],
            color=color,
            s=float(marker_sizes[index]),
            edgecolors="black",
            linewidths=_line_width(0.65),
            depthshade=False,
        )
        atom.set_zorder(_depth_zorder(ax, coords[index]))

    if mark_target_site:
        target_size = float(marker_sizes[target_site_idx])
        target_ring = ax.scatter(
            coords[target_site_idx, 0],
            coords[target_site_idx, 1],
            coords[target_site_idx, 2],
            s=max(EDGE_TARGET_RING_MIN_SIZE, target_size * 2.6),
            facecolors="none",
            edgecolors=[TARGET_COLOR],
            linewidths=_line_width(2.0),
            depthshade=False,
        )
        target_ring.set_zorder(DEPTH_OVERLAY_ZORDER)

    if label_atoms:
        for i, (x, y, z_coord) in enumerate(coords):
            ax.text(
                x,
                y,
                z_coord,
                f" {element_symbol(int(atomic_numbers[i]))}{i}",
                fontsize=7,
                color="black",
                zorder=DEPTH_OVERLAY_ZORDER,
            )


def element_legend_handles(atomic_numbers: np.ndarray) -> list[Line2D]:
    """Build compact legend handles for the elements present in a sample."""

    colors = element_colors(atomic_numbers)
    handles: list[Line2D] = []
    seen: set[int] = set()
    for i, z_value in enumerate(atomic_numbers.tolist()):
        z = int(z_value)
        if z in seen:
            continue
        seen.add(z)
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=PUBLICATION_LEGEND_MARKER_SIZE,
                markerfacecolor=colors[i],
                markeredgecolor="black",
                markeredgewidth=_line_width(0.65),
                label=element_symbol(z),
            )
        )
    return handles


def target_legend_handle(target_site_idx: int, color: Any = TARGET_COLOR) -> Line2D:
    """Build a consistent legend handle for the highlighted target site."""

    return Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=PUBLICATION_LEGEND_MARKER_SIZE,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=_line_width(1.8),
        label=f"target site ({target_site_idx})",
    )


def voronoi_cell_legend_handle() -> Patch:
    """Build a legend handle for translucent Voronoi cell faces."""

    return Patch(
        facecolor=VORONOI_CELL_FACE_COLOR,
        edgecolor=VORONOI_CELL_EDGE_COLOR,
        linewidth=_line_width(VORONOI_CELL_LINE_WIDTH),
        label="Voronoi cell",
    )


def graph_legend_handles(
    atomic_numbers: np.ndarray,
    target_site_idx: int,
    is_periodic: bool,
    n_visual: int | None = None,
    n_pbc: int = 0,
    raw_edge_count: int | None = None,
    include_periodic_styles: bool = False,
    mark_target_site: bool = True,
    include_voronoi_cells: bool = False,
) -> list[Line2D | Patch]:
    """Build the shared legend used by edge figures and standalone legends."""

    handles: list[Line2D | Patch] = []
    handles.extend(element_legend_handles(atomic_numbers))
    edge_label = "graph edge" if n_visual is None else f"graph edge ({n_visual})"
    handles.append(Line2D([], [], color=EDGE_COLOR, linewidth=_line_width(EDGE_LINE_WIDTH), label=edge_label))
    if include_voronoi_cells:
        handles.append(voronoi_cell_legend_handle())
    if is_periodic and include_periodic_styles:
        pbc_label = "periodic edge" if n_pbc == 0 else f"periodic edge ({n_pbc})"
        handles.append(
            Line2D(
                [],
                [],
                color=PBC_EDGE_COLOR,
                linewidth=_line_width(PBC_EDGE_LINE_WIDTH),
                linestyle=(0, (5, 3)),
                label=pbc_label,
            )
        )
        handles.append(
            Line2D(
                [],
                [],
                color=PBC_EDGE_COLOR,
                linewidth=0.0,
                marker="o",
                markersize=PUBLICATION_LEGEND_MARKER_SIZE,
                markerfacecolor="white",
                markeredgecolor=PBC_EDGE_COLOR,
                markeredgewidth=_line_width(0.9),
                label="image atom",
            )
        )
    if mark_target_site:
        handles.append(target_legend_handle(target_site_idx))
    if is_periodic:
        handles.append(
            Line2D(
                [],
                [],
                color=CELL_COLOR,
                linewidth=_line_width(CELL_LINE_WIDTH),
                label="unit-cell boundary",
            )
        )
    if raw_edge_count is not None and n_visual is not None and raw_edge_count != n_visual:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="",
                label=f"directed edges: {raw_edge_count}",
            )
        )
    return handles


def _periodic_translation(
    source: int,
    destination: int,
    edge_vec: np.ndarray,
    coords: np.ndarray,
    lattice: np.ndarray,
) -> tuple[int, int, int]:
    """Recover the integer lattice translation encoded by a periodic edge."""

    displacement_from_destination = coords[source] + edge_vec - coords[destination]
    fractional_translation = displacement_from_destination @ np.linalg.inv(lattice)
    rounded_translation = np.rint(fractional_translation).astype(np.int64)
    return int(rounded_translation[0]), int(rounded_translation[1]), int(rounded_translation[2])


def _edge_key(
    source: int,
    destination: int,
    edge_vec: np.ndarray,
    coords: np.ndarray,
    lattice: np.ndarray | None,
    is_periodic: bool,
) -> EdgeKey:
    """Return a key that removes reverse duplicates without losing PBC edges."""

    if is_periodic and lattice is not None:
        translation = _periodic_translation(source, destination, edge_vec, coords, lattice)
        reverse_translation = tuple(-value for value in translation)
        forward = (source, destination, translation)
        reverse = (destination, source, reverse_translation)
        return min(forward, reverse)
    return (min(source, destination), max(source, destination))


def prepare_visual_edges(
    coords: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_vec: np.ndarray,
    is_periodic: bool,
    lattice: np.ndarray | None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[bool]]:
    """Prepare unique edge segments and identify periodic-boundary crossings.

    For a periodic structure, each segment starts at an atom in the displayed
    unit cell and ends at the translated image encoded by ``edge_vec``. Reverse
    graph directions are collapsed into one visual bond, while distinct lattice
    translations are retained.
    """

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    crosses_boundary: list[bool] = []
    seen: set[EdgeKey] = set()
    for source_value, destination_value, vector in zip(edge_src.tolist(), edge_dst.tolist(), edge_vec):
        source = int(source_value)
        destination = int(destination_value)
        vector_np = np.asarray(vector, dtype=np.float64)
        key = _edge_key(source, destination, vector_np, coords, lattice, is_periodic)
        if key in seen:
            continue
        seen.add(key)
        start = np.asarray(coords[source], dtype=np.float64)
        end = start + vector_np
        starts.append(start)
        ends.append(end)
        crosses_boundary.append(is_periodic and not np.allclose(end, coords[destination], atol=1e-6))
    return starts, ends, crosses_boundary


def plot_edges(
    ax: Any,
    starts: list[np.ndarray],
    ends: list[np.ndarray],
    crosses_boundary: list[bool],
    depth_range: tuple[float, float],
) -> None:
    """Draw graph edges with camera-depth-aware atom/edge layering.

    Each edge and atom is kept as an individual 3D artist.  Their z-orders are
    assigned from the current camera projection so that a nearer atom can
    cover an edge while a farther atom remains behind it.  The edge depth is
    represented by its midpoint, which is the appropriate painter's-algorithm
    approximation for a single 3D line artist.
    """

    intra_segments: list[np.ndarray] = []
    pbc_segments: list[np.ndarray] = []
    pbc_endpoints: list[np.ndarray] = []
    for start, end, crosses in zip(starts, ends, crosses_boundary):
        segment = np.stack([start, end], axis=0)
        if crosses:
            pbc_segments.append(segment)
            pbc_endpoints.append(end)
        else:
            intra_segments.append(segment)

    for segment in intra_segments:
        (line,) = ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=_depth_shaded_color(ax, segment.mean(axis=0), depth_range, EDGE_COLOR, EDGE_FAR_COLOR),
            linewidth=_line_width(EDGE_LINE_WIDTH),
            linestyle="-",
            solid_capstyle="round",
        )
        line.set_zorder(_depth_zorder(ax, segment.mean(axis=0)))
    for segment in pbc_segments:
        (line,) = ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=_depth_shaded_color(
                ax,
                segment.mean(axis=0),
                depth_range,
                PBC_EDGE_COLOR,
                PBC_EDGE_FAR_COLOR,
            ),
            linewidth=_line_width(PBC_EDGE_LINE_WIDTH),
            linestyle=(0, (5, 3)),
            dash_capstyle="round",
        )
        line.set_zorder(_depth_zorder(ax, segment.mean(axis=0)))
    for endpoint in pbc_endpoints:
        image_atom = ax.scatter(
            [endpoint[0]],
            [endpoint[1]],
            [endpoint[2]],
            s=EDGE_IMAGE_MARKER_SIZE,
            facecolors="white",
            edgecolors=[PBC_EDGE_COLOR],
            linewidths=_line_width(0.9),
            alpha=0.85,
            depthshade=False,
        )
        image_atom.set_zorder(_depth_zorder(ax, endpoint))


def equalize_3d_axes(
    ax: Any,
    points: np.ndarray,
    elev: float,
    azim: float,
    zoom: float = 1.0,
    show_axes: bool = True,
    show_axis_labels: bool = True,
) -> None:
    """Set equal, padded 3D limits for an axis and its camera view."""

    if points.size == 0:
        points = np.zeros((1, 3), dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    max_range = float(np.max(ranges))
    if not np.isfinite(max_range) or max_range < 1e-8:
        max_range = 1.0
    zoom_factor = float(zoom) if np.isfinite(zoom) and zoom > 0.0 else 1.0
    # A zoomed graph overlay deliberately gets tighter data limits.  This
    # magnifies every graph artist together, while the separate base axis
    # keeps its original limits, ticks, labels, and title.
    padding = EDGE_FRAME_PADDING / zoom_factor
    half_range = 0.5 * max_range * padding
    center = 0.5 * (mins + maxs)
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    if show_axes and show_axis_labels:
        ax.set_xlabel("x [Å]", labelpad=2, fontsize=PUBLICATION_LABEL_SIZE)
        ax.set_ylabel("y [Å]", labelpad=2, fontsize=PUBLICATION_LABEL_SIZE)
        ax.set_zlabel("z [Å]", labelpad=2, fontsize=PUBLICATION_LABEL_SIZE)
    if show_axes:
        ax.tick_params(axis="both", which="major", labelsize=PUBLICATION_TICK_SIZE, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((0.75, 0.75, 0.75, 0.65))


def _register_latin_modern_fonts() -> str:
    """Register LaTeX's Latin Modern Roman font when it is installed."""

    font_paths = [
        font_path
        for directory in LATIN_MODERN_FONT_DIRECTORIES
        if directory.is_dir()
        for font_path in directory.glob("lmroman*.otf")
    ]
    for font_path in font_paths:
        font_manager.fontManager.addfont(str(font_path))
    return "Latin Modern Roman" if font_paths else "serif"


def configure_publication_style() -> None:
    """Use stable typography and an installed LaTeX-style serif font."""

    publication_font_family = _register_latin_modern_fonts()
    plt.rcParams.update(
        {
            "font.family": publication_font_family,
            "font.serif": PUBLICATION_SERIF_FONTS,
            "font.size": PUBLICATION_FONT_SIZE,
            "axes.titlesize": PUBLICATION_TITLE_SIZE,
            "axes.titleweight": "normal",
            "axes.labelsize": PUBLICATION_LABEL_SIZE,
            "axes.linewidth": _line_width(0.8),
            "xtick.labelsize": PUBLICATION_TICK_SIZE,
            "ytick.labelsize": PUBLICATION_TICK_SIZE,
            "legend.fontsize": PUBLICATION_LEGEND_SIZE,
            "mathtext.fontset": "cm",
            "figure.dpi": 150,
            "savefig.dpi": PUBLICATION_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def make_structure_figure(
    pmg_obj: PMGObject,
    stem: str,
    atomic_numbers: np.ndarray,
    target_site_idx: int,
    label_atoms: bool,
    show_legend: bool = True,
    zoom: float = 1.0,
    mark_target_site: bool = True,
) -> Figure:
    """Create the standalone analysis-style structure figure."""

    fig = plt.figure(figsize=(4.8, 4.8), facecolor="white")
    ax = fig.add_subplot(111)
    _draw_structure_projection(
        ax,
        pmg_obj,
        atomic_numbers,
        target_site_idx,
        label_atoms,
        show_legend,
        zoom,
        mark_target_site,
    )
    ax.set_title(f"Structure · {stem}", loc="left", pad=6)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)
    return fig


def make_edges_figure(
    pmg_obj: PMGObject,
    stem: str,
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    target_site_idx: int,
    is_periodic: bool,
    starts: list[np.ndarray],
    ends: list[np.ndarray],
    crosses_boundary: list[bool],
    points: np.ndarray,
    graph_method: str,
    cutoff: float,
    raw_edge_count: int,
    label_atoms: bool,
    elev: float,
    azim: float,
    show_legend: bool = True,
    zoom: float = 1.0,
    mark_target_site: bool = True,
    show_axis_labels: bool = True,
) -> Figure:
    """Create the standalone graph-edge figure."""

    fig = plt.figure(figsize=(7.4, 6.4), facecolor="white")
    axis_ax = fig.add_subplot(111, projection="3d")
    # Keep the coordinate axes on an unzoomed base layer.  The structure is
    # drawn on a transparent overlay so --zoom changes only the graph layer.
    equalize_3d_axes(
        axis_ax,
        points,
        elev,
        azim,
        zoom=1.0,
        show_axis_labels=show_axis_labels,
    )
    fig.subplots_adjust(left=0.0, right=0.78 if show_legend else 0.98, bottom=0.02, top=0.91)

    axis_position = axis_ax.get_position()
    graph_position: tuple[float, float, float, float] = (
        float(axis_position.x0),
        float(axis_position.y0),
        float(axis_position.width),
        float(axis_position.height),
    )
    graph_ax: Any = fig.add_axes(graph_position, projection="3d", label="graph-depth-overlay")
    # Axes3D depth-sorts collections and lines separately.  Disable that
    # mixed-artist behavior because edges and atoms must share one z-order.
    graph_ax.computed_zorder = False
    graph_ax.set_zorder(axis_ax.get_zorder() + 1)
    graph_ax.patch.set_alpha(0.0)
    graph_ax.set_facecolor("none")
    equalize_3d_axes(graph_ax, points, elev, azim, zoom, show_axes=False)
    graph_ax.set_axis_off()
    edge_midpoints = (
        np.asarray([(start + end) / 2.0 for start, end in zip(starts, ends)], dtype=np.float64) if starts else points
    )
    # Normalize against the edge depths themselves so the available blue
    # range is used even when atom/cell points occupy a wider depth range.
    depth_range = _projected_depth_range(graph_ax, edge_midpoints)
    n_visual = len(starts)
    n_pbc = sum(crosses_boundary)
    kind = "periodic graph" if is_periodic else "molecular graph"
    if is_periodic:
        draw_unit_cell(graph_ax, pmg_obj)
    plot_edges(graph_ax, starts, ends, crosses_boundary, depth_range)
    plot_atoms(
        graph_ax,
        coords,
        atomic_numbers,
        target_site_idx,
        label_atoms,
        mark_target_site,
    )
    axis_ax.set_title(
        f"Graph edges · {stem}\n{kind} · {graph_method} · cutoff={cutoff:g} Å · {n_visual} bonds",
        loc="left",
        pad=10,
    )
    if show_legend:
        handles = graph_legend_handles(
            atomic_numbers,
            target_site_idx,
            is_periodic,
            n_visual=n_visual,
            n_pbc=n_pbc,
            raw_edge_count=raw_edge_count,
            include_periodic_styles=n_pbc > 0,
            mark_target_site=mark_target_site,
        )
        axis_ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            borderaxespad=0.0,
            fontsize=PUBLICATION_LEGEND_SIZE,
        )
    return fig


def make_voronoi_cells_figure(
    pmg_obj: PMGObject,
    stem: str,
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    target_site_idx: int,
    is_periodic: bool,
    starts: list[np.ndarray],
    ends: list[np.ndarray],
    crosses_boundary: list[bool],
    points: np.ndarray,
    cells: list[VoronoiCell],
    faces: list[VoronoiFace],
    cell_scope: str,
    graph_method: str,
    cutoff: float,
    raw_edge_count: int,
    label_atoms: bool,
    elev: float,
    azim: float,
    show_legend: bool = True,
    zoom: float = 1.0,
    mark_target_site: bool = True,
    show_axis_labels: bool = True,
) -> Figure:
    """Create a graph-style figure with closed Voronoi cells overlaid."""

    fig = plt.figure(figsize=(7.4, 6.4), facecolor="white")
    axis_ax = fig.add_subplot(111, projection="3d")
    equalize_3d_axes(
        axis_ax,
        points,
        elev,
        azim,
        zoom=1.0,
        show_axis_labels=show_axis_labels,
    )
    fig.subplots_adjust(left=0.0, right=0.78 if show_legend else 0.98, bottom=0.02, top=0.91)

    axis_position = axis_ax.get_position()
    graph_position: tuple[float, float, float, float] = (
        float(axis_position.x0),
        float(axis_position.y0),
        float(axis_position.width),
        float(axis_position.height),
    )
    graph_ax: Any = fig.add_axes(graph_position, projection="3d", label="voronoi-cell-overlay")
    graph_ax.computed_zorder = False
    graph_ax.set_zorder(axis_ax.get_zorder() + 1)
    graph_ax.patch.set_alpha(0.0)
    graph_ax.set_facecolor("none")
    equalize_3d_axes(graph_ax, points, elev, azim, zoom, show_axes=False)
    graph_ax.set_axis_off()

    edge_midpoints = (
        np.asarray([(start + end) / 2.0 for start, end in zip(starts, ends)], dtype=np.float64) if starts else points
    )
    depth_range = _projected_depth_range(graph_ax, edge_midpoints)
    n_visual = len(starts)
    n_pbc = sum(crosses_boundary)
    kind = "periodic graph" if is_periodic else "molecular graph"
    if is_periodic:
        draw_unit_cell(graph_ax, pmg_obj)
    plot_voronoi_cells(graph_ax, cells, faces, atomic_numbers, target_site_idx, mark_target_site)
    plot_edges(graph_ax, starts, ends, crosses_boundary, depth_range)
    plot_atoms(
        graph_ax,
        coords,
        atomic_numbers,
        target_site_idx,
        label_atoms,
        mark_target_site,
    )
    axis_ax.set_title(
        f"Voronoi cells · {stem}\n{kind} · {graph_method} · {cell_scope} · "
        f"cutoff={cutoff:g} Å · {len(cells)} cells",
        loc="left",
        pad=10,
    )
    if show_legend:
        handles = graph_legend_handles(
            atomic_numbers,
            target_site_idx,
            is_periodic,
            n_visual=n_visual,
            n_pbc=n_pbc,
            raw_edge_count=raw_edge_count,
            include_periodic_styles=n_pbc > 0,
            mark_target_site=mark_target_site,
            include_voronoi_cells=True,
        )
        axis_ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            borderaxespad=0.0,
            fontsize=PUBLICATION_LEGEND_SIZE,
        )
    return fig


def make_legend_figure(
    atomic_numbers: np.ndarray,
    target_site_idx: int,
    is_periodic: bool,
    stem: str,
    mark_target_site: bool = True,
    include_voronoi_cells: bool = False,
) -> Figure:
    """Create one shared legend suitable for placing beside multiple figures."""

    handles = graph_legend_handles(
        atomic_numbers,
        target_site_idx,
        is_periodic,
        include_periodic_styles=is_periodic,
        mark_target_site=mark_target_site,
        include_voronoi_cells=include_voronoi_cells,
    )
    n_columns = min(4, len(handles))
    n_rows = int(np.ceil(len(handles) / n_columns))
    fig = plt.figure(figsize=(7.2, 0.85 + 0.42 * n_rows), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.legend(
        handles=handles,
        loc="center",
        ncol=n_columns,
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.6,
        handlelength=1.4,
        handletextpad=0.45,
        labelspacing=0.65,
        fontsize=PUBLICATION_LEGEND_SIZE,
    )
    ax.set_title(f"Legend · {stem}", loc="left", pad=3)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.86)
    return fig


def save_figure(fig: Figure, path: Path | None) -> None:
    """Save a figure at publication resolution when a path is supplied."""

    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Axes3D does not include its projected axis labels in its tight bounding
    # box.  Include them explicitly so labels keep the same padding on all
    # three axes without being clipped by publication cropping.
    bbox_extra_artists: list[Any] = []
    for ax in fig.axes:
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis = getattr(ax, axis_name, None)
            label = getattr(axis, "label", None)
            if label is not None and label.get_visible():
                bbox_extra_artists.append(label)
        legend = ax.get_legend()
        if legend is not None and legend.get_visible():
            bbox_extra_artists.append(legend)
    fig.savefig(
        path,
        dpi=PUBLICATION_DPI,
        bbox_inches="tight",
        bbox_extra_artists=bbox_extra_artists,
        pad_inches=0.05,
        facecolor="white",
    )
    print(f"Figure saved to {path}")


def main() -> None:
    """Run the publication graph visualization command-line interface."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json-dir", type=Path, required=True, help="PMGJSON datasource directory (one .json per sample)"
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument("--index", type=int, help="Sample index into the sorted datasource")
    sample_group.add_argument("--file", type=str, help="Sample file stem without the .json suffix")
    parser.add_argument("--cutoff", type=float, default=6.0, help="Maximum graph edge length in angstroms")
    parser.add_argument("--max-neighbors", type=int, default=32, help="Maximum outgoing neighbors per atom")
    parser.add_argument(
        "--graph-method",
        type=str,
        default="radius",
        choices=list(GraphBuilderRegistry.list()),
        help="Edge construction method",
    )
    parser.add_argument(
        "--min-facet-area",
        type=str,
        default=None,
        help="Voronoi only: absolute area in A^2 or a percentage such as '1.0%%'",
    )
    parser.add_argument(
        "--cov-radii-scale",
        type=float,
        default=1.5,
        help="cov_radius only: scale factor on the covalent-radius sum",
    )
    parser.add_argument("--target-site-idx", type=int, default=0, help="Atom index to use as the target site")
    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument("--label-atoms", action="store_true", help="Draw element/index labels beside atoms")
    label_group.add_argument(
        "--no-atom-labels", action="store_true", help="Do not draw element/index labels beside atoms"
    )
    parser.add_argument(
        "--no-target-marker",
        "--no-target-site-marker",
        dest="no_target_marker",
        action="store_true",
        help="Do not highlight the selected target site in figures or legends",
    )
    parser.add_argument(
        "--no-axis-labels",
        "--no-xyz-labels",
        dest="no_axis_labels",
        action="store_true",
        help="Do not draw the x, y, and z labels on 3D graph figures",
    )
    parser.add_argument("--view-elev", type=float, default=22.0, help="3D camera elevation in degrees")
    parser.add_argument("--view-azim", type=float, default=-58.0, help="3D camera azimuth in degrees")
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.12,
        help="Edge-plot zoom (>1 enlarges graph geometry while keeping axes and text fixed)",
    )
    parser.add_argument(
        "--legend-mode",
        choices=LEGEND_MODES,
        default="per-figure",
        help="Show legends per figure, save a standalone legend, show both, or hide legends",
    )
    parser.add_argument(
        "--save-structure", type=Path, default=None, help="Output path for the standalone structure figure"
    )
    parser.add_argument(
        "--save-edges", type=Path, default=None, help="Output path for the standalone graph-edge figure"
    )
    parser.add_argument(
        "--save-voronoi-cells",
        "--save-cells",
        dest="save_voronoi_cells",
        type=Path,
        default=None,
        help="Output path for a graph-style figure with closed Voronoi cells",
    )
    parser.add_argument(
        "--voronoi-cell-scope",
        "--cell-scope",
        choices=VORONOI_CELL_SCOPES,
        default="first-shell",
        help="Cells to draw: target, its graph first shell, or all sites",
    )
    parser.add_argument(
        "--save-legend", type=Path, default=None, help="Output path for a standalone shared legend figure"
    )
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening an interactive window")
    args = parser.parse_args()

    if args.legend_mode in {"standalone", "both"} and args.save_legend is None:
        parser.error("--legend-mode standalone/both requires --save-legend")

    configure_publication_style()
    pmg_obj = load_sample(args.json_dir, args.index, args.file)
    stem = str(pmg_obj.properties.get("sample_id", "<unknown>"))
    is_periodic = isinstance(pmg_obj, Structure)
    coords = np.asarray(pmg_obj.cart_coords, dtype=np.float64)
    atomic_numbers = np.asarray(pmg_obj.atomic_numbers, dtype=np.int64)
    n_atoms = len(pmg_obj)
    if not (0 <= args.target_site_idx < n_atoms):
        raise SystemExit(f"--target-site-idx {args.target_site_idx} out of range [0, {n_atoms})")

    min_facet_area = args.min_facet_area
    if min_facet_area is not None and not min_facet_area.endswith("%"):
        min_facet_area = float(min_facet_area)
    builder_kwargs: dict[str, object] = {
        "graph_builder_type": args.graph_method,
        "cutoff": args.cutoff,
        "max_num_neighbors": args.max_neighbors,
    }
    if args.graph_method == "cov_radius":
        builder_kwargs["cov_radii_scale"] = args.cov_radii_scale
    if args.graph_method == "voronoi":
        builder_kwargs["min_facet_area"] = min_facet_area
    builder = GraphBuilderRegistry.create(args.graph_method, **builder_kwargs)
    edge_index, _, edge_vec, _ = builder.build(pmg_obj, compute_vectors=True)
    assert edge_vec is not None
    edge_src = edge_index[0].numpy()
    edge_dst = edge_index[1].numpy()
    edge_vec_np = edge_vec.numpy()
    lattice = np.asarray(pmg_obj.lattice.matrix, dtype=np.float64) if is_periodic else None
    starts, ends, crosses_boundary = prepare_visual_edges(
        coords,
        edge_src,
        edge_dst,
        edge_vec_np,
        is_periodic,
        lattice,
    )
    edge_ends = np.asarray(ends, dtype=np.float64) if ends else np.zeros((0, 3), dtype=np.float64)
    cell_corners = _cell_corner_coords(pmg_obj) if is_periodic else np.zeros((0, 3), dtype=np.float64)
    edge_points = np.concatenate([coords, cell_corners, edge_ends], axis=0)
    label_atoms = args.label_atoms and not args.no_atom_labels
    mark_target_site = not args.no_target_marker
    voronoi_cells: list[VoronoiCell] = []
    voronoi_faces: list[VoronoiFace] = []
    voronoi_cells_fig: Figure | None = None
    if args.save_voronoi_cells is not None:
        cell_site_indices = select_voronoi_cell_sites(
            args.voronoi_cell_scope,
            args.target_site_idx,
            edge_src,
            edge_dst,
            n_atoms,
        )
        voronoi_cells, voronoi_faces = compute_voronoi_geometry(
            pmg_obj,
            args.cutoff,
            cell_site_indices,
        )
        voronoi_points = _cell_points(voronoi_cells)
        cell_figure_points = np.concatenate([edge_points, voronoi_points], axis=0)
        voronoi_cells_fig = make_voronoi_cells_figure(
            pmg_obj,
            stem,
            coords,
            atomic_numbers,
            args.target_site_idx,
            is_periodic,
            starts,
            ends,
            crosses_boundary,
            cell_figure_points,
            voronoi_cells,
            voronoi_faces,
            args.voronoi_cell_scope,
            args.graph_method,
            args.cutoff,
            int(edge_index.shape[1]),
            label_atoms,
            args.view_elev,
            args.view_azim,
            show_legend=args.legend_mode in {"per-figure", "both"},
            zoom=args.zoom,
            mark_target_site=mark_target_site,
            show_axis_labels=not args.no_axis_labels,
        )
    structure_fig = make_structure_figure(
        pmg_obj,
        stem,
        atomic_numbers,
        args.target_site_idx,
        label_atoms,
        show_legend=args.legend_mode in {"per-figure", "both"},
        zoom=1.0,
        mark_target_site=mark_target_site,
    )
    edges_fig = make_edges_figure(
        pmg_obj,
        stem,
        coords,
        atomic_numbers,
        args.target_site_idx,
        is_periodic,
        starts,
        ends,
        crosses_boundary,
        edge_points,
        args.graph_method,
        args.cutoff,
        int(edge_index.shape[1]),
        label_atoms,
        args.view_elev,
        args.view_azim,
        show_legend=args.legend_mode in {"per-figure", "both"},
        zoom=args.zoom,
        mark_target_site=mark_target_site,
        show_axis_labels=not args.no_axis_labels,
    )
    legend_fig = (
        make_legend_figure(
            atomic_numbers,
            args.target_site_idx,
            is_periodic,
            stem,
            mark_target_site,
            include_voronoi_cells=args.save_voronoi_cells is not None,
        )
        if args.save_legend is not None and args.legend_mode != "none"
        else None
    )
    save_figure(structure_fig, args.save_structure)
    save_figure(edges_fig, args.save_edges)
    if voronoi_cells_fig is not None:
        assert args.save_voronoi_cells is not None
        save_figure(voronoi_cells_fig, args.save_voronoi_cells)
    if legend_fig is not None:
        assert args.save_legend is not None
        save_figure(legend_fig, args.save_legend)

    print("=" * 66)
    print(f"datasource:      {args.json_dir}")
    print(f"sample:          {stem}")
    print(f"kind:            {'periodic Structure' if is_periodic else 'Molecule'}")
    print(f"# atoms:         {n_atoms}")
    print(f"target site:     idx={args.target_site_idx} ({element_symbol(int(atomic_numbers[args.target_site_idx]))})")
    print(f"target marker:   {'enabled' if mark_target_site else 'disabled'}")
    print(f"cutoff:          {args.cutoff} A   max_neighbors: {args.max_neighbors}")
    print(f"graph method:    {args.graph_method}")
    print(f"directed edges:  {edge_index.shape[1]}   visual bonds: {len(starts)}")
    print(f"PBC crossings:   {sum(crosses_boundary)}")
    if voronoi_cells_fig is not None:
        print(f"Voronoi cells:   {len(voronoi_cells)} ({args.voronoi_cell_scope})")
        print(f"Voronoi faces:   {len(voronoi_faces)}")
    print(f"legend mode:     {args.legend_mode}")
    print("=" * 66)

    if not args.no_show:
        plt.show()
    plt.close(structure_fig)
    plt.close(edges_fig)
    if voronoi_cells_fig is not None:
        plt.close(voronoi_cells_fig)
    if legend_fig is not None:
        plt.close(legend_fig)


if __name__ == "__main__":
    main()
