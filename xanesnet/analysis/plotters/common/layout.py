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

"""Figure layout helpers shared by every plotter.

Single-panel figures use :func:`compact_layout`. Figures that compare all
methods side by side use :func:`method_grid` to build a near-square grid of
cells and :func:`finish_grid` to hide unused cells and place the axis labels.
"""

import math

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Font size of the per-cell titles and tick labels inside a method grid. Cells
# are small, so they use tighter type than single-panel figures.
GRID_CELL_TITLE_FONTSIZE: float = 5.5
GRID_CELL_TICK_FONTSIZE: float = 5.5
GRID_AXIS_LABEL_FONTSIZE: float = 8.0

# Gaps between grid cells. Every method grid uses the same spacing; only the
# outer margins differ per plotter.
_GRID_WSPACE: float = 0.08
_GRID_HSPACE: float = 0.18


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
            the axes, as fractions of the figure.
        pad: Padding between the figure edge and the axes, as a fraction of
            the font size.
    """
    fig.tight_layout(pad=pad, rect=rect)


def grid_shape(n_cells: int) -> tuple[int, int]:
    """Return the near-square row and column count for a number of cells.

    Args:
        n_cells: Number of cells that must fit into the grid.

    Returns:
        ``(nrows, ncols)`` with ``nrows * ncols >= n_cells``.
    """
    ncols = math.ceil(math.sqrt(n_cells))
    nrows = math.ceil(n_cells / ncols)
    return nrows, ncols


def method_grid(
    n_cells: int,
    cell_width: float,
    cell_height: float,
    width_margin: float = 0.0,
    height_margin: float = 0.0,
    sharex: bool = True,
    sharey: bool = True,
) -> tuple[Figure, list[list[Axes]]]:
    """Create a near-square grid of cells, one per method.

    Args:
        n_cells: Number of methods to show.
        cell_width: Width of one cell in inches.
        cell_height: Height of one cell in inches.
        width_margin: Extra figure width in inches, for example for a colourbar.
        height_margin: Extra figure height in inches, for example for a legend.
        sharex: Whether all cells share one x range.
        sharey: Whether all cells share one y range.

    Returns:
        ``(figure, axes)`` where ``axes`` is always a nested row-major list of
        shape ``(nrows, ncols)``.
    """
    nrows, ncols = grid_shape(n_cells)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(cell_width * ncols + width_margin, cell_height * nrows + height_margin),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    return fig, [list(row) for row in axes]


def finish_grid(
    axes: list[list[Axes]],
    n_cells: int,
    xlabel: str,
    ylabel: str,
    fontsize: float = GRID_AXIS_LABEL_FONTSIZE,
) -> None:
    """Hide unused grid cells and label the outer cells of a method grid.

    Shared axes hide the tick labels of every cell but the bottom row, so the
    bottom-most used cell of each column has them restored. Every left-column
    cell receives the y label and every bottom-row cell the x label, which
    keeps the labels next to their tick values.

    Args:
        axes: Nested axes list from :func:`method_grid`.
        n_cells: Number of cells that carry data, in row-major order.
        xlabel: Label for the shared x quantity.
        ylabel: Label for the shared y quantity.
        fontsize: Font size of the axis labels.
    """
    nrows = len(axes)
    ncols = len(axes[0])
    for index in range(n_cells, nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    for col in range(ncols):
        used_rows = [row for row in range(nrows) if row * ncols + col < n_cells]
        if used_rows:
            axes[used_rows[-1]][col].tick_params(labelbottom=True)

    for row in range(nrows):
        axes[row][0].set_ylabel(ylabel, fontsize=fontsize)
    for col in range(ncols):
        axes[nrows - 1][col].set_xlabel(xlabel, fontsize=fontsize)


def style_grid_cell(ax: Axes, title_lines: list[str]) -> None:
    """Apply the shared grid-cell title and tick styling to one cell.

    The method label lines are joined into a single line so the prediction
    name and selector sit side by side instead of stacking and overlapping
    the cell above.

    Args:
        ax: Cell axis to style.
        title_lines: Stacked method label lines used as the cell title.
    """
    ax.set_title("  |  ".join(title_lines), fontsize=GRID_CELL_TITLE_FONTSIZE)
    ax.tick_params(labelsize=GRID_CELL_TICK_FONTSIZE)


def adjust_grid(
    fig: Figure,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> None:
    """Set the outer margins of a method grid.

    ``tight_layout`` reserves a large unusable band above grids, so method
    grids place their axes manually. Only the outer margins differ per
    plotter; the gaps between cells are identical everywhere.

    Args:
        fig: Matplotlib figure holding the grid.
        left: Left margin as a fraction of the figure width.
        right: Right margin as a fraction of the figure width.
        top: Top margin as a fraction of the figure height.
        bottom: Bottom margin as a fraction of the figure height.
    """
    fig.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=_GRID_WSPACE,
        hspace=_GRID_HSPACE,
    )
