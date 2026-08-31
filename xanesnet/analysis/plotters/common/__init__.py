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

"""Drawing building blocks shared by the analysis plotters.

Each module owns one concern, so a new plotter can pick exactly the pieces it
needs without pulling in unrelated code:

* :mod:`style` - colours, fonts, and boxed annotations.
* :mod:`layout` - figure layout and method grids.
* :mod:`spectra_pages` - single-sample spectra pages.
* :mod:`structures` - atomic structure rendering.
* :mod:`latex` - standalone LaTeX documents and their compilation.

The names re-exported here are the ones plotters use most; anything else is
imported from its module directly.
"""

from .layout import (
    adjust_grid,
    compact_layout,
    finish_grid,
    grid_shape,
    method_grid,
    style_grid_cell,
)
from .spectra_pages import (
    combined_spectra_page_figure,
    spectra_page_figure,
    spectra_structure_page_figure,
)
from .structures import draw_structure
from .style import (
    COLOUR_ACCENT_GREEN,
    COLOUR_ACCENT_RED,
    COLOUR_PREDICTION,
    COLOUR_TARGET,
    METHOD_COLOURS,
    add_subtitle,
    annotate_box,
    method_colour,
    style_axis,
)

__all__ = [
    "COLOUR_ACCENT_GREEN",
    "COLOUR_ACCENT_RED",
    "COLOUR_PREDICTION",
    "COLOUR_TARGET",
    "METHOD_COLOURS",
    "add_subtitle",
    "adjust_grid",
    "annotate_box",
    "combined_spectra_page_figure",
    "compact_layout",
    "draw_structure",
    "finish_grid",
    "grid_shape",
    "method_colour",
    "method_grid",
    "spectra_page_figure",
    "spectra_structure_page_figure",
    "style_axis",
    "style_grid_cell",
]
