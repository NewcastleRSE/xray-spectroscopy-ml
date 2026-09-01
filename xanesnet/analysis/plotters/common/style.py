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

"""Colors, fonts, and annotations shared by every plotter figure."""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

from .formatting import apply_decimal_tick_format

# Palette used to distinguish methods, cycled by method index. The first
# colors are maximally distinguishable; later colors fill the gaps and are
# necessarily closer together, so neighboring entries stay readable even when
# more methods than colors are drawn.
METHOD_COLORS: list[str] = [
    "#0072b2",  # blue
    "#e69f00",  # orange
    "#009e73",  # bluish green
    "#d55e00",  # vermillion
    "#cc79a7",  # reddish purple
    "#56b4e9",  # sky blue
    "#f0e442",  # yellow
    "#7a5195",  # purple
    "#8c564b",  # brown
    "#17becf",  # cyan
    "#7f7f7f",  # gray
    "#d62728",  # red
    "#bcbd22",  # olive
    "#e7298a",  # magenta
    "#000075",  # navy
    "#9a6324",  # bronze
    "#008080",  # teal
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
    "#98df8a",  # light green
]

# Fixed roles that must stay recognizable across all figures. The target is a
# neutral reference gray and the prediction a warm orange, chosen for clear
# contrast with each other and with the method palette.
COLOR_TARGET: str = "#5a5a5a"
COLOR_PREDICTION: str = "#e69f00"

# Accent colors for paired quantities such as best/worst subsets or the two
# terms of an error decomposition, plus warnings drawn onto a figure.
COLOR_ACCENT_RED: str = "#e85651"
COLOR_ACCENT_GREEN: str = "#3f9e6e"

# Neutral tones for annotations that must not compete with the data.
COLOR_BOX_EDGE: str = "#bbbbbb"
COLOR_SUBTITLE: str = "black"

# Uniform font sizes for every plotter figure. Explicit per-call font sizes
# elsewhere should match these values.
FONT_SIZES: dict[str, float] = {
    "axes.labelsize": 9.0,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
}

# Font size of figure-level subtitles and of the boxed annotations that report
# per-figure statistics.
SUBTITLE_FONTSIZE: float = 8.0
ANNOTATION_FONTSIZE: float = 7.0

# The x label needs a larger gap to its tick labels than the y label does.
_X_LABELPAD: float = 7.0
_Y_LABELPAD: float = 2.5


def method_color(index: int) -> str:
    """Return the palette color for a method index (cycling).

    Args:
        index: Zero-based method index in first-seen order.

    Returns:
        Matplotlib-compatible color string.
    """
    return METHOD_COLORS[index % len(METHOD_COLORS)]


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
    ax.xaxis.label.set_fontsize(FONT_SIZES["axes.labelsize"])
    ax.yaxis.label.set_fontsize(FONT_SIZES["axes.labelsize"])
    ax.tick_params(labelsize=FONT_SIZES["xtick.labelsize"])
    apply_decimal_tick_format(ax)


def add_subtitle(fig: Figure, text: str) -> None:
    """Add centered subtitle text below a figure.

    Figures carry no title; the subtitle holds the prediction and selector
    context instead.

    Args:
        fig: Matplotlib figure to annotate.
        text: Subtitle text.
    """
    fig.text(0.5, -0.01, text, ha="center", va="top", fontsize=SUBTITLE_FONTSIZE, color=COLOR_SUBTITLE)


def annotate_box(
    ax: Axes,
    text: str,
    x: float,
    y: float,
    horizontal_alignment: str,
    vertical_alignment: str,
    fontsize: float = ANNOTATION_FONTSIZE,
    alpha: float = 0.8,
    monospace: bool = True,
) -> Text:
    """Draw a boxed annotation in one corner of an axes.

    Args:
        ax: Matplotlib axis to annotate.
        text: Annotation text; newlines separate rows.
        x: Horizontal position in axes coordinates.
        y: Vertical position in axes coordinates.
        horizontal_alignment: Matplotlib horizontal alignment of the text.
        vertical_alignment: Matplotlib vertical alignment of the text.
        fontsize: Annotation font size.
        alpha: Opacity of the box background.
        monospace: Whether to render the text in a monospace font, which keeps
            multi-row ``key=value`` annotations aligned.

    Returns:
        The rendered text object, so callers can measure its extent.
    """
    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        horizontalalignment=horizontal_alignment,
        verticalalignment=vertical_alignment,
        fontfamily="monospace" if monospace else None,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=alpha, edgecolor=COLOR_BOX_EDGE),
    )
