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

"""Plotter that renders aggregated statistics as table PDFs and LaTeX sources."""

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar, cast

import matplotlib.pyplot as plt

from ..reporters.base import selector_label
from ..result import AnalysisResults
from .base import Plotter
from .registry import PlotterRegistry
from .utils import method_label_lines

_LATEX_PREAMBLE = r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{float}
\usepackage[table]{xcolor}
\usepackage{multirow}
\pagestyle{empty}
\begin{document}
"""

_LATEX_FOOTER = "\\end{document}\n"

_LATEX_MARK_COLOURS: dict[str, str] = {"best": "green!15", "worst": "red!15"}

_MATPLOTLIB_MARK_COLOURS: dict[str, str] = {"best": "#d5f5d5", "worst": "#f5d5d5"}


@PlotterRegistry.register("stat_table")
class StatTablePlotter(Plotter):
    """Render comparison tables of aggregated statistics as PDF figures.

    For each scalar value key found across aggregator results, a table is
    produced where rows are prediction-reader/selector combinations and columns are statistics such
    as ``mean``, ``std``, and ``median``.

    In addition, a compact combined table is rendered per aggregator. Rows are
    grouped by method (prediction-reader/selector combination) with one
    sub-row per statistic key, and columns are the scalar value keys. The row
    label of each method group is merged into a single cell spanning its
    sub-rows.

    For every rendered PDF a standalone LaTeX source (``.tex``) is written
    alongside and, when a LaTeX installation is available, compiled to a
    separate ``<stem>_latex.pdf``.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        stat_keys: Ordered statistic keys used as table columns and as
            sub-rows of the combined table.
        precision: Number of significant digits used when formatting table values.
    """

    DEFAULT_STAT_KEYS: ClassVar[list[str]] = ["mean", "std", "median", "min", "max"]

    def __init__(
        self,
        plotter_type: str,
        stat_keys: list[str] | None = None,
        precision: int = 4,
    ) -> None:
        """Initialize a statistics table plotter."""
        super().__init__(plotter_type)
        self.stat_keys = stat_keys if stat_keys is not None else self.DEFAULT_STAT_KEYS
        self.precision = precision

    def plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write table PDFs and LaTeX sources for aggregated scalar statistics.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``stat_tables`` tree should be written.
        """
        if not results.aggregator_results:
            logging.info("    No aggregator results available, skipping.")
            return

        root = output_dir / "stat_tables"

        table_data: dict[tuple[str, int, str], dict[str, dict[str, float]]] = {}
        combined_data: dict[tuple[str, int], dict[str, dict[str, dict[str, float]]]] = {}
        row_order: list[str] = []
        value_order: dict[tuple[str, int], list[str]] = {}
        reader_names = results.prediction_names
        row_label_lines: dict[str, list[str]] = {}

        for reader_idx, reader_results in enumerate(results.aggregator_results):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.aggregator_results)}.")

            for sel_idx, agg_results in enumerate(reader_results):
                sel_label_str = selector_label(results.selectors_config, sel_idx)
                sel_cfg = results.selectors_config[sel_idx]
                reader_name = reader_names[reader_idx]
                label_lines = method_label_lines(reader_name, sel_label_str, sel_cfg)
                row_label = "  |  ".join(label_lines)
                if row_label not in row_order:
                    row_order.append(row_label)
                    row_label_lines[row_label] = label_lines

                for agg_result in agg_results:
                    agg_key = (agg_result.aggregator_type, agg_result.aggregator_index)
                    for value_key, stats in agg_result.data.items():
                        if not isinstance(stats, dict):
                            continue
                        stat_vals = cast(dict[str, float], stats)
                        table_data.setdefault((*agg_key, value_key), {})[row_label] = stat_vals
                        combined_data.setdefault(agg_key, {}).setdefault(row_label, {})[value_key] = stat_vals
                        if value_key not in value_order.setdefault(agg_key, []):
                            value_order[agg_key].append(value_key)

        if not table_data:
            logging.info("    No table data collected, skipping.")
            return

        for (agg_type, agg_idx, value_key), rows in table_data.items():
            agg_dir = root / f"{agg_type}_{agg_idx:03d}"
            agg_dir.mkdir(parents=True, exist_ok=True)
            filepath = agg_dir / f"{value_key}.pdf"

            self._render_table(rows, value_key, agg_type, agg_idx, filepath)

        for agg_key, rows in combined_data.items():
            agg_type, agg_idx = agg_key
            agg_dir = root / f"{agg_type}_{agg_idx:03d}"
            agg_dir.mkdir(parents=True, exist_ok=True)
            value_keys = value_order.get(agg_key, [])
            combined_name = "combined.pdf" if "combined" not in value_keys else "combined_table.pdf"
            filepath = agg_dir / combined_name

            self._render_combined_table(rows, row_order, value_keys, row_label_lines, agg_type, agg_idx, filepath)

    def _render_table(
        self,
        rows: dict[str, dict[str, float]],
        value_key: str,
        agg_type: str,
        agg_idx: int,
        filepath: Path,
    ) -> None:
        """Render a single comparison table to a PDF using Matplotlib.

        Additionally, a standalone LaTeX source with the same stem is written
        and compiled to ``<stem>_latex.pdf`` when a LaTeX installation is
        available.

        Args:
            rows: Mapping from row label to statistic values.
            value_key: Scalar value key represented by the table.
            agg_type: Registered aggregator name that produced the statistics.
            agg_idx: Zero-based aggregator index from the analysis configuration.
            filepath: Destination PDF path.
        """
        row_labels = list(rows.keys())
        col_labels = [s for s in self.stat_keys if any(s in stats for stats in rows.values())]

        if not col_labels or not row_labels:
            return

        cell_text: list[list[str]] = []
        cell_values: list[list[float | None]] = []
        for rl in row_labels:
            stats = rows[rl]
            text_row: list[str] = []
            val_row: list[float | None] = []
            for cl in col_labels:
                v = stats.get(cl)
                if v is not None:
                    text_row.append(f"{v:.{self.precision}g}")
                    val_row.append(v)
                else:
                    text_row.append("-")
                    val_row.append(None)
            cell_text.append(text_row)
            cell_values.append(val_row)

        cell_marks = self._cell_marks(cell_values)
        cell_colours = self._cell_colours(cell_marks)

        n_rows, n_cols = len(row_labels), len(col_labels)
        fig_width = max(4, 1.4 * n_cols + 2)
        fig_height = max(1.6, 0.38 * n_rows + 1.2)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellColours=cell_colours,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)

        for (r, c), cell in table.get_celld().items():
            cell.PAD = 0.03
            if r == 0:
                cell.set_facecolor("#f0f0f0")
                cell.set_text_props(weight="bold")
            if c == -1:
                cell.set_text_props(fontsize=7, ha="right")

        fig.tight_layout()
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)

        latex_body = self._latex_single_table_body(value_key, row_labels, col_labels, cell_values, cell_marks)
        self._write_latex(filepath, latex_body)

    def _render_combined_table(
        self,
        rows: dict[str, dict[str, dict[str, float]]],
        row_order: list[str],
        value_keys: list[str],
        row_label_lines: dict[str, list[str]],
        agg_type: str,
        agg_idx: int,
        filepath: Path,
    ) -> None:
        """Render one compact combined table for all scalar value keys to a PDF.

        Additionally, a standalone LaTeX source with the same stem is written
        and compiled to ``<stem>_latex.pdf`` when a LaTeX installation is
        available.

        Rows are grouped by method (prediction-reader/selector combination)
        with one sub-row per statistic key. Columns are the scalar value keys
        plus a leading statistic-name column. The row label of each method
        group is merged into a single cell spanning all its sub-rows, with the
        prediction reader name and selector stacked on separate lines when the
        group has at least two sub-rows.

        Args:
            rows: Mapping from row label to value key to statistic values.
            row_order: Row labels in first-seen order.
            value_keys: Scalar value keys in first-seen order.
            row_label_lines: Label lines per row label used for the merged
                group cells.
            agg_type: Registered aggregator name that produced the statistics.
            agg_idx: Zero-based aggregator index from the analysis configuration.
            filepath: Destination PDF path.
        """
        value_cols = [vk for vk in value_keys if any(rows[rl].get(vk) for rl in row_order)]
        row_labels = [rl for rl in row_order if rl in rows and any(rows[rl].get(vk) for vk in value_cols)]
        if not value_cols or not row_labels:
            return

        stat_keys = [
            sk for sk in self.stat_keys if any(sk in rows[rl].get(vk, {}) for rl in row_labels for vk in value_cols)
        ]
        if not stat_keys:
            return

        cell_text: list[list[str]] = []
        cell_values: list[list[float | None]] = []
        row_stats: list[str] = []
        groups: list[tuple[int, int]] = []

        def add_sub_row(row_label: str, stat_key: str) -> None:
            """Append one sub-row for the given method and statistic key."""
            text_row: list[str] = [stat_key]
            val_row: list[float | None] = []
            for vk in value_cols:
                v = rows[row_label].get(vk, {}).get(stat_key)
                if v is not None:
                    text_row.append(f"{v:.{self.precision}g}")
                    val_row.append(v)
                else:
                    text_row.append("-")
                    val_row.append(None)
            cell_text.append(text_row)
            cell_values.append(val_row)
            row_stats.append(stat_key)

        for rl in row_labels:
            start = len(cell_text)
            for sk in stat_keys:
                if any(sk in rows[rl].get(vk, {}) for vk in value_cols):
                    add_sub_row(rl, sk)
            if len(cell_text) == start:
                # Fallback: no configured stat key matches this method's data;
                # render one sub-row using the first statistic key found.
                fallback_sk = next((sk for vk in value_cols for sk in rows[rl].get(vk, {})), None)
                if fallback_sk is not None:
                    add_sub_row(rl, fallback_sk)
            groups.append((start, len(cell_text) - 1))

        cell_marks = self._cell_marks(cell_values, row_groups=row_stats)
        cell_colours = [["white"] + colours_row for colours_row in self._cell_colours(cell_marks)]

        n_rows = len(cell_text)
        n_cols = 1 + len(value_cols)
        fontsize = 8 if n_rows <= 12 else 7
        fig_width = max(4, 1.1 * n_cols + 2)
        fig_height = max(2.0, 0.3 * n_rows + 1.4)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=cell_text,
            rowLabels=[""] * n_rows,
            colLabels=[""] + value_cols,
            cellColours=cell_colours,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1, 1.15)

        for (r, c), cell in table.get_celld().items():
            cell.PAD = 0.03
            if r == 0:
                cell.set_facecolor("#f0f0f0")
                cell.set_text_props(weight="bold", fontsize=fontsize)
                continue
            if c == 0:
                cell.set_text_props(fontsize=fontsize - 1, ha="left")
            elif c == -1:
                cell.set_text_props(fontsize=7, ha="left")

        # Merge the row-label cell of each method group across its sub-rows.
        # The reader name and selector are spread over the first sub-rows so
        # long prediction names do not widen the label column.
        for group_idx, (start, end) in enumerate(groups):
            lines = row_label_lines[row_labels[group_idx]]
            for r in range(start, end + 1):
                cell = table[r + 1, -1]
                cell.visible_edges = "LR" + ("T" if r == start else "") + ("B" if r == end else "")
                cell.set_facecolor("#eef1f8")
                if end == start:
                    cell.get_text().set_text("  |  ".join(lines))
                else:
                    line = lines[r - start] if r - start < len(lines) else ""
                    cell.get_text().set_text(line)

        fig.tight_layout()
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)

        latex_body = self._latex_combined_table_body(
            groups, row_labels, row_label_lines, value_cols, cell_text, cell_values, cell_marks
        )
        self._write_latex(filepath, latex_body)

    @staticmethod
    def _cell_marks(
        cell_values: list[list[float | None]],
        row_groups: list[Any] | None = None,
    ) -> list[list[str | None]]:
        """Determine per-cell best/worst marks for highlighting.

        Within each column, the best (lowest) and worst (highest) values are
        marked. When ``row_groups`` is given, best and worst are determined
        among rows that share the same group key, so only comparable rows are
        compared (e.g. one group per statistic key).

        Args:
            cell_values: Numeric table values with missing values represented by ``None``.
            row_groups: Optional per-row group keys; rows with the same key are
                compared against each other within each column.

        Returns:
            Matrix of ``"best"``, ``"worst"``, or ``None`` entries matching ``cell_values``.
        """
        n_rows = len(cell_values)
        n_cols = len(cell_values[0]) if cell_values else 0
        marks: list[list[str | None]] = [[None] * n_cols for _ in range(n_rows)]

        if n_rows < 2:
            return marks

        groups = row_groups if row_groups is not None else [0] * n_rows

        for c in range(n_cols):
            for group_key in sorted(set(groups), key=str):
                col_vals: list[tuple[int, float]] = []
                for r in range(n_rows):
                    if groups[r] != group_key:
                        continue
                    v = cell_values[r][c]
                    if v is not None:
                        col_vals.append((r, v))
                if len(col_vals) < 2:
                    continue
                sorted_vals = sorted(col_vals, key=lambda x: x[1])
                marks[sorted_vals[0][0]][c] = "best"
                marks[sorted_vals[-1][0]][c] = "worst"

        return marks

    @staticmethod
    def _cell_colours(cell_marks: list[list[str | None]]) -> list[list[str]]:
        """Map best/worst cell marks to Matplotlib background colours.

        Args:
            cell_marks: Per-cell marks from ``_cell_marks``.

        Returns:
            Matrix of Matplotlib-compatible colour strings matching ``cell_marks``.
        """
        return [
            [_MATPLOTLIB_MARK_COLOURS[mark] if mark is not None else "white" for mark in marks_row]
            for marks_row in cell_marks
        ]

    def _latex_single_table_body(
        self,
        value_key: str,
        row_labels: list[str],
        col_labels: list[str],
        cell_values: list[list[float | None]],
        cell_marks: list[list[str | None]],
    ) -> str:
        """Build the LaTeX table float for a single-value statistics table.

        Rows are the methods and columns the statistic keys; best and worst
        cells are highlighted with ``\\cellcolor``. The table is scaled to
        ``\\textwidth`` (change to ``\\columnwidth`` directly in the
        generated code for two-column layouts) and wrapped in a float with a
        placeholder caption and label.

        Args:
            value_key: Scalar value key represented by the table.
            row_labels: Row labels (methods).
            col_labels: Column labels (statistic keys).
            cell_values: Numeric table values with missing values represented by ``None``.
            cell_marks: Per-cell marks from ``_cell_marks`` aligned with ``cell_values``.

        Returns:
            Complete ``table`` float as a LaTeX string.
        """
        col_spec = "l" + "c" * len(col_labels)
        caption = f"PLACEHOLDER: statistics for '{_escape_latex(value_key)}'."
        label = f"tab:stat_{_sanitize_label(value_key)}"
        lines: list[str] = [
            r"\begin{table}[H]",
            r"\centering",
            r"% Width: change \textwidth to \columnwidth for two-column layouts.",
            r"\resizebox{\textwidth}{!}{%",
            r"\scriptsize",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\hline",
            r"\rowcolor{gray!12} & " + " & ".join(_escape_latex(cl) for cl in col_labels) + r" \\",
            r"\hline",
        ]
        for r, row_label in enumerate(row_labels):
            cells = [_latex_label_text(row_label)]
            for c in range(len(col_labels)):
                cells.append(_latex_cell(cell_values[r][c], cell_marks[r][c], self.precision))
            lines.append(" & ".join(cells) + r" \\")
        lines.extend(
            [
                r"\hline",
                r"\end{tabular}%",
                "}",
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                r"\end{table}",
            ]
        )
        return "\n".join(lines)

    def _latex_combined_table_body(
        self,
        groups: list[tuple[int, int]],
        row_labels: list[str],
        row_label_lines: dict[str, list[str]],
        value_cols: list[str],
        cell_text: list[list[str]],
        cell_values: list[list[float | None]],
        cell_marks: list[list[str | None]],
    ) -> str:
        """Build the LaTeX table float for the combined statistics table.

        Rows are grouped by method with one sub-row per statistic key; the
        method label is merged across its sub-rows with ``\\multirow`` and
        stacks the reader name and selector with ``\\shortstack`` when the
        group has at least two sub-rows. The table is scaled to ``\\textwidth``
        (change to ``\\columnwidth`` directly in the generated code for
        two-column layouts) and wrapped in a float with a placeholder caption
        and label.

        Args:
            groups: Sub-row index ranges (start, end) for each method group.
            row_labels: Method labels in the same order as ``groups``.
            row_label_lines: Label lines per row label.
            value_cols: Scalar value keys used as data columns.
            cell_text: Table cell texts including the leading statistic-name column.
            cell_values: Numeric table values for the value columns only.
            cell_marks: Per-cell marks from ``_cell_marks`` aligned with ``cell_values``.

        Returns:
            Complete ``table`` float as a LaTeX string.
        """
        caption = "PLACEHOLDER: combined statistics table."
        label = "tab:stat_combined"
        lines: list[str] = [
            r"\begin{table}[H]",
            r"\centering",
            r"% Width: change \textwidth to \columnwidth for two-column layouts.",
            r"\resizebox{\textwidth}{!}{%",
            r"\scriptsize",
            rf"\begin{{tabular}}{{ll{'c' * len(value_cols)}}}",
            r"\hline",
            r"\rowcolor{gray!12} & & " + " & ".join(_escape_latex(vk) for vk in value_cols) + r" \\",
            r"\hline",
        ]
        for group_idx, (start, end) in enumerate(groups):
            if start > end:
                continue
            n_sub_rows = end - start + 1
            label_lines = row_label_lines[row_labels[group_idx]]
            if n_sub_rows >= 2 and len(label_lines) >= 2:
                merged_label = (
                    r"\shortstack[l]{"
                    + _latex_label_text(label_lines[0])
                    + r"\\"
                    + _latex_label_text(label_lines[1])
                    + "}"
                )
            else:
                merged_label = _latex_label_text("  |  ".join(label_lines))
            for r in range(start, end + 1):
                label_cell = rf"\multirow{{{n_sub_rows}}}{{*}}{{{merged_label}}}" if r == start else ""
                cells = [label_cell, _escape_latex(cell_text[r][0])]
                for c in range(len(value_cols)):
                    cells.append(_latex_cell(cell_values[r][c], cell_marks[r][c], self.precision))
                lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\hline")
        lines.extend(
            [
                r"\end{tabular}%",
                "}",
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                r"\end{table}",
            ]
        )
        return "\n".join(lines)

    def _write_latex(self, filepath: Path, body: str) -> None:
        """Write a standalone LaTeX document for a rendered table.

        The LaTeX source uses the same stem as the Matplotlib-rendered PDF and
        is compiled to ``<stem>_latex.pdf`` when a LaTeX installation is
        available. The Matplotlib-rendered PDF is never overwritten.

        Args:
            filepath: Path of the Matplotlib-rendered PDF.
            body: LaTeX table float for the table.
        """
        tex_path = filepath.with_suffix(".tex")
        tex_path.write_text(
            f"{_LATEX_PREAMBLE}{body}\n{_LATEX_FOOTER}",
            encoding="utf-8",
        )
        out_pdf = filepath.with_name(f"{filepath.stem}_latex.pdf")
        _compile_latex_pdf(tex_path, out_pdf)


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in plain text.

    Args:
        text: Plain text to escape.

    Returns:
        LaTeX-safe representation of ``text``.
    """
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("_", r"\_")
        .replace("~", r"\textasciitilde{}")
    )


def _latex_label_text(text: str) -> str:
    """Escape one table label line for LaTeX.

    Args:
        text: Plain text label line.

    Returns:
        LaTeX-safe label line with pipe characters rendered in math mode.
    """
    return _escape_latex(text).replace("|", "$|$")


def _format_latex_number(value: float, precision: int) -> str:
    """Format a numeric value for LaTeX output.

    Values that ``g`` formatting renders in scientific notation are converted
    to a compact ``$m \\times 10^{e}$`` math expression.

    Args:
        value: Numeric value to format.
        precision: Number of significant digits.

    Returns:
        LaTeX-formatted number as a string.
    """
    text = f"{value:.{precision}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return text


def _latex_cell(value: float | None, mark: str | None, precision: int) -> str:
    """Format one table cell for LaTeX.

    Args:
        value: Numeric cell value; missing values are rendered as a dash.
        mark: Optional ``"best"``/``"worst"`` mark controlling colour highlighting.
        precision: Number of significant digits.

    Returns:
        LaTeX cell content with an optional ``\\cellcolor`` prefix.
    """
    text = _format_latex_number(value, precision) if value is not None else "-"
    colour = _LATEX_MARK_COLOURS[mark] if mark is not None else ""
    return rf"\cellcolor{{{colour}}} {text}" if colour else text


def _sanitize_label(text: str) -> str:
    """Convert a value key into a safe LaTeX label suffix.

    Args:
        text: Value key to sanitize.

    Returns:
        Sanitized suffix containing only letters, digits, and underscores.
    """
    return re.sub(r"[^0-9A-Za-z_]", "_", text)


def _compile_latex_pdf(tex_path: Path, out_pdf: Path) -> bool:
    """Compile a standalone LaTeX table document to PDF.

    Compilation runs in a temporary directory so no auxiliary files are left
    behind. The compiled page is cropped to the table with ``pdfcrop`` when
    available. When no LaTeX installation is found or compilation fails, the
    Matplotlib-rendered PDF remains the rendered figure.

    Args:
        tex_path: Path to the standalone LaTeX document.
        out_pdf: Destination path for the compiled PDF.

    Returns:
        True when the PDF was compiled and copied to ``out_pdf``.
    """
    exe = shutil.which("pdflatex")
    if exe is None:
        logging.info("    pdflatex not found, skipping LaTeX rendering: %s", tex_path.name)
        return False

    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir)
        work_tex = work_dir / tex_path.name
        shutil.copy(tex_path, work_tex)
        try:
            proc = subprocess.run(
                [exe, "-interaction=nonstopmode", "-halt-on-error", work_tex.name],
                cwd=work_dir,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            proc = None
        compiled_pdf = work_dir / f"{work_tex.stem}.pdf"
        if proc is None or proc.returncode != 0 or not compiled_pdf.exists():
            logging.warning(
                "    LaTeX compilation failed for %s, keeping Matplotlib-rendered PDF.",
                tex_path.name,
            )
            return False

        crop_exe = shutil.which("pdfcrop")
        if crop_exe is not None:
            cropped_pdf = work_dir / f"{work_tex.stem}_crop.pdf"
            crop_proc = subprocess.run(
                [crop_exe, "--margins", "2", compiled_pdf.name, cropped_pdf.name],
                cwd=work_dir,
                capture_output=True,
                timeout=30,
                check=False,
            )
            if crop_proc.returncode == 0 and cropped_pdf.exists():
                shutil.copy(cropped_pdf, out_pdf)
                return True
        shutil.copy(compiled_pdf, out_pdf)
        return True
