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

"""Plotter that renders aggregated statistics as publication-ready LaTeX tables."""

import logging
from pathlib import Path

from xanesnet.serialization.config import Config

from ..result import AnalysisResults
from .base import Plotter
from .common import latex, stat_tables
from .registry import PlotterRegistry

# Marker appended to the header of the column whose values order the rows.
_SORT_INDICATOR = r" $\downarrow$"


@PlotterRegistry.register("stat_table_latex")
class StatTableLatexPlotter(Plotter):
    """Render comparison tables of aggregated statistics as LaTeX sources.

    The tables mirror the ones produced by the ``stat_table`` plotter: one
    table per scalar value key comparing all prediction-reader/selector
    combinations, plus a compact combined table per aggregator whose rows are
    grouped by method with one sub-row per statistic key.

    Each table is written as a standalone LaTeX document (``<stem>.tex``) that
    can be dropped into a paper, and is compiled to ``<stem>_latex.pdf`` when a
    LaTeX installation is available. The caption and label are placeholders
    that are meant to be edited afterwards.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        stat_keys: Ordered statistic keys used as table columns and as
            sub-rows of the combined table.
        precision: Number of significant digits used when formatting table values.
        sort_key: Scalar value key whose first statistic orders the combined
            table rows. ``None`` uses the first value key.
    """

    def __init__(
        self,
        plotter_type: str,
        stat_keys: list[str],
        precision: int,
        sort_key: str | None,
    ) -> None:
        """Initialize a LaTeX statistics table plotter."""
        super().__init__(plotter_type)
        self.stat_keys = stat_keys
        self.precision = precision
        self.sort_key = sort_key

    def plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write LaTeX sources and compiled PDFs for aggregated scalar statistics.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``stat_tables_latex`` tree should be written.
        """
        if not results.aggregator_results:
            logging.info("    No aggregator results available, skipping.")
            return

        root = output_dir / "stat_tables_latex"
        source = stat_tables.collect_tables(results)

        if not source.single:
            logging.info("    No table data collected, skipping.")
            return

        for (agg_type, agg_idx, value_key), rows in source.single.items():
            table = stat_tables.build_single_table(rows, self.stat_keys, self.precision)
            if table is None:
                continue
            agg_dir = root / stat_tables.aggregator_dir_name(agg_type, agg_idx)
            agg_dir.mkdir(parents=True, exist_ok=True)
            self._write(agg_dir / f"{value_key}.tex", self._single_table_body(table, value_key))

        for (agg_type, agg_idx), rows in source.combined.items():
            value_keys = source.value_order.get((agg_type, agg_idx), [])
            table = stat_tables.build_combined_table(
                rows, source.row_order, value_keys, self.stat_keys, self.precision, self.sort_key
            )
            if table is None:
                continue
            agg_dir = root / stat_tables.aggregator_dir_name(agg_type, agg_idx)
            agg_dir.mkdir(parents=True, exist_ok=True)
            stem = stat_tables.combined_stem(value_keys)
            self._write(agg_dir / f"{stem}.tex", self._combined_table_body(table, source.row_label_lines))

        logging.info(f"    Wrote LaTeX tables to '{root}'.")

    def _single_table_body(self, table: stat_tables.SingleTable, value_key: str) -> str:
        """Build the LaTeX table float for a single-value statistics table.

        Rows are the methods and columns the statistic keys; best and worst
        cells are highlighted with ``\\cellcolor``. The table is scaled to
        ``\\textwidth`` (change to ``\\columnwidth`` directly in the generated
        code for two-column layouts) and wrapped in a float with a placeholder
        caption and label.

        Args:
            table: Laid-out single-value table.
            value_key: Scalar value key represented by the table.

        Returns:
            Complete ``table`` float as a LaTeX string.
        """
        col_spec = "l" + "c" * len(table.col_labels)
        caption = f"PLACEHOLDER: statistics for '{latex.escape_latex(value_key)}'."
        label = f"tab:stat_{latex.sanitize_label(value_key)}"
        headers = [
            latex.escape_latex(label) + (_SORT_INDICATOR if idx == table.sort_col else "")
            for idx, label in enumerate(table.col_labels)
        ]
        lines: list[str] = [
            r"\begin{table}[H]",
            r"\centering",
            r"% Width: change \textwidth to \columnwidth for two-column layouts.",
            r"\resizebox{\textwidth}{!}{%",
            r"\scriptsize",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\hline",
            r"\rowcolor{gray!12} & " + " & ".join(headers) + r" \\",
            r"\hline",
        ]
        for r, row_label in enumerate(table.row_labels):
            cells = [latex.escape_label_line(row_label)]
            for c in range(len(table.col_labels)):
                cells.append(latex.format_cell(table.cell_values[r][c], table.cell_marks[r][c], self.precision))
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

    def _combined_table_body(
        self,
        table: stat_tables.CombinedTable,
        row_label_lines: dict[str, list[str]],
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
            table: Laid-out combined table.
            row_label_lines: Label lines per method row label.

        Returns:
            Complete ``table`` float as a LaTeX string.
        """
        caption = "PLACEHOLDER: combined statistics table."
        label = "tab:stat_combined"
        headers = [
            latex.escape_latex(value_col) + (_SORT_INDICATOR if idx == table.sort_col else "")
            for idx, value_col in enumerate(table.value_cols)
        ]
        lines: list[str] = [
            r"\begin{table}[H]",
            r"\centering",
            r"% Width: change \textwidth to \columnwidth for two-column layouts.",
            r"\resizebox{\textwidth}{!}{%",
            r"\scriptsize",
            rf"\begin{{tabular}}{{ll{'c' * len(table.value_cols)}}}",
            r"\hline",
            r"\rowcolor{gray!12} & & " + " & ".join(headers) + r" \\",
            r"\hline",
        ]
        for group_idx, (start, end) in enumerate(table.groups):
            if start > end:
                continue
            n_sub_rows = end - start + 1
            label_lines = row_label_lines[table.row_labels[group_idx]]
            if n_sub_rows >= 2 and len(label_lines) >= 2:
                merged_label = (
                    r"\shortstack[l]{"
                    + latex.escape_label_line(label_lines[0])
                    + r"\\"
                    + latex.escape_label_line(label_lines[1])
                    + "}"
                )
            else:
                merged_label = latex.escape_label_line("  |  ".join(label_lines))
            for r in range(start, end + 1):
                label_cell = rf"\multirow{{{n_sub_rows}}}{{*}}{{{merged_label}}}" if r == start else ""
                cells = [label_cell, latex.escape_latex(table.cell_text[r][0])]
                for c in range(len(table.value_cols)):
                    cells.append(latex.format_cell(table.cell_values[r][c], table.cell_marks[r][c], self.precision))
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

    @staticmethod
    def _write(tex_path: Path, body: str) -> None:
        """Write a standalone LaTeX document and compile it when possible.

        Args:
            tex_path: Destination ``.tex`` path.
            body: LaTeX table float for the table.
        """
        latex.write_document(tex_path, body)
        latex.compile_pdf(tex_path, tex_path.with_name(f"{tex_path.stem}_latex.pdf"))

    @property
    def signature(self) -> Config:
        """Return the LaTeX statistics table plotter signature."""
        signature = super().signature
        signature.update_with_dict(
            {"stat_keys": self.stat_keys, "precision": self.precision, "sort_key": self.sort_key}
        )
        return signature
