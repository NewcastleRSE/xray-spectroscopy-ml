# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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

"""Shared helpers for analysis plotters."""

from typing import cast

from xanesnet.analysis.utils import ScalarValue, is_scalar_value
from xanesnet.serialization.jsonl_stream import JSONLStream

from ..selectors import Selector


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
                if key != "file_name" and is_scalar_value(val):
                    values.setdefault(key, []).append(cast(float, val))
            for key, val in col_sample.items():
                if key != "file_name" and is_scalar_value(val):
                    values.setdefault(key, []).append(cast(float, val))
    else:
        for sel_sample in selector:
            for key, val in sel_sample.items():
                if key != "file_name" and is_scalar_value(val):
                    values.setdefault(key, []).append(cast(float, val))
    return values
