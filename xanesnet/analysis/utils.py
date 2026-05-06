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

"""Shared helpers for analysis scalar values."""

from typing import Any, TypeGuard

import numpy as np

ScalarValue = int | float | np.integer | np.floating


def is_scalar_value(value: Any) -> TypeGuard[ScalarValue]:
    """Return whether ``value`` is a non-boolean numeric scalar.

    Args:
        value: Candidate value from a prediction sample or collector output.

    Returns:
        ``True`` when ``value`` is a Python or NumPy integer/floating scalar,
        excluding booleans.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, (np.integer, np.floating)):
        return True
    return False
