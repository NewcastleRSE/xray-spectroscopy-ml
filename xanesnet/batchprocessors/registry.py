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

"""Registry instance for XANESNET batch processor classes."""

from xanesnet.utils.registry import Registry

from .base import BatchProcessor

BatchProcessorKey = tuple[str, str]


def _normalize_key(key: BatchProcessorKey) -> BatchProcessorKey:
    """Return a case-insensitive dataset/model registry key."""
    dataset_type, model_type = key
    return dataset_type.lower(), model_type.lower()


def _format_key(key: BatchProcessorKey) -> str:
    """Return a readable dataset/model key for error messages."""
    dataset_type, model_type = key
    return f"for {dataset_type}, {model_type}"


BatchProcessorRegistry: Registry[type[BatchProcessor], BatchProcessorKey] = Registry(
    "BatchProcessor",
    normalize_key=_normalize_key,
    format_key=_format_key,
)
