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

"""Shared helpers for the analysis pipeline."""

from collections.abc import Iterator, Mapping
from typing import Any, TypeGuard

import numpy as np
import torch

ScalarValue = int | float | np.integer | np.floating
VectorValue = list | tuple | np.ndarray | torch.Tensor

# Keys that identify or locate a sample rather than measure it. They are
# skipped by every stage that summarizes, reports, or plots scalar values, so
# that identifiers and site indices never appear as if they were metrics.
SAMPLE_METADATA_KEYS: frozenset[str] = frozenset({"sample_id", "target_site_index"})

# Keys whose values are the raw prediction and target spectra. Spectra are
# summarized by the ``spectrum`` aggregator, so vector summaries skip them.
SPECTRUM_KEYS: frozenset[str] = frozenset({"prediction", "target"})


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


def iter_scalar_items(
    record: Mapping[str, Any],
    include_metadata: bool = False,
) -> Iterator[tuple[str, ScalarValue]]:
    """Yield the measurable scalar entries of one prediction or collector record.

    Spectra arrays, structures, identifiers, and every other non-scalar or
    metadata entry are skipped, so all pipeline stages agree on which record
    entries count as scalar values.

    Args:
        record: Prediction sample or collector output mapping.
        include_metadata: Whether to also yield the scalar entries of
            :data:`SAMPLE_METADATA_KEYS`. Only per-sample views that describe a
            single sample should enable this; summaries over many samples must
            not treat identifiers as measurements.

    Yields:
        ``(key, value)`` pairs for every measurable scalar entry.
    """
    for key, value in record.items():
        if not include_metadata and key in SAMPLE_METADATA_KEYS:
            continue
        if is_scalar_value(value):
            yield key, value


def is_vector_value(value: Any) -> TypeGuard[VectorValue]:
    """Return whether ``value`` is a one-dimensional vector.

    Args:
        value: Candidate value from a prediction sample or collector output.

    Returns:
        ``True`` when ``value`` is a list, tuple, or one-dimensional NumPy
        array or torch tensor.
    """
    if isinstance(value, (list, tuple)):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 1
    if isinstance(value, torch.Tensor):
        return value.ndim == 1
    return False


def as_float_vector(value: VectorValue) -> np.ndarray:
    """Return ``value`` as a one-dimensional float array.

    Args:
        value: List, tuple, one-dimensional NumPy array, or one-dimensional
            torch tensor.

    Returns:
        The vector converted to a one-dimensional ``float`` NumPy array.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def iter_vector_items(
    record: Mapping[str, Any],
    include_metadata: bool = False,
) -> Iterator[tuple[str, VectorValue]]:
    """Yield the measurable vector entries of one prediction or collector record.

    Scalars, identifiers, the ``prediction`` and ``target`` spectra, and every
    other non-vector value are skipped, so the vector aggregator summarizes
    only vector measurements such as energy-resolved loss curves or per-sample
    encodings.

    Args:
        record: Prediction sample or collector output mapping.
        include_metadata: Whether to also yield the vector entries of
            :data:`SAMPLE_METADATA_KEYS`. Only per-sample views that describe a
            single sample should enable this; summaries over many samples must
            not treat identifiers as measurements.

    Yields:
        ``(key, value)`` pairs for every one-dimensional vector entry.
    """
    for key, value in record.items():
        if not include_metadata and key in SAMPLE_METADATA_KEYS:
            continue
        if key in SPECTRUM_KEYS:
            continue
        if is_vector_value(value):
            yield key, value


def component_repr(type_name: str, config: dict[str, Any]) -> str:
    """Build the detailed single-line representation of a configured component.

    Args:
        type_name: Name of the component class.
        config: Component configuration dictionary from a ``signature``.

    Returns:
        Representation of the form ``"ClassName(key=value, ...)"``.
    """
    args = ", ".join(f"{key}={value!r}" for key, value in config.items())
    return f"{type_name}({args})"
