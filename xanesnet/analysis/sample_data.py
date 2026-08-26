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

"""Access helpers for prediction samples and their collector values.

Aggregators, reporters, and plotters all read the same two aligned sources:
the samples yielded by a selector and the per-sample records a collector
stream wrote for them. These helpers pair the two sources, merge their scalar
values, and derive the per-sample error used to rank samples.
"""

from collections.abc import Iterator
from typing import Any, cast

import numpy as np

from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.serialization.prediction_readers import PredictionSample

from .selectors import Selector
from .utils import ScalarValue, is_scalar_value, iter_scalar_items


def iter_aligned(
    selector: Selector,
    stream: JSONLStream | None,
) -> Iterator[tuple[PredictionSample, dict[str, Any]]]:
    """Iterate selected samples together with their collector record.

    Selectors and collector streams are aligned by position: the collector
    stream was written while iterating the same selector, so the n-th stream
    record belongs to the n-th selected sample.

    Args:
        selector: Selector over prediction samples for one method.
        stream: Collector result stream aligned with ``selector``, or ``None``
            when no collectors were configured.

    Yields:
        ``(sample, collector_record)`` pairs. The record is empty when no
        collector stream is available.
    """
    if stream is None:
        for sample in selector:
            yield sample, {}
        return
    for sample, record in zip(selector, stream):
        yield sample, record


def merged_scalars(
    sample: PredictionSample,
    col_scalars: dict[str, Any],
    include_metadata: bool = False,
) -> dict[str, ScalarValue]:
    """Merge the scalar values of one sample and its collector record.

    Collector values take precedence over sample fields of the same name.

    Args:
        sample: Prediction sample providing scalar fields.
        col_scalars: Collector record aligned with ``sample``.
        include_metadata: Whether to also include sample metadata such as the
            target site index. Enable this only for views of a single sample.

    Returns:
        Mapping from scalar key to value.
    """
    scalars: dict[str, ScalarValue] = dict(iter_scalar_items(sample, include_metadata))
    scalars.update(iter_scalar_items(col_scalars, include_metadata))
    return scalars


def spectrum_error_value(
    sample: PredictionSample,
    col_scalars: dict[str, Any],
    sort_key: str | None,
) -> float:
    """Return the scalar used to rank one sample by prediction error.

    The configured ``sort_key`` is used when it holds a scalar value;
    otherwise the mean squared error between the predicted and target spectra
    is computed.

    Args:
        sample: Prediction sample containing spectra arrays and optional scalars.
        col_scalars: Collector record aligned with ``sample``.
        sort_key: Preferred scalar key; ``None`` falls back to computed MSE.

    Returns:
        Ranking value for the sample; lower means a better prediction.
    """
    if sort_key is not None:
        value = col_scalars.get(sort_key, sample.get(sort_key))
        if is_scalar_value(value):
            return cast(float, value)
    pred = np.asarray(sample["prediction"]).ravel()
    target = np.asarray(sample["target"]).ravel()
    return float(np.mean((pred - target) ** 2))
