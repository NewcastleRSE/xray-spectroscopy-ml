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

"""Aggregator that projects collected vectors onto their principal components."""

import logging
from typing import Any

import numpy as np

from xanesnet.analysis.utils import as_float_vector
from xanesnet.serialization.config import Config
from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.utils.exceptions import ConfigError

from ..selectors import Selector
from .base import Aggregator, AggregatorResult
from .registry import AggregatorRegistry


@AggregatorRegistry.register("pca")
class PcaAggregator(Aggregator):
    """Project a collected vector key onto its leading principal components.

    The configured ``key`` is read from every collector record, stacked into an
    ``(M, D)`` matrix, optionally standardized, and projected onto the first
    ``n_components`` principal components via SVD. The aggregator is generic
    over the vector source, so it projects descriptor vectors as well as any
    future encoding collector output.

    Args:
        aggregator_type: Registered aggregator name from the analysis configuration.
        key: Collector key whose vector values are projected. Defaults to
            ``"descriptor"``.
        n_components: Number of principal components to keep.
        standardize: Whether to standardize the vectors before projection.
    """

    def __init__(
        self,
        aggregator_type: str,
        key: str,
        n_components: int,
        standardize: bool,
    ) -> None:
        """Initialize a PCA aggregator.

        Raises:
            ValueError: If ``n_components`` is below one.
        """
        super().__init__(aggregator_type)
        if n_components < 1:
            raise ValueError(f"n_components must be at least 1, got {n_components}")
        self.key = key
        self.n_components = n_components
        self.standardize = standardize

    def aggregate(self, selector: Selector, per_sample_values: JSONLStream | None, index: int) -> AggregatorResult:
        """Project the collected vectors onto their principal components.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
                Not used directly; vectors come from ``per_sample_values``.
            per_sample_values: Collector result stream aligned with ``selector``; must be provided
                so the configured ``key`` can be read.
            index: Zero-based aggregator index from the analysis configuration.

        Returns:
            ``projections`` with shape ``(M, n_components)``, the aligned
            ``sample_ids``, and ``n_components``. The data is empty when fewer
            than two vectors were collected.

        Raises:
            ConfigError: If no collector stream is available or the ``key`` is
                missing from a collector record.
        """
        if per_sample_values is None:
            raise ConfigError(f"PCA aggregator requires a collector stream to read key '{self.key}'.")

        sample_ids: list[str] = []
        vectors: list[np.ndarray] = []
        for record in per_sample_values:
            value = record.get(self.key)
            if value is None:
                raise ConfigError(
                    f"Key '{self.key}' is missing from a collector record. Configure a vector collector that produces it."
                )
            vectors.append(as_float_vector(value))
            sample_ids.append(str(record.get("sample_id", "")))

        if len(vectors) < 2:
            logging.info("      Need at least two vectors for PCA, skipping.")
            return AggregatorResult(aggregator_type=self.aggregator_type, aggregator_index=index, data={})

        matrix = np.stack(vectors)
        if self.standardize:
            matrix = self._standardize(matrix)
        projections = self._project(matrix)

        data: dict[str, Any] = {
            "sample_ids": sample_ids,
            "projections": projections,
            "n_components": int(projections.shape[1]),
        }
        return AggregatorResult(aggregator_type=self.aggregator_type, aggregator_index=index, data=data)

    @staticmethod
    def _standardize(features: np.ndarray) -> np.ndarray:
        """Center feature columns and scale them to unit variance.

        Args:
            features: Feature matrix with shape ``(M, D)``.

        Returns:
            Standardized feature matrix with shape ``(M, D)``.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = (features - features.mean(axis=0)) / features.std(axis=0)
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    def _project(self, matrix: np.ndarray) -> np.ndarray:
        """Project standardized features onto their first principal components.

        Args:
            matrix: Standardized feature matrix with shape ``(M, D)``.

        Returns:
            Projection with shape ``(M, n_components)``; missing components are
            zero-filled.
        """
        centered = matrix - matrix.mean(axis=0)
        if centered.shape[1] < 2:
            proj = centered[:, :1]
        else:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ vt[: min(self.n_components, vt.shape[0])].T
        if proj.shape[1] < self.n_components:
            proj = np.column_stack([proj, np.zeros((len(matrix), self.n_components - proj.shape[1]))])
        return proj

    @property
    def signature(self) -> Config:
        """Return the PCA aggregator signature."""
        signature = super().signature
        signature.update_with_dict(
            {"key": self.key, "n_components": self.n_components, "standardize": self.standardize}
        )
        return signature
