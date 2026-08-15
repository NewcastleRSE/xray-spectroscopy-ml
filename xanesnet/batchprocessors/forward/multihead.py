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

"""Batch processor for multi-head models with descriptor datasets."""

from typing import Any

import numpy as np
import torch

from xanesnet.datasets import MultiheadData

from ..registry import BatchProcessorRegistry
from .base import ForwardBatchProcessor


@BatchProcessorRegistry.register(("multihead", "mh_mlp"))
@BatchProcessorRegistry.register(("multihead_mp", "mh_mlp"))
class MultiheadBatchProcessor(ForwardBatchProcessor):
    """Batch processor for ``MultiheadData`` + multi-head models.

    Multi-head models run a shared trunk and return predictions from every
    head when ``active_head_idx`` is omitted. This processor selects the
    per-sample head output indicated by ``head_idx`` on the batch before the
    loss is computed.
    """

    def input_preparation(self, batch: MultiheadData) -> dict[str, torch.Tensor]:
        """Prepare multi-head model inputs from a multihead batch.

        Args:
            batch: Collated multihead batch.

        Returns:
            Dict with ``"x"`` containing the descriptor tensor. ``(batch_size, n_features)``.
        """
        return {"x": batch.x}  # type: ignore[dict-item]

    def target_preparation(self, batch: MultiheadData) -> torch.Tensor:
        """Prepare raw spectral targets from a multihead batch.

        Args:
            batch: Collated multihead batch.

        Returns:
            Raw spectral intensity tensor. ``(batch_size, n_energies)``.
        """
        return batch.y  # type: ignore[return-value]

    def element_preparation(self, batch: MultiheadData) -> torch.Tensor | None:
        """Extract absorber atomic numbers from a multihead batch.

        Args:
            batch: Collated multihead batch.

        Returns:
            Per-sample absorber atomic numbers ``(batch_size,)``, or
            ``None`` if the dataset was built without element information.
        """
        return batch.element

    def sample_id_extraction(self, batch: MultiheadData) -> np.ndarray:
        """Extract file names from a multihead batch.

        Args:
            batch: Collated multihead batch.

        Returns:
            Array of file name strings. ``(batch_size,)``.
        """
        return np.array(batch.sample_id, dtype=str)

    def prediction_preparation(self, batch: MultiheadData, predictions: torch.Tensor) -> torch.Tensor:
        """Select the active head prediction for each sample in the batch.

        Args:
            batch: Collated batch carrying ``head_idx``.
            predictions: Raw model output with all heads stacked.
                ``(num_heads, batch_size, n_features)``.

        Returns:
            Per-sample predictions from the selected head.
            ``(batch_size, n_features)``.
        """
        predictions = predictions.permute(1, 0, 2)
        # predictions shape = (Batch, Heads, Features)
        # Select the specific head's prediction for each sample in the batch
        predictions = predictions[torch.arange(batch.x.shape[0]), batch.head_idx]
        # predictions shape = (Batch, Features)
        return predictions