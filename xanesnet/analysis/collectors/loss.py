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

"""Collector for scalar and energy-resolved loss values between prediction and target spectra."""

from typing import Any

import numpy as np
import torch

from xanesnet.losses import LossRegistry
from xanesnet.serialization.prediction_readers import PredictionSample

from .base import Collector
from .registry import CollectorRegistry


@CollectorRegistry.register("loss")
class LossCollector(Collector):
    """Compute a configured loss for one sample, optionally energy-resolved.

    Args:
        collector_type: Registered collector name from the analysis configuration.
        loss_type: Registered loss name from ``xanesnet.losses``.
        energy_resolved: If ``True``, return the per-channel loss vector under
            the key ``<loss_type>_energy`` instead of the scalar loss value.
        **loss_kwargs: Keyword arguments forwarded to the configured loss class.
    """

    def __init__(
        self,
        collector_type: str,
        loss_type: str,
        energy_resolved: bool = False,
        **loss_kwargs: Any,
    ) -> None:
        """Initialize the configured loss function."""
        super().__init__(collector_type)

        self.loss_type = loss_type
        self.energy_resolved = energy_resolved
        self.loss_fn = LossRegistry.create(loss_type, loss_type=loss_type, **loss_kwargs)

    def process(self, sample: PredictionSample) -> dict[str, Any]:
        """Compute the configured loss for one prediction sample.

        Args:
            sample: Prediction sample with ``prediction`` and ``target`` spectra. One-dimensional
                spectra are treated as ``(N,)`` and batched to ``(1, N)`` before loss evaluation.

        Returns:
            Mapping from ``loss_type`` to the scalar loss value, or from
            ``<loss_type>_energy`` to the per-channel loss vector when
            ``energy_resolved`` is enabled.
        """
        pred_torch = torch.as_tensor(sample["prediction"], dtype=torch.float32)
        target_torch = torch.as_tensor(sample["target"], dtype=torch.float32)

        # Losses operate on batched spectral tensors with shape (B, N).
        if pred_torch.ndim == 1:
            pred_torch = pred_torch.unsqueeze(0)
        if target_torch.ndim == 1:
            target_torch = target_torch.unsqueeze(0)

        if self.energy_resolved:
            loss_map = self.loss_fn(pred_torch, target_torch, reduction="none")
            if loss_map.ndim != 2 or loss_map.shape[0] != 1:
                raise ValueError(f"Expected loss map with shape (1, N), got {tuple(loss_map.shape)}")
            return {f"{self.loss_type}_energy": loss_map.squeeze(0).detach().cpu().numpy()}

        loss_value = self.loss_fn(pred_torch, target_torch)
        return {self.loss_type: float(loss_value.item())}
