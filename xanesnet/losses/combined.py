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

"""Weighted combination of multiple XANESNET loss functions."""

import torch

from .base import Loss


class CombinedLoss(Loss):
    """Weighted combination of multiple loss functions.

    Computes a normalized weighted sum of component losses.

    Args:
        losses: Ordered list of :class:`Loss` modules to combine.
        weights: Raw (un-normalized) weight for each component loss.
            Must be positive and have the same length as ``losses``.

    Raises:
        ValueError: If ``losses`` is empty or ``losses`` and ``weights`` have
            different lengths.
    """

    def __init__(self, losses: list[Loss], weights: list[float]) -> None:
        """Initialize ``CombinedLoss``."""
        super().__init__(loss_type="combined")

        if len(losses) == 0:
            raise ValueError("CombinedLoss requires at least one component loss.")
        if len(losses) != len(weights):
            raise ValueError(f"Length mismatch: {len(losses)} losses but {len(weights)} weights.")

        self.losses = torch.nn.ModuleList(losses)

        total = sum(weights)
        normalized = [w / total for w in weights]
        self.register_buffer("weights", torch.tensor(normalized, dtype=torch.float32))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the normalized weighted sum of component losses.

        Args:
            preds: Model output predictions ``(B, N)``.
            targets: Ground-truth target values ``(B, N)``.

        Returns:
            Scalar loss tensor.
        """
        stacked = torch.stack([loss(preds, targets) for loss in self.losses])
        return (self.weights * stacked).sum()
