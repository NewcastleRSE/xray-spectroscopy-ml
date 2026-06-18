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

"""Public API for XANESNET loss functions."""

from .base import Loss
from .bcewithlogits import BCEWithLogitsLoss
from .combined import CombinedLoss
from .emd import EMDLoss
from .l1 import L1Loss
from .mse import MSELoss
from .msssim import MultiScale_SSIM
from .registry import LossRegistry
from .specplus import SpectralLossPlus
from .wcc import WCCLoss

__all__ = [
    "Loss",
    "LossRegistry",
    "BCEWithLogitsLoss",
    "CombinedLoss",
    "EMDLoss",
    "L1Loss",
    "MSELoss",
    "MultiScale_SSIM",
    "SpectralLossPlus",
    "WCCLoss",
]
