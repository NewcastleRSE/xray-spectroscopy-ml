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

"""Public API for E3EE layer modules."""

from .atom_encoder import EquivariantAtomEncoder
from .basic import (
    MLP,
    CosineCutoff,
    EnergyRBFEmbedding,
    GaussianRBF,
    IrrepNorm,
    RadialMLP,
)
from .branch_attention import EnergyConditionedAtomAttention
from .branch_convolution import (
    EnergyConditionedAtomConvolution,
    EnergyConditionedEquivariantAtomConvolution,
)
from .branch_eq_attention import EnergyConditionedEquivariantAtomAttention
from .branch_equivariant import (
    EnergyConditionedEquivariantTargetSiteHead,
    EnergyIrrepModulation,
)
from .branch_fusion import GatedBranchFusion
from .branch_path import PairElementEnergyScattering, TargetSitePathAggregator
from .branch_target_site import EnergyConditionedTargetSiteBranch
from .interactions import EquivariantInteractionBlock

__all__ = [
    "TargetSitePathAggregator",
    "CosineCutoff",
    "EnergyConditionedTargetSiteBranch",
    "EnergyConditionedAtomAttention",
    "EnergyConditionedAtomConvolution",
    "EnergyConditionedEquivariantTargetSiteHead",
    "EnergyConditionedEquivariantAtomAttention",
    "EnergyConditionedEquivariantAtomConvolution",
    "EnergyIrrepModulation",
    "EnergyRBFEmbedding",
    "EquivariantAtomEncoder",
    "EquivariantInteractionBlock",
    "GaussianRBF",
    "GatedBranchFusion",
    "IrrepNorm",
    "MLP",
    "PairElementEnergyScattering",
    "RadialMLP",
]
