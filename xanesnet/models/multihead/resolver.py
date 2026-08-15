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

"""Automatic configuration resolvers for multi-head models."""

from typing import Any

import torch

from xanesnet.serialization.auto_config.registries import ModelAutoResolver
from xanesnet.serialization.config import ConfigRaw


@ModelAutoResolver.register("mh_mlp")
def resolve_mh_mlp(inputs: dict[str, Any], target: torch.Tensor) -> ConfigRaw:
    """Resolve MH-MLP input and output dimensions.

    Args:
        inputs: Prepared model input dictionary.
        target: Prepared target tensor.

    Returns:
        Mapping with MH-MLP automatic fields ``in_size`` and ``out_size``.
    """
    print(inputs)
    return {
        "in_size": int(inputs["x"].shape[-1]),
        "out_size": int(target.shape[-1]),
    }


@ModelAutoResolver.register("mh_cnn")
def resolve_mh_cnn(inputs: dict[str, Any], target: torch.Tensor) -> ConfigRaw:
    """Resolve MH-CNN input and output dimensions.

    Args:
        inputs: Prepared model input dictionary.
        target: Prepared target tensor.

    Returns:
        Mapping with MH-CNN automatic fields ``in_size`` and ``out_size``.
    """
    return {
        "in_size": int(inputs["x"].shape[-1]),
        "out_size": int(target.shape[-1]),
    }

