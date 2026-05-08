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

"""Registry instances for tensor initialization callables."""

from typing import Protocol

import torch
from torch import nn

from xanesnet.utils.registry import Registry


class InitFn(Protocol):
    """Protocol for in-place tensor initialization callables."""

    def __call__(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Initialize ``tensor`` in place and return it.

        Args:
            tensor: Tensor to initialize.
            *args: Positional arguments forwarded to the initializer.
            **kwargs: Keyword arguments forwarded to the initializer.

        Returns:
            The initialized tensor.
        """
        ...


def _noop(tensor: torch.Tensor, **_) -> torch.Tensor:
    """Return ``tensor`` unchanged."""
    return tensor


WeightInitRegistry: Registry[InitFn] = Registry("Weight initializer", normalize_key=str.lower)
BiasInitRegistry: Registry[InitFn] = Registry("Bias initializer", normalize_key=str.lower)


# register weights inits
WeightInitRegistry.register("uniform")(nn.init.uniform_)
WeightInitRegistry.register("normal")(nn.init.normal_)
WeightInitRegistry.register("xavier_uniform")(nn.init.xavier_uniform_)
WeightInitRegistry.register("xavier_normal")(nn.init.xavier_normal_)
WeightInitRegistry.register("kaiming_uniform")(nn.init.kaiming_uniform_)
WeightInitRegistry.register("kaiming_normal")(nn.init.kaiming_normal_)
WeightInitRegistry.register("default")(_noop)


# register bias inits
BiasInitRegistry.register("zeros")(nn.init.zeros_)
BiasInitRegistry.register("ones")(nn.init.ones_)
