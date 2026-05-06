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

"""Registry instance for optimizer classes."""

import torch.optim as optim

from xanesnet.utils.registry import Registry

OptimizerRegistry: Registry[type[optim.Optimizer]] = Registry("Optimizer", normalize_key=str.lower)


# register optimizers
OptimizerRegistry.register("adam")(optim.Adam)
OptimizerRegistry.register("sgd")(optim.SGD)
OptimizerRegistry.register("rmsprop")(optim.RMSprop)
OptimizerRegistry.register("adamw")(optim.AdamW)
OptimizerRegistry.register("adagrad")(optim.Adagrad)
