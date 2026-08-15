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

import torch
from torch import nn

from xanesnet.components import ActivationRegistry

class MLPHead(nn.Module):
    """             
    A class for constructing a customisable MLP (Multi-Layer Perceptron) model that
    is used as one head of a larger multi-headed MLP network. The model consists of
    a set of hidden layers. All the layers expect the final layer, are comprised of
    a linear layer, a dropout layer, and an activation function. The final (output)
    layer is a linear layer.

    The size of each hidden linear layer is determined by the input dimension
    (input_size) and the output dimension (output_size) that reduces the layer
    dimension multiplicatively.
    """ 

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        dropout: float,
        num_hidden_layers: int,
        shrink_rate: float,
        activation: str,
    ):
        """
        Args:
            in_size (integer): Size of input data
            out_size (integer): Size of output data
            hidden_size (integer): Size of the initial hidden layer.
            dropout (float): Dropout probability for hidden layers.
            num_hidden_layers (int): Number of hidden layers, excluding input and output layers
            shrink_rate (float): Rate to reduce the hidden layer size multiplicatively.
            activation (str): Name of activation function for hidden layers.
        """
        
        super().__init__()

        layers: list[nn.Module] = []

        # Initialise input and hidden layers
        current_size = in_size
        for i in range(num_hidden_layers):
            next_size = int(hidden_size * (shrink_rate**i))
            if next_size < 1:
                raise ValueError(f"Hidden layer {i + 1} size is less than 1. Adjust hidden_size or shrink_rate.")


            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.BatchNorm1d(next_size))
            layers.append(nn.Dropout(dropout))
            layers.append(ActivationRegistry.create(activation))
            current_size = next_size

        # Initialise output layer    
        layers.append(nn.Sequential(nn.Linear(current_size, out_size), nn.Softplus()))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)