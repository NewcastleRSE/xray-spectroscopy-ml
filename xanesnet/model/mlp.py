"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
import numpy as np
from torch import nn

from xanesnet.model.base_model import Model
from xanesnet.utils_model import ActivationSwitch


class MLP(Model):
    """
    A class for constructing a customisable MLP (Multi-Layer Perceptron) model.
    The model consists of a set of hidden layers. All the layers expect the final
    layer, are comprised of a linear layer, a dropout layer, and an activation function.
    The final (output) layer is a linear layer.

    The size of each hidden linear layer is determined by the initial dimension
    (hidden_size) and a reduction factor (shrink_rate) that reduces the layer
    dimension multiplicatively.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        num_hidden_layers: int,
        shrink_rate: float,
        activation: str,
        x_data: np.ndarray,
        y_data: np.ndarray,
    ):
        """
        Args:
            hidden_size (integer): Size of the initial hidden layer.
            dropout (float): If none-zero, add dropout layer on the outputs
                of each hidden layer with dropout probability equal to dropout.
            num_hidden_layers (integer): Number of hidden layers
                in the network.
            shrink_rate (float): Rate to reduce the hidden layer
                size multiplicatively.
            activation (string): Name of activation function applied
                to the hidden layers.
            x_data (NumPy array): Input data for the network
            y_data (Numpy array): Output data for the network
        """

        super().__init__()

        self.nn_flag = 1
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.shrink_rate = shrink_rate
        self.activation = activation

        input_size = x_data.shape[1]
        output_size = y_data[0].size

        # Instantiate ActivationSwitch for dynamic activation selection
        activation_switch = ActivationSwitch()
        act_fn = activation_switch.fn(activation)

        # Check if the last hidden layer size is at least 1 and not less than the output size
        last_size = int(
            self.hidden_size * self.shrink_rate ** (self.num_hidden_layers - 1)
        )
        if last_size < 1:
            raise ValueError(
                "The size of the last hidden layer is less than 1, please adjust hyperparameters."
            )

        # Construct each hidden layer with shrink rate
        layers = []
        for i in range(self.num_hidden_layers - 1):
            if i == 0:
                layer = nn.Sequential(
                    nn.Linear(input_size, self.hidden_size),
                    nn.Dropout(self.dropout),
                    act_fn(),
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(
                        int(self.hidden_size * self.shrink_rate ** (i - 1)),
                        int(self.hidden_size * self.shrink_rate**i),
                    ),
                    nn.Dropout(self.dropout),
                    act_fn(),
                )

            layers.append(layer)

        # Construct the output layer
        output_layer = nn.Sequential(
            nn.Linear(
                int(
                    self.hidden_size * self.shrink_rate ** (self.num_hidden_layers - 2)
                ),
                output_size,
            )
        )
        layers.append(output_layer)

        # Construct the dense layers as a sequential module by
        # combining all the individual layers created earlier
        self.dense_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feed forward through dense layers
        out = self.dense_layers(x)

        return out
