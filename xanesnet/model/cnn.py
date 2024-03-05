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
import numpy as np
import torch
from torch import nn

from xanesnet.model.base_model import Model
from xanesnet.utils_model import get_conv_layers_output_size, ActivationSwitch


class CNN(Model):
    """
    A class for constructing a customisable CNN (Convolutional Neural Network) model.
    The model consists of a set of convolutional layers and two dense layers.
    Each convolutional layer comprises a 1D convolution, batch normalisation,
    an activation function, a dropout layer.

    The Number of output channels in the convolutional layers (out_channels) is increased
    multiplicatively at each layer, and it's controlled by a multiplication factor (channel_mul).
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        num_conv_layers: int,
        activation: str,
        out_channel: int,
        channel_mul: int,
        kernel_size: int,
        stride: int,
        x_data: np.ndarray,
        y_data: np.ndarray,
    ):
        """
        Args:
            hidden_size (integer): Size of hidden layers
                in the dense (fully connected) layers.
            dropout (float): Dropout rate applied to
                convolutional layers for regularization.
            num_conv_layers (integer): Number of convolutional layers
            activation (string): Name of activation function
                for convolutional and dense layers.
            out_channel (integer): Number of output channels
                for the initial convolutional layer.
            channel_mul (integer): Channel multiplication factor
                for increasing output channels in subsequent
                convolutional layers.
            kernel_size (integer): Size of the convolutional kernel (filter).
            stride (integer): Stride of the convolution operation.
            x_data (NumPy array): Input data for the network
            y_data (Numpy array): Output data for the network
        """
        super().__init__()

        self.nn_flag = 1
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.num_conv_layers = num_conv_layers
        self.out_channel = out_channel
        self.channel_mul = channel_mul
        self.kernel_size = kernel_size
        self.stride = stride

        self.input_size = x_data.shape[1]
        self.output_size = y_data[0].size

        # Instantiate ActivationSwitch for dynamic activation selection
        activation_switch = ActivationSwitch()
        act_fn = activation_switch.fn(activation)

        # Get the output size of convolutional layers
        out_conv_block_size = get_conv_layers_output_size(
            self.input_size,
            self.num_conv_layers,
            self.channel_mul,
            self.kernel_size,
            self.stride,
            self.out_channel,
            include_pooling=False,
        )

        # Construct convolutional Layers
        conv_layers = []
        in_channel = 1
        for layer in range(num_conv_layers):
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                ),
                nn.BatchNorm1d(num_features=out_channel),
                act_fn(),
                nn.Dropout(p=self.dropout),
            )

            conv_layers.append(conv_layer)

            # Update in/out channels
            in_channel = out_channel
            out_channel = out_channel * channel_mul

        self.conv_layers = nn.Sequential(*conv_layers)

        # Construct dense layers
        dense_layers = []
        dense_layer1 = nn.Sequential(
            nn.Linear(
                out_conv_block_size,
                self.hidden_size,
            ),
            act_fn(),
        )

        # Construct the dense layers as a sequential module by
        # combining all the individual layers created earlier
        dense_layer2 = nn.Sequential(nn.Linear(self.hidden_size, self.output_size))
        dense_layers.append(dense_layer1)
        dense_layers.append(dense_layer2)

        self.dense_layers = nn.Sequential(*dense_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through convolutional layers
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.conv_layers(x)

        # Forward pass through dense layers
        x = x.view(x.size(0), -1)
        out = self.dense_layers(x)

        return out
