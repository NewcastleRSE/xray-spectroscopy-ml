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
from torch import nn

from xanesnet.model_utils import get_conv_layers_output_size, ActivationSwitch


class CNN(nn.Module):
    def __init__(
        self,
        hidden_size,
        dropout,
        num_conv_layers,
        activation,
        out_channel,
        channel_mul,
        kernel_size,
        stride,
        x_data,
        y_data,
    ):
        super().__init__()

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

        activation_switch = ActivationSwitch()
        act_fn = activation_switch.fn(activation)

        # Convolutional layers output size
        out_conv_block_size = get_conv_layers_output_size(
            self.input_size,
            self.num_conv_layers,
            self.channel_mul,
            self.kernel_size,
            self.stride,
            self.out_channel,
            include_pooling=False,
        )

        # Convolutional Layers
        conv_layers = []
        dense_layers = []

        in_channel = 1

        for layer in range(num_conv_layers):
            # Collect layers
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

        # Fully Connected Layers

        dense_layer1 = nn.Sequential(
            nn.Linear(
                out_conv_block_size,
                self.hidden_size,
            ),
            act_fn(),
        )

        dense_layer2 = nn.Sequential(nn.Linear(self.hidden_size, self.output_size))

        dense_layers.append(dense_layer1)
        dense_layers.append(dense_layer2)

        self.dense_layers = nn.Sequential(*dense_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        out = self.dense_layers(x)

        return out
