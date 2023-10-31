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

from torch import nn

from xanesnet.model_utils import ActivationSwitch


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        dropout,
        num_hidden_layers,
        shrink_rate,
        activation,
        x_data,
        y_data,
    ):
        super(MLP, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.shrink_rate = shrink_rate
        self.activation = activation

        input_size = x_data.shape[1]
        output_size = y_data[0].size

        activation_switch = ActivationSwitch()
        act_fn = activation_switch.fn(activation)

        # check if the last hidden layer size is at least 1 and not less than the output size
        last_hidden_layer_size = int(
            self.hidden_size * self.shrink_rate ** (self.num_hidden_layers - 1)
        )

        if last_hidden_layer_size < 1:
            raise ValueError(
                "The size of the last hidden layer is less than 1, please adjust hyperparameters."
            )
        # if last_hidden_layer_size < self.output_size:
        #     raise ValueError(
        #         "The size of the last hidden layer is less than the output size, please adjust hyperparameters.")

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

        output_layer = nn.Sequential(
            nn.Linear(
                int(
                    self.hidden_size * self.shrink_rate ** (self.num_hidden_layers - 2)
                ),
                output_size,
            )
        )

        layers.append(output_layer)

        self.dense_layers = nn.Sequential(*layers)

    def forward(self, x):
        # Feed forward through hidden layers
        out = self.dense_layers(x)

        # Feed forward through output layer
        # x = self.output_layer(x)

        return out
