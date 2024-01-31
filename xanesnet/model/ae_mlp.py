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
from xanesnet.utils_model import ActivationSwitch


class AE_MLP(Model):
    """
    A class for constructing an AE-MLP (Autoencoder Multilayer Perceptron Network).
    The model has three main components: encoder, decoder and dense layers.
    The reconstruction of input data is performed as a forward pass through the encoder
    and decoder. The prediction is performed as a forward pass through the encoder and
    dense layers. Hyperparameter specification is the same as for the MLP model type.
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

        self.ae_flag = 1
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.shrink_rate = shrink_rate
        self.activation = activation

        self.input_size = x_data.shape[1]
        self.output_size = y_data[0].size

        # Instantiate ActivationSwitch for dynamic activation selection
        activation_switch = ActivationSwitch()
        act_fn = activation_switch.fn(activation)

        # Check if the last hidden layer size is at least 1 and not less than the output size
        last_hidden_layer_size = int(
            self.hidden_size * self.shrink_rate ** (self.num_hidden_layers - 1)
        )
        if last_hidden_layer_size < 1:
            raise ValueError(
                "The size of the last hidden layer is less than 1, please adjust hyperparameters."
            )

        # Construct encoder and decoder layers with shrink rate
        enc_layers = []
        dec_layers = []
        for i in range(self.num_hidden_layers):
            if i == 0:
                enc_layer = nn.Sequential(
                    nn.Linear(self.input_size, self.hidden_size),
                    act_fn(),
                )
                dec_layer = nn.Sequential(
                    nn.Linear(self.hidden_size, self.input_size),
                    act_fn(),
                )
            else:
                enc_layer = nn.Sequential(
                    nn.Linear(
                        int(self.hidden_size * self.shrink_rate ** (i - 1)),
                        int(self.hidden_size * self.shrink_rate**i),
                    ),
                    act_fn(),
                )
                dec_layer = nn.Sequential(
                    nn.Linear(
                        int(self.hidden_size * self.shrink_rate**i),
                        int(self.hidden_size * self.shrink_rate ** (i - 1)),
                    ),
                    act_fn(),
                )

            enc_layers.append(enc_layer)
            dec_layers.insert(0, dec_layer)

        self.encoder_layers = nn.Sequential(*enc_layers)
        self.decoder_layers = nn.Sequential(*dec_layers)

        # Construct dense layers
        fc_layers = []
        fc_layer1 = nn.Sequential(
            nn.Linear(
                int(
                    self.hidden_size * self.shrink_rate ** (self.num_hidden_layers - 1)
                ),
                self.hidden_size,
            ),
            act_fn(),
            nn.Dropout(self.dropout),
        )

        # Construct the dense layers as a sequential module by
        # combining all the individual layers created earlier
        fc_layer2 = nn.Sequential(nn.Linear(self.hidden_size, self.output_size))
        fc_layers.append(fc_layer1)
        fc_layers.append(fc_layer2)

        self.dense_layers = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Feed forward through dense layers
        out = self.encoder_layers(x)
        pred = self.dense_layers(out)
        recon = self.decoder_layers(out)

        return recon, pred

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # Generate predictions based on the encoded representation of the input.
        out = self.encoder_layers(x)
        pred = self.dense_layers(out)

        return pred

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct the input data based on the encoded representation.
        out = self.encoder_layers(x)
        recon = self.decoder_layers(out)

        return recon
