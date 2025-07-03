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


class Freeze:
    def __init__(self, path):
        # Load model from file
        path = path + "/model.pt"
        self.model = torch.load(path, map_location=torch.device("cpu"))

    def get_fn(self, name, params):
        return getattr(self, name)(params)

    def mlp(self, params):
        """
        params = {
            # Number of layers to freeze
            n_dense: int
        }
        """
        n_dense = params["n_dense"]

        if n_dense > 0:
            count = 0
            for layer in self.model.dense_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_dense:
                        for param in layer.parameters():
                            param.requires_grad = False

        return self.model

    def cnn(self, params):
        """
        freeze_params = {
            # Number of layers to freeze
            n_conv : int
            n_dense : int
        }
        """
        n_conv = params["n_conv"]
        n_dense = params["n_dense"]

        if n_conv > 0:
            count = 0
            for layer in self.model.conv_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_conv:
                        for param in layer.parameters():
                            param.requires_grad = False

        if n_dense > 0:
            count = 0
            for layer in self.model.dense_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_dense:
                        for param in layer.parameters():
                            param.requires_grad = False

        return self.model

    def lstm(self, params):
        """
        freeze_params = {
            # Number of layers to freeze
            n_lstm: int
            n_dense: int
        }
        """
        n_lstm = params["n_lstm"]
        n_dense = params["n_dense"]

        if n_lstm > 0:
            LSTM_BLOCK_SIZE = 8
            count = 0
            for name, param in self.model.lstm.named_parameters():
                if count // LSTM_BLOCK_SIZE + 1 <= n_lstm:
                    param.requires_grad = False
                count += 1

        if n_dense > 0:
            print(">>> Note current implementation freezes all model.fc layers.")
            for name, param in self.model.fc.named_parameters():
                param.requires_grad = False

        return self.model

    def ae_mlp(self, params):
        """
        freeze_params = {
            # Number of layers to freeze
            n_encoder : int
            n_decoder : int
            n_dense : int
        }
        """

        n_encoder = params["n_encoder"]
        n_decoder = params["n_decoder"]
        n_dense = params["n_dense"]

        if n_encoder > 0:
            count = 0
            for layer in self.model.encoder_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_encoder:
                        for param in layer.parameters():
                            param.requires_grad = False

        if n_decoder > 0:
            count = 0
            for layer in self.model.decoder_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_decoder:
                        for param in layer.parameters():
                            param.requires_grad = False

        if n_dense > 0:
            count = 0
            for layer in self.model.dense_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_dense:
                        for param in layer.parameters():
                            param.requires_grad = False

        return self.model

    def ae_cnn(self, params):
        """
        freeze_params = {
            # Number of layers to freeze
            n_encoder : int
            n_decoder : int
            n_dense : int
            n_conv: int
        }
        """

        n_encoder = params["n_encoder"]
        n_decoder = params["n_decoder"]
        n_conv = params["n_conv"]
        n_dense = params["n_dense"]

        if n_encoder > 0:
            count = 0
            for layer in self.model.encoder_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_encoder:
                        for param in layer.parameters():
                            param.requires_grad = False

        if n_decoder > 0:
            count = 0
            for layer in self.model.decoder_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_decoder:
                        for param in layer.parameters():
                            param.requires_grad = False

        if n_dense > 0:
            count = 0
            for layer in self.model.dense_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_dense:
                        for param in layer.parameters():
                            param.requires_grad = False

        if n_conv > 0:
            count = 0
            for layer in self.model.conv_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_conv:
                        for param in layer.parameters():
                            param.requires_grad = False

        return self.model


def freeze_layers_aegan_mlp(model, freeze_params):
    """
    freeze_params = {
        # Number of layers to freeze
        n_encoder1 : int
        n_encoder2 : int
        n_decoder1 : int
        n_decoder2 : int
        n_shared_encoder : int
        n_shared_decoder : int
        n_discrim1 : int
        n_discrim2 : int
    }
    """

    n_encoder1 = freeze_params["n_encoder1"]
    n_encoder2 = freeze_params["n_encoder2"]
    n_decoder1 = freeze_params["n_decoder1"]
    n_decoder2 = freeze_params["n_decoder2"]
    n_shared_encoder = freeze_params["n_shared_encoder"]
    n_shared_decoder = freeze_params["n_shared_decoder"]
    n_discrim1 = freeze_params["n_discrim1"]
    n_discrim2 = freeze_params["n_discrim2"]

    # Encoder 1
    if n_encoder1 > 0:
        count = 0
        for layer in model.gen_a.encoder_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_encoder1:
                    for param in layer.parameters():
                        param.requires_grad = False

    # Encoder 2
    if n_encoder2 > 0:
        count = 0
        for layer in model.gen_b.encoder_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_encoder2:
                    for param in layer.parameters():
                        param.requires_grad = False

    # Decoder 1
    if n_decoder1 > 0:
        count = 0
        for layer in model.gen_a.decoder_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_decoder1:
                    for param in layer.parameters():
                        param.requires_grad = False

    # Decoder 2
    if n_decoder2 > 0:
        count = 0
        for layer in model.gen_b.decoder_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_decoder2:
                    for param in layer.parameters():
                        param.requires_grad = False

    # Shared Encoder Layers
    if n_shared_encoder > 0:
        count = 0
        for layer in model.shared_encoder.dense_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_shared_encoder:
                    for param in layer.parameters():
                        param.requires_grad = False

    if n_shared_decoder > 0:
        count = 0
        for layer in model.shared_decoder.dense_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_shared_decoder:
                    for param in layer.parameters():
                        param.requires_grad = False

    # Discriminator 1
    if n_discrim1 > 0:
        count = 0
        for layer in model.dis_a.dense_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_discrim1:
                    for param in layer.parameters():
                        param.requires_grad = False

    # Discriminator 2
    if n_discrim1 > 0:
        count = 0
        for layer in model.dis_b.dense_layers.children():
            if isinstance(layer, nn.Sequential):
                count += 1
                if count <= n_discrim2:
                    for param in layer.parameters():
                        param.requires_grad = False

    return model
