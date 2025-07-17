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

    def aegan_mlp(self, params):
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

        n_encoder1 = params["n_encoder1"]
        n_encoder2 = params["n_encoder2"]
        n_decoder1 = params["n_decoder1"]
        n_decoder2 = params["n_decoder2"]
        n_shared_encoder = params["n_shared_encoder"]
        n_shared_decoder = params["n_shared_decoder"]
        n_discrim1 = params["n_discrim1"]
        n_discrim2 = params["n_discrim2"]

        # Encoder 1
        if n_encoder1 > 0:
            count = 0
            for layer in self.model.gen_a.encoder_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_encoder1:
                        for param in layer.parameters():
                            param.requires_grad = False

        # Encoder 2
        if n_encoder2 > 0:
            count = 0
            for layer in self.model.gen_b.encoder_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_encoder2:
                        for param in layer.parameters():
                            param.requires_grad = False

        # Decoder 1
        if n_decoder1 > 0:
            count = 0
            for layer in self.model.gen_a.decoder_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_decoder1:
                        for param in layer.parameters():
                            param.requires_grad = False

        # Decoder 2
        if n_decoder2 > 0:
            count = 0
            for layer in self.model.gen_b.decoder_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_decoder2:
                        for param in layer.parameters():
                            param.requires_grad = False

        # Shared Encoder Layers
        if n_shared_encoder > 0:
            count = 0
            for layer in self.model.enc_shared.dense_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_shared_encoder:
                        for param in layer.parameters():
                            param.requires_grad = False

        if n_shared_decoder > 0:
            count = 0
            for layer in self.model.dec_shared.dense_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_shared_decoder:
                        for param in layer.parameters():
                            param.requires_grad = False

        # Discriminator 1
        if n_discrim1 > 0:
            count = 0
            for layer in self.model.dis_a.dense_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_discrim1:
                        for param in layer.parameters():
                            param.requires_grad = False

        # Discriminator 2
        if n_discrim1 > 0:
            count = 0
            for layer in self.model.dis_b.dense_layers.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_discrim2:
                        for param in layer.parameters():
                            param.requires_grad = False

        return self.model

    def gnn(self, params):
        """
        freeze_params = {
            n_gnn : int
            n_dense : int
        }
        """
        print(">>> FREEZING FOR GNNS")
        n_gnn = params["n_gnn"]
        n_dense = params["n_dense"]

        block_count = 0
        module_count = 0
        modules_per_block = 4

        total_layers = len(self.model.layers)
        while module_count < total_layers and block_count < n_gnn:
            current_block_size = (
                modules_per_block
                if (module_count + modules_per_block) < total_layers
                else 1
            )

            for i in range(current_block_size):
                layer = self.model.layers[module_count + i]
                for param in layer.parameters():
                    param.requires_grad = False

            block_count += 1
            module_count += current_block_size

        if n_dense > 0:
            count = 0
            for layer in self.model.head.children():
                if isinstance(layer, nn.Sequential):
                    count += 1
                    if count <= n_dense:
                        for param in layer.parameters():
                            param.requires_grad = False

        return self.model
