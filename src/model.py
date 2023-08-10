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
from model_utils import ActivationSwitch
from model_utils import LossSwitch
from model_utils import OptimSwitch
from model_utils import get_conv_layers_output_size


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        dropout_rate,
        num_hidden_layers,
        hl_shrink,
        output_size,
        act_fn,
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.hl_shrink = hl_shrink
        self.act_fn = act_fn

        # check if the last hidden layer size is at least 1 and not less than the output size
        last_hidden_layer_size = int(
            self.hidden_size * self.hl_shrink ** (self.num_hidden_layers - 1)
        )

        if last_hidden_layer_size < 1:
            raise ValueError(
                "The size of the last hidden layer is less than 1, please adjust hyperparameters."
            )
        # if last_hidden_layer_size < self.output_size:
        #     raise ValueError(
        #         "The size of the last hidden layer is less than the output size, please adjust hyperparameters.")


        layers = []

        for i in range(self.num_hidden_layers-1):

            if i == 0:
                layer = nn.Sequential(
                            nn.Linear(self.input_size, self.hidden_size),
                            nn.Dropout(self.dropout_rate),
                            self.act_fn(),
                        )
            else:
                layer = nn.Sequential(
                            nn.Linear(int(self.hidden_size * self.hl_shrink ** (i - 1)), int(self.hidden_size * self.hl_shrink ** i)),
                            nn.Dropout(self.dropout_rate),
                            self.act_fn(),
                        )

            layers.append(layer)

        output_layer = nn.Sequential(
            nn.Linear(int(self.hidden_size * self.hl_shrink ** (self.num_hidden_layers - 2)), self.output_size)
        )

        layers.append(output_layer)

        self.dense_layers = nn.Sequential(*layers)

    def forward(self, x):
        # Feed forward through hidden layers
        out = self.dense_layers(x)

        # Feed forward through output layer
        # x = self.output_layer(x)

        return out


class CNN(nn.Module):
    def __init__(
        self,
        input_size,
        out_channel,
        channel_mul,
        hidden_layer,
        out_dim,
        dropout,
        kernel_size,
        stride,
        act_fn,
        n_cl,
    ):
        super().__init__()

        self.input_size = input_size
        self.out_channel = out_channel
        self.channel_mul = channel_mul
        self.hidden_layer = hidden_layer
        self.out_dim = out_dim
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.stride = stride
        self.act_fn = act_fn
        self.n_cl = n_cl

        # Convolutional layers output size
        out_conv_block_size = get_conv_layers_output_size(
            self.input_size,
            self.n_cl,
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

        for layer in range(n_cl):
            # Collect layers
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                ),
                nn.BatchNorm1d(num_features=out_channel),
                self.act_fn(),
                nn.Dropout(p=self.dropout),
            )

            conv_layers.append(conv_layer)

            # Update in/out channels
            in_channel = out_channel
            out_channel = out_channel * channel_mul

        self.conv_layers = nn.Sequential(*conv_layers)

        # Fully Connected Layers

        dense_layer1 = nn.Sequential(
            nn.Linear(out_conv_block_size, self.hidden_layer,), 
            self.act_fn()
            )

        dense_layer2 = nn.Sequential(
            nn.Linear(self.hidden_layer, self.out_dim)
            )

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


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        hl_size,
        out_dim,
        act_fn,
    ):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.hl_size = hl_size
        self.out_dim = out_dim
        self.act_fn = act_fn

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, hl_size),
            act_fn(),
            nn.Dropout(p=dropout),
            nn.Linear(hl_size, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        out = self.fc(x)

        return out


class AE_mlp(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        dropout_rate,
        num_hidden_layers,
        hl_shrink,
        output_size,
        act_fn,
    ):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.hl_shrink = hl_shrink
        self.act_fn = act_fn

        # check if the last hidden layer size is at least 1 and not less than the output size
        last_hidden_layer_size = int(
            self.hidden_size * self.hl_shrink ** (self.num_hidden_layers - 1)
        )

        if last_hidden_layer_size < 1:
            raise ValueError(
                "The size of the last hidden layer is less than 1, please adjust hyperparameters."
            )

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
                                nn.Linear(int(self.hidden_size * self.hl_shrink ** (i - 1)), int(self.hidden_size * self.hl_shrink ** i)),
                                act_fn(),
                            )
                dec_layer = nn.Sequential(
                                nn.Linear(int(self.hidden_size * self.hl_shrink ** i), int(self.hidden_size * self.hl_shrink ** (i - 1))),
                                act_fn(),
                            )

            enc_layers.append(enc_layer)
            dec_layers.insert(0,dec_layer)


        self.encoder_layers = nn.Sequential(*enc_layers)
        self.decoder_layers = nn.Sequential(*dec_layers)


        fc_layers = []

        fc_layer1 = nn.Sequential(
                        nn.Linear(int(self.hidden_size * self.hl_shrink ** (self.num_hidden_layers-1)), self.hidden_size),
                        act_fn(),
                        nn.Dropout(self.dropout_rate),
                    )
        fc_layer2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size)
        )

        fc_layers.append(fc_layer1)
        fc_layers.append(fc_layer2)

        self.dense_layers = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.encoder_layers(x)

        pred = self.dense_layers(out)

        recon = self.decoder_layers(out)

        return recon, pred

    def predict(self, x: torch.Tensor) -> torch.Tensor:

        out = self.encoder_layers(x)
        pred = self.dense_layers(out)

        return pred

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:

        out = self.encoder_layers(x)
        recon = self.decoder_layers(out)

        return recon


class AE_cnn(nn.Module):
    def __init__(
        self,
        input_size,
        out_channel,
        channel_mul,
        hidden_layer,
        out_dim,
        dropout,
        kernel_size,
        stride,
        act_fn,
        n_cl,
    ):
        super().__init__()

        self.input_size = input_size
        self.out_channel = out_channel
        self.channel_mul = channel_mul
        self.hidden_layer = hidden_layer
        self.out_dim = out_dim
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.stride = stride
        self.act_fn = act_fn
        self.n_cl = n_cl

        # Start collecting shape of convolutional layers for calculating padding
        all_conv_shapes = [input_size]

        # Starting shape
        conv_shape = self.input_size

        # ENCODER CONVOLUTIONAL LAYERS
        enc_layers = []

        enc_in_channel = 1
        enc_out_channel = self.out_channel

        for block in range(self.n_cl):

            # Create conv layer
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=enc_in_channel,
                    out_channels=enc_out_channel,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                ),
                self.act_fn(),
            )

            enc_layers.append(conv_layer)

            # Update in and out channels
            enc_in_channel = enc_out_channel
            enc_out_channel = enc_out_channel * self.channel_mul

            # Update output shape for conv layer
            conv_shape = int(((conv_shape - self.kernel_size) / self.stride) + 1)
            all_conv_shapes.append(conv_shape)

        self.encoder_layers = nn.Sequential(*enc_layers)

        # PREDICTOR DENSE LAYERS

        dense_in_shape = self.out_channel * self.channel_mul ** (self.n_cl-1) * all_conv_shapes[-1]

        dense_layers = []

        dense_layer1 = nn.Sequential(
            nn.Linear(dense_in_shape, self.hidden_layer),
            self.act_fn(),
            nn.Dropout(self.dropout)
        )

        dense_layer2 = nn.Sequential(
            nn.Linear(self.hidden_layer, self.out_dim),
        )

        dense_layers.append(dense_layer1)
        dense_layers.append(dense_layer2)

        self.dense_layers = nn.Sequential(*dense_layers)
        
        # DECODER TRANSPOSE CONVOLUTIONAL LAYERS

        dec_in_channel = self.out_channel * self.channel_mul ** (self.n_cl-1)
        dec_out_channel = self.out_channel * self.channel_mul  ** (self.n_cl-2)

        dec_layers = []

        for block in range(self.n_cl):

            tconv_out_shape = all_conv_shapes[self.n_cl - block - 1]
            tconv_in_shape = all_conv_shapes[self.n_cl - block]

            tconv_shape = int(((tconv_in_shape - 1) * self.stride) + self.kernel_size)

            # Calculate padding to input or output of transpose conv layer
            if tconv_shape != tconv_out_shape:
                if tconv_shape > tconv_out_shape:
                    # Pad input to transpose conv layer
                    padding = tconv_shape - tconv_out_shape
                    output_padding = 0
                elif tconv_shape < tconv_out_shape:
                    # Pad output of transpose conv layer
                    padding = 0
                    output_padding = tconv_out_shape - tconv_shape
            else:
                padding = 0
                output_padding = 0

            if block == self.n_cl - 1:
                dec_out_channel = 1

            tconv_layer = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=dec_in_channel,
                    out_channels=dec_out_channel,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    output_padding=output_padding,
                    padding=padding,
                ),
                self.act_fn(),
            )
            dec_layers.append(tconv_layer)

            # Update in/out channels
            if block < self.n_cl - 1:
                dec_in_channel = dec_out_channel
                dec_out_channel = dec_out_channel // self.channel_mul

        self.decoder_layers = nn.Sequential(*dec_layers)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)

        out = self.encoder_layers(x)

        pred = out.view(out.size(0), -1)
        pred = self.dense_layers(pred)

        recon = self.decoder_layers(out)
        recon = recon.squeeze(dim=1)

        return recon, pred

    def predict(self, x):
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)

        out = self.encoder_layers(x)

        pred = out.view(out.size(0), -1)
        pred = self.dense_layers(pred)

        return pred

    def reconstruct(self, x):
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)

        out = self.encoder_layers(x)

        recon = self.decoder_layers(out)
        recon = recon.squeeze(dim=1)

        return recon


class AEGANTrainer(nn.Module):
    def __init__(self, **params):
        super().__init__()

        self.input_size_a = params["dim_a"]
        self.input_size_b = params["dim_b"]
        self.hidden_size = params["hidden_size"]
        self.dropout = params["dropout"]

        self.n_hl_gen = params["n_hl_gen"]
        self.n_hl_shared = params["n_hl_shared"]
        self.n_hl_dis = params["n_hl_dis"]

        # Select activation function
        activation_switch = ActivationSwitch()
        self.activation = activation_switch.fn(params["activation"])

        # Select loss functions
        loss_gen_fn = params["loss_gen"]["loss_fn"]
        loss_dis_fn = params["loss_dis"]["loss_fn"]

        loss_gen_args = params["loss_gen"]["loss_args"]
        loss_dis_args = params["loss_dis"]["loss_args"]

        self.loss_fn_gen = LossSwitch().fn(loss_gen_fn, loss_gen_args)
        self.loss_fn_dis = LossSwitch().fn(loss_dis_fn, loss_dis_args)

        # Initialise the generative autoencoder networks
        self.gen_a = AEGen(
            input_size=self.input_size_a,
            num_hidden_layer=self.n_hl_gen,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
        )  # autoencoder for domain a
        self.gen_b = AEGen(
            input_size=self.input_size_b,
            num_hidden_layer=self.n_hl_gen,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
        )  # autoencoder for domain b

        # Initialise the shared autoencoder layers
        self.enc_shared = SharedLayer(
            num_hidden_layer=self.n_hl_shared,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.dec_shared = SharedLayer(
            num_hidden_layer=self.n_hl_shared,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
        )

        # Initialise the discriminator networks
        self.dis_a = Dis(
            input_size=self.input_size_a,
            num_hidden_layer=self.n_hl_dis,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
            loss_fn=self.loss_fn_dis,
        )  # discriminator for domain a
        self.dis_b = Dis(
            input_size=self.input_size_b,
            num_hidden_layer=self.n_hl_dis,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
            loss_fn=self.loss_fn_dis,
        )  # discriminator for domain b

        # Learning rate
        self.lr_gen = params["lr_gen"]
        self.lr_dis = params["lr_dis"]

        params_gen = [
            param for name, param in self.named_parameters() if "dis" not in name
        ]
        params_dis = [param for name, param in self.named_parameters() if "dis" in name]

        # Optim for generators and discriminators
        optim_fn_gen = OptimSwitch().fn(params["optim_fn_gen"])
        optim_fn_dis = OptimSwitch().fn(params["optim_fn_dis"])

        self.gen_opt = optim_fn_gen(params_gen, lr=self.lr_gen, weight_decay=1e-5)
        self.dis_opt = optim_fn_dis(params_dis, lr=self.lr_dis, weight_decay=1e-5)

    def get_optimizer(self):
        return self.gen_opt, self.dis_opt

    # Reconstruct structure from structure
    def reconstruct_structure(self, x):
        enc = self.gen_a.encode(x)
        shared_enc = self.enc_shared.forward(enc)
        shared_dec = self.dec_shared.forward(shared_enc)
        recon = self.gen_a.decode(shared_dec)
        return recon

    # Reconstruct spectrum from spectrum
    def reconstruct_spectrum(self, x):
        enc = self.gen_b.encode(x)
        shared_enc = self.enc_shared.forward(enc)
        shared_dec = self.dec_shared.forward(shared_enc)
        recon = self.gen_b.decode(shared_dec)
        return recon

    # Predict spectrum from structure
    def predict_spectrum(self, x):
        enc = self.gen_a.encode(x)
        shared_enc = self.enc_shared.forward(enc)
        shared_dec = self.dec_shared.forward(shared_enc)
        pred = self.gen_b.decode(shared_dec)
        return pred

    # Predict structure from spectrum
    def predict_structure(self, x):
        enc = self.gen_b.encode(x)
        shared_enc = self.enc_shared.forward(enc)
        shared_dec = self.dec_shared.forward(shared_enc)
        pred = self.gen_a.decode(shared_dec)
        return pred

    # Reconstruct and predict spectrum and structure from inputs
    def reconstruct_all_predict_all(self, x_a, x_b):
        enc_a = self.gen_a.encode(x_a)
        enc_b = self.gen_b.encode(x_b)

        shared_enc_a = self.enc_shared.forward(enc_a)
        shared_enc_b = self.enc_shared.forward(enc_b)

        shared_dec_a = self.dec_shared.forward(shared_enc_a)
        shared_dec_b = self.dec_shared.forward(shared_enc_b)

        x_ba = self.gen_a.decode(shared_dec_b)
        x_ab = self.gen_b.decode(shared_dec_a)

        x_a_recon = self.gen_a.decode(shared_dec_a)
        x_b_recon = self.gen_b.decode(shared_dec_b)

        return x_a_recon, x_b_recon, x_ba, x_ab

    def recon_criterion(self, pred, target):
        loss_fn = self.loss_fn_gen
        loss = loss_fn(pred, target)
        return loss

    def forward(self, x_a, x_b):
        enc_a = self.gen_a.encode(x_a)
        enc_b = self.gen_b.encode(x_b)

        shared_enc_a = self.enc_shared.forward(enc_a)
        shared_enc_b = self.enc_shared.forward(enc_b)

        shared_dec_a = self.dec_shared.forward(shared_enc_a)
        shared_dec_b = self.dec_shared.forward(shared_enc_b)

        x_ba = self.gen_a.decode(shared_dec_b)
        x_ab = self.gen_b.decode(shared_dec_a)
        return x_ab, x_ba

    def gen_update(self, x_a, x_b):
        self.gen_opt.zero_grad()
        # encode
        enc_a = self.gen_a.encode(x_a)
        enc_b = self.gen_b.encode(x_b)

        # encode shared layer
        shared_enc_a = self.enc_shared.forward(enc_a)
        shared_enc_b = self.enc_shared.forward(enc_b)
        # decode shared layer
        shared_dec_a = self.dec_shared.forward(shared_enc_a)
        shared_dec_b = self.dec_shared.forward(shared_enc_b)

        # decode (within domain)
        x_a_recon = self.gen_a.decode(shared_dec_a)
        x_b_recon = self.gen_b.decode(shared_dec_b)

        # decode (cross domain)
        x_ba = self.gen_a.decode(shared_dec_b)
        x_ab = self.gen_b.decode(shared_dec_a)

        # scale loss by mean-maximum value of input
        a_max = torch.max(x_a)
        b_max = torch.max(x_b)

        # reconstruction loss
        loss_recon_a = self.recon_criterion(x_a_recon, x_a) / a_max
        loss_recon_b = self.recon_criterion(x_b_recon, x_b) / b_max
        loss_pred_a = self.recon_criterion(x_ba, x_a) / a_max
        loss_pred_b = self.recon_criterion(x_ab, x_b) / b_max

        # total loss
        loss_total = loss_recon_a + loss_recon_b + loss_pred_a + loss_pred_b

        # loss_gen = loss_gen_total.item()

        loss_total.backward()
        self.gen_opt.step()

    def dis_update(self, x_a, x_b):
        # encode
        self.dis_opt.zero_grad()
        y_a = self.gen_a.encode(x_a)
        y_b = self.gen_b.encode(x_b)

        # encode shared layer
        shared_enc_a = self.enc_shared.forward(y_a)
        shared_enc_b = self.enc_shared.forward(y_b)
        # decode shared layer
        shared_dec_a = self.dec_shared.forward(shared_enc_a)
        shared_dec_b = self.dec_shared.forward(shared_enc_b)

        # decode (cross domain)
        x_ba = self.gen_a.decode(shared_dec_b)
        x_ab = self.gen_b.decode(shared_dec_a)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(shared_dec_a)
        x_b_recon = self.gen_b.decode(shared_dec_b)

        # Discriminator loss for real inputs
        loss_dis_adv_a = self.dis_a.calc_gen_loss(x_a)
        loss_dis_adv_b = self.dis_b.calc_gen_loss(x_b)

        loss_gen_adv_a = self.dis_a.calc_dis_loss(x_ba, x_a)
        loss_gen_adv_b = self.dis_b.calc_dis_loss(x_ab, x_b)

        loss_gen_recon_a = self.dis_a.calc_dis_loss(x_a_recon, x_a)
        loss_gen_recon_b = self.dis_b.calc_dis_loss(x_b_recon, x_b)

        loss_real = 0.5 * (loss_dis_adv_a + loss_dis_adv_b)
        loss_fake = 0.25 * (
            loss_gen_adv_a + loss_gen_recon_a + loss_gen_adv_b + loss_gen_recon_b
        )

        self.loss_dis_total = loss_real + loss_fake

        self.loss_dis_total.backward()
        self.dis_opt.step()


class SharedLayer(nn.Module):
    # Autoencoder architecture
    def __init__(self, num_hidden_layer, hidden_size, dropout, activation):
        super().__init__()
        self.num_hidden_layer = num_hidden_layer
        self.hidden_size = hidden_size
        self.activation = activation
        self.dropout = dropout

        dense_layers = []
        for layer in range(num_hidden_layer):
            dense_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                self.activation()
            )
            dense_layers.append(dense_layer)
        
        output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
        )

        dense_layers.append(output_layer)



        self.dense_layers = nn.Sequential(*dense_layers)

    def forward(self, x):
        out = self.dense_layers(x)
        return out


class AEGen(nn.Module):
    # Autoencoder architecture
    def __init__(self, input_size, num_hidden_layer, hidden_size, dropout, activation):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.num_hidden_layer = num_hidden_layer
        self.activation = activation
        self.hidden_size = hidden_size
        self.dropout = dropout

        # Encoder input layer
        encoder_input = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            self.activation(),
        )
        # Encoder mid layers
        encoder_mid_layers = []
        for layer in range(num_hidden_layer - 1):
            mid_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                self.activation()
            )
            encoder_mid_layers.append(mid_layer)
        encoder_mid_layers = nn.Sequential(*encoder_mid_layers)
        # Encoder output layer
        encoder_output = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
        )

        # Decoder input and mid layer
        decoder_mid_layers = []
        for layer in range(num_hidden_layer):
            mid_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                self.activation()
            )
            decoder_mid_layers.append(mid_layer)
        decoder_mid_layers = nn.Sequential(*decoder_mid_layers)
        # Decoder output layer
        decoder_output = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
            self.activation()
        )

        # Collect encoder layers
        encoder_layers = []
        encoder_layers.append(encoder_input)
        encoder_layers.append(encoder_mid_layers)
        encoder_layers.append(encoder_output)
        self.encoder_layers = nn.Sequential(*encoder_layers)

        # Collect decoder layers
        decoder_layers = []
        decoder_layers.append(decoder_mid_layers)
        decoder_layers.append(decoder_output)
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def encode(self, x):
        out = self.encoder_layers(x)
        return out

    def decode(self, x):
        out = self.decoder_layers(x)
        return out


class Dis(nn.Module):
    # Discriminator architecture
    def __init__(
        self, input_size, num_hidden_layer, hidden_size, dropout, activation, loss_fn
    ):
        super().__init__()
        self.input_size = input_size
        self.num_hidden_layer = num_hidden_layer
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.loss_fn = loss_fn

        mid_layers = []
        for layer in range(num_hidden_layer - 1):

            mid_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                self.activation()            
            )
            mid_layers.append(mid_layer)

        mid_layers = nn.Sequential(*mid_layers)

        input_layer = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            self.activation(),
        )
        
        output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 1), 
            nn.Sigmoid()
            )

        dense_layers = []

        dense_layers.append(input_layer)
        dense_layers.append(mid_layers)
        dense_layers.append(output_layer)

        self.dense_layers = nn.Sequential(*dense_layers)

    def forward(self, x):
        out = self.dense_layers(x)
        return out

    def calc_dis_loss(self, input_fake, input_real):
        # Calculate the loss to train D
        out0 = self.forward(input_fake)
        out1 = self.forward(input_real)
        loss = self.loss_fn(out0, out1)
        return loss

    def calc_gen_loss(self, input_fake):
        # Calculate the loss to train G
        out0 = self.forward(input_fake)
        ones = torch.ones((input_fake.size(0), 1))
        loss = self.loss_fn(out0, ones)
        return loss


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outs = [model(x) for model in self.models]
        out = torch.stack(outs).mean(dim=0)
        return out


class AutoencoderEnsemble(nn.Module):
    def __init__(self, models):
        super(AutoencoderEnsemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        reconstructions = []
        predictions = []
        for model in self.models:
            # Compute the reconstruction and prediction outputs for each sub-model
            reconstruction, prediction = model(x)
            reconstructions.append(reconstruction)
            predictions.append(prediction)

        # Stack the reconstruction and prediction outputs along a new dimension
        reconstructions = torch.stack(reconstructions).mean(dim=0)
        predictions = torch.stack(predictions).mean(dim=0)

        return reconstructions, predictions


class AEGANEnsemble(nn.Module):
    def __init__(self, models):
        super(AEGANEnsemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, y):
        if x is not None and y is not None:
            y_predictions = []
            x_predictions = []
            x_reconstructions = []
            y_reconstructions = []
            for model in self.models:
                # Compute the reconstruction and prediction outputs for each sub-model
                y_predictions.append(model.predict_spectrum(x))
                x_predictions.append(model.predict_structure(y))
                x_reconstructions.append(model.reconstruct_structure(x))
                y_reconstructions.append(model.reconstruct_spectrum(y))
            # Stack the reconstruction and prediction outputs along a new dimension
            x_reconstructions = torch.stack(x_reconstructions).mean(dim=0)
            y_reconstructions = torch.stack(y_reconstructions).mean(dim=0)
            x_predictions = torch.stack(x_predictions).mean(dim=0)
            y_predictions = torch.stack(y_predictions).mean(dim=0)

        elif x is not None and y is None:
            x_reconstructions = []
            y_predictions = []
            for model in self.models:
                x_reconstructions.append(model.reconstruct_structure(x))
                y_predictions.append(model.predict_spectrum(x))
            x_reconstructions = torch.stack(x_reconstructions).mean(dim=0)
            y_reconstructions = None
            x_predictions = None
            y_predictions = torch.stack(y_predictions).mean(dim=0)

        elif y is not None and x is None:
            y_reconstructions = []
            x_predictions = []
            for model in self.models:
                y_reconstructions.append(model.reconstruct_spectrum(y))
                x_predictions.append(model.predict_structure(y))
            x_reconstructions = None
            y_reconstructions = torch.stack(y_reconstructions).mean(dim=0)
            x_predictions = torch.stack(x_predictions).mean(dim=0)
            y_predictions = None

        return x_reconstructions, y_reconstructions, x_predictions, y_predictions
