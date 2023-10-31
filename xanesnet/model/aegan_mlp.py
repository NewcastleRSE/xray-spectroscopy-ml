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

from xanesnet.model_utils import (
    ActivationSwitch,
    LossSwitch,
    OptimSwitch,
)


class AEGAN_MLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        dropout,
        n_hl_gen,
        n_hl_shared,
        n_hl_dis,
        activation,
        lr_gen,
        lr_dis,
        optim_fn_gen,
        optim_fn_dis,
        loss_gen,
        loss_dis,
        x_data,
        y_data,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout

        self.n_hl_gen = n_hl_gen
        self.n_hl_shared = n_hl_shared
        self.n_hl_dis = n_hl_dis

        # Select activation function
        activation_switch = ActivationSwitch()
        self.activation = activation_switch.fn(activation)

        # Select loss functions
        loss_gen_fn = loss_gen["loss_fn"]
        loss_dis_fn = loss_dis["loss_fn"]

        loss_gen_args = loss_gen["loss_args"]
        loss_dis_args = loss_dis["loss_args"]

        self.loss_fn_gen = LossSwitch().fn(loss_gen_fn, loss_gen_args)
        self.loss_fn_dis = LossSwitch().fn(loss_dis_fn, loss_dis_args)

        self.input_size_a = x_data.shape[1]
        self.input_size_b = y_data.shape[1]

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
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis

        params_gen = [
            param for name, param in self.named_parameters() if "dis" not in name
        ]
        params_dis = [param for name, param in self.named_parameters() if "dis" in name]

        # Optim for generators and discriminators
        optim_fn_gen = OptimSwitch().fn(optim_fn_gen)
        optim_fn_dis = OptimSwitch().fn(optim_fn_dis)

        self.gen_opt = optim_fn_gen(params_gen, lr=self.lr_gen, weight_decay=1e-5)
        self.dis_opt = optim_fn_dis(params_dis, lr=self.lr_dis, weight_decay=1e-5)

    def get_optimizer(self):
        return self.gen_opt, self.dis_opt

    # Reconstruct descriptor from descriptor
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

    # Predict spectrum from descriptor
    def predict_spectrum(self, x):
        enc = self.gen_a.encode(x)
        shared_enc = self.enc_shared.forward(enc)
        shared_dec = self.dec_shared.forward(shared_enc)
        pred = self.gen_b.decode(shared_dec)
        return pred

    # Predict descriptor from spectrum
    def predict_structure(self, x):
        enc = self.gen_b.encode(x)
        shared_enc = self.enc_shared.forward(enc)
        shared_dec = self.dec_shared.forward(shared_enc)
        pred = self.gen_a.decode(shared_dec)
        return pred

    # Reconstruct and predict spectrum and descriptor from inputs
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

    def dis_update(self, x_a, x_b, device):
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
        loss_dis_adv_a = self.dis_a.calc_gen_loss(x_a, device)
        loss_dis_adv_b = self.dis_b.calc_gen_loss(x_b, device)

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
                self.activation(),
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
                self.activation(),
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
                self.activation(),
            )
            decoder_mid_layers.append(mid_layer)
        decoder_mid_layers = nn.Sequential(*decoder_mid_layers)
        # Decoder output layer
        decoder_output = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size), self.activation()
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
                self.activation(),
            )
            mid_layers.append(mid_layer)

        mid_layers = nn.Sequential(*mid_layers)

        input_layer = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            self.activation(),
        )

        output_layer = nn.Sequential(nn.Linear(self.hidden_size, 1), nn.Sigmoid())

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

    def calc_gen_loss(self, input_fake, device):
        # Calculate the loss to train G
        out0 = self.forward(input_fake)
        ones = torch.ones((input_fake.size(0), 1), device=device)
        loss = self.loss_fn(out0, ones)
        return loss
