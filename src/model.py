import torch
from torch import nn
from model_utils import ActivationSwitch
from model_utils import LossSwitch


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, hl_size, out_dim, act_fn):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hl_size = hl_size
        self.out_dim = out_dim
        self.act_fn = act_fn

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            self.act_fn(),
            nn.Dropout(p=self.dropout),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hl_size),
            self.act_fn(),
            nn.Dropout(p=self.dropout),
        )

        self.fc3 = nn.Sequential(nn.Linear(self.hl_size, self.out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)

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

        self.conv1_shape = int(((self.input_size - self.kernel_size) / self.stride) + 1)
        self.conv2_shape = int(
            ((self.conv1_shape - self.kernel_size) / self.stride) + 1
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.out_channel,
                kernel_size=self.kernel_size,
                stride=self.stride,
            ),
            nn.BatchNorm1d(num_features=self.out_channel),
            self.act_fn(),
            nn.Dropout(p=self.dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.out_channel,
                out_channels=self.out_channel * self.channel_mul,
                kernel_size=self.kernel_size,
                stride=self.stride,
            ),
            nn.BatchNorm1d(num_features=self.out_channel * self.channel_mul),
            self.act_fn(),
            nn.Dropout(p=self.dropout),
        )

        self.dense_layer1 = nn.Sequential(
            nn.Linear(
                self.conv2_shape * (self.out_channel * self.channel_mul),
                self.hidden_layer,
            ),
            self.act_fn(),
        )

        self.dense_layer2 = nn.Sequential(nn.Linear(self.hidden_layer, self.out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dense_layer1(x)
        out = self.dense_layer2(x)

        return out


class AE_mlp(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, hl_size, out_dim, act_fn):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hl_size = hl_size
        self.out_dim = out_dim
        self.act_fn = act_fn

        self.encoder_hidden_1 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            self.act_fn(),
        )

        self.encoder_hidden_2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hl_size),
            self.act_fn(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.hl_size, self.hl_size),
            self.act_fn(),
            nn.Dropout(self.dropout),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.hl_size, self.out_dim),
        )

        self.decoder_hidden_1 = nn.Sequential(
            nn.Linear(self.hl_size, self.hidden_size),
            self.act_fn(),
        )

        self.decoder_hidden_2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.input_size),
            # self.act_fn(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_hidden_1(x)
        latent_space = self.encoder_hidden_2(x)

        in_fc = self.fc1(latent_space)
        pred_y = self.fc2(in_fc)

        out = self.decoder_hidden_1(latent_space)
        recon = self.decoder_hidden_2(out)

        return recon, pred_y

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_hidden_1(x)
        latent_space = self.encoder_hidden_2(x)

        in_fc = self.fc1(latent_space)
        pred_y = self.fc2(in_fc)

        return pred_y

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_hidden_1(x)
        latent_space = self.encoder_hidden_2(x)

        out = self.decoder_hidden_1(latent_space)
        recon = self.decoder_hidden_2(out)

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

        # checking the sign for encoder for input_channel for linear
        self.conv1_shape = int(((self.input_size - self.kernel_size) / self.stride) + 1)
        self.conv2_shape = int(
            ((self.conv1_shape - self.kernel_size) / self.stride) + 1
        )
        self.conv3_shape = int(
            ((self.conv2_shape - self.kernel_size) / self.stride) + 1
        )

        # checking the size for decoder to assign padding
        self.convt1_shape = int(
            ((self.conv3_shape - 1) * self.stride) + self.kernel_size
        )
        if self.convt1_shape != self.conv2_shape:
            if self.convt1_shape > self.conv2_shape:
                self.p1 = self.convt1_shape - self.conv2_shape
                self.op1 = 0
            elif self.convt1_shape < self.conv2_shape:
                self.op1 = self.conv2_shape - self.convt1_shape
                self.p1 = 0
        else:
            self.p1 = 0
            self.op1 = 0

        self.convt2_shape = int(
            ((self.conv2_shape - 1) * self.stride) + self.kernel_size
        )
        if self.convt2_shape != self.conv1_shape:
            if self.convt2_shape > self.conv1_shape:
                self.p2 = self.convt2_shape - self.conv1_shape
                self.op2 = 0
            elif self.convt2_shape < self.conv1_shape:
                self.op2 = self.conv1_shape - self.convt2_shape
                self.p2 = 0
        else:
            self.p2 = 0
            self.op2 = 0

        self.convt3_shape = int(
            ((self.conv1_shape - 1) * self.stride) + self.kernel_size
        )
        if self.convt3_shape != self.input_size:
            if self.convt3_shape > self.input_size:
                self.p3 = self.convt3_shape - self.input_size
                self.op3 = 0
            elif self.convt3_shape < self.input_size:
                self.op3 = self.input_size - self.convt3_shape
                self.p3 = 0
        else:
            self.p3 = 0
            self.op3 = 0

        self.encoder_layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.out_channel,
                kernel_size=self.kernel_size,
                stride=self.stride,
            ),
            self.act_fn(),
        )

        self.encoder_layer2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.out_channel,
                out_channels=int(self.out_channel * self.channel_mul),
                kernel_size=self.kernel_size,
                stride=self.stride,
            ),
            self.act_fn(),
        )

        self.encoder_output = nn.Sequential(
            nn.Conv1d(
                in_channels=int(self.out_channel * self.channel_mul),
                out_channels=int(self.out_channel * self.channel_mul * 2),
                kernel_size=self.kernel_size,
                stride=self.stride,
            ),
            self.act_fn(),
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(
                self.conv3_shape * int(self.out_channel * self.channel_mul * 2),
                self.hidden_layer,
            ),
            self.act_fn(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layer, self.out_dim),
        )

        self.decoder_layer1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=int(self.out_channel * self.channel_mul * 2),
                out_channels=int(self.out_channel * self.channel_mul),
                kernel_size=self.kernel_size,
                stride=self.stride,
                output_padding=self.op1,
                padding=self.p1,
            ),
            self.act_fn(),
        )

        self.decoder_layer2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=int(self.out_channel * self.channel_mul),
                out_channels=self.out_channel,
                kernel_size=self.kernel_size,
                stride=self.stride,
                output_padding=self.op2,
                padding=self.p2,
            ),
            self.act_fn(),
        )

        self.decoder_output = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.out_channel,
                out_channels=1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                output_padding=self.op3,
                padding=self.p3,
            ),
            self.act_fn(),
        )

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)

        out = self.encoder_layer1(x)
        out = self.encoder_layer2(out)
        out = self.encoder_output(out)

        pred = out.view(out.size(0), -1)
        pred = self.dense_layers(pred)

        recon = self.decoder_layer1(out)
        recon = self.decoder_layer2(recon)
        recon = self.decoder_output(recon)
        recon = recon.squeeze(dim=1)

        return recon, pred

    def predict(self, x):
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)

        out = self.encoder_layer1(x)
        out = self.encoder_layer2(out)
        out = self.encoder_output(out)

        pred = out.view(out.size(0), -1)
        pred = self.dense_layers(pred)

        return pred

    def reconstruct(self, x):
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)

        out = self.encoder_layer1(x)
        out = self.encoder_layer2(out)
        out = self.encoder_output(out)

        recon = self.decoder_layer1(out)
        recon = self.decoder_layer2(recon)
        recon = self.decoder_output(recon)
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
        self.gen_opt = torch.optim.Adam(params_gen, lr=self.lr_gen, weight_decay=1e-5)
        self.dis_opt = torch.optim.Adam(params_dis, lr=self.lr_dis, weight_decay=1e-5)

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
        linear_layers = []
        for layer in range(num_hidden_layer):
            linear_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            linear_layers.append(nn.BatchNorm1d(self.hidden_size))
            linear_layers.append(self.activation())
        self.linear_layers = nn.Sequential(*linear_layers)
        self.out_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
        )

    def forward(self, x):
        x = self.linear_layers(x)
        x = self.out_layer(x)
        return x


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

        enc_layer = []
        for layer in range(num_hidden_layer - 1):
            enc_layer.append(nn.Linear(self.hidden_size, self.hidden_size))
            enc_layer.append(nn.BatchNorm1d(self.hidden_size))
            enc_layer.append(self.activation())
        self.enc_layer = nn.Sequential(*enc_layer)

        dec_layer = []
        for layer in range(num_hidden_layer):
            dec_layer.append(nn.Linear(self.hidden_size, self.hidden_size))
            dec_layer.append(nn.BatchNorm1d(self.hidden_size))
            dec_layer.append(self.activation())
        self.dec_layer = nn.Sequential(*dec_layer)
        self.enc_input = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            self.activation(),
        )
        self.enc_output = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
        )
        self.dec_output = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size), self.activation()
        )

    def encode(self, x):
        x = self.enc_input(x)
        x = self.enc_layer(x)
        out = self.enc_output(x)
        return x

    def decode(self, x):
        x = self.dec_layer(x)
        out = self.dec_output(x)
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
        linear_layers = []
        for layer in range(num_hidden_layer - 1):
            linear_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            linear_layers.append(nn.BatchNorm1d(self.hidden_size))
            linear_layers.append(self.activation())
        self.linear_layers = nn.Sequential(*linear_layers)
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            self.activation(),
        )
        self.output_layer = nn.Sequential(nn.Linear(self.hidden_size, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.input_layer(x)
        x = self.linear_layers(x)
        out = self.output_layer(x)
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
