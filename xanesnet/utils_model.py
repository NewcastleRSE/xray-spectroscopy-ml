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

import random
import warnings
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler

from torch import optim
from torch import nn

# Suppress non-significant warning for shap and WCCLoss function
warnings.filterwarnings("ignore")


# Select activation function from hyperparams inputs
class ActivationSwitch:
    def fn(self, activation):
        default = nn.PReLU()
        return getattr(
            self, f"activation_function_{activation.lower()}", lambda: default
        )()

    def activation_function_relu(self):
        return nn.ReLU

    def activation_function_prelu(self):
        return nn.PReLU

    def activation_function_tanh(self):
        return nn.Tanh

    def activation_function_sigmoid(self):
        return nn.Sigmoid

    def activation_function_elu(self):
        return nn.ELU

    def activation_function_leakyrelu(self):
        return nn.LeakyReLU

    def activation_function_selu(self):
        return nn.SELU


# Select loss function from hyperparams inputs
class LossSwitch:
    def fn(self, loss_fn, *args):
        default = nn.MSELoss()
        return getattr(self, f"loss_function_{loss_fn.lower()}", lambda: default)(*args)

    def loss_function_mse(self, *args):
        return nn.MSELoss(*args)

    def loss_function_bce(self, *args):
        return nn.BCEWithLogitsLoss()

    def loss_function_emd(self, *args):
        return EMDLoss()

    def loss_function_cosine(self, *args):
        return CosineSimilarityLoss()

    def loss_function_l1(self, *args):
        return nn.L1Loss(*args)

    def loss_function_wcc(self, *args):
        return WCCLoss(*args)


class WeightInitSwitch:
    def fn(self, weight_init_fn):
        default = nn.init.xavier_uniform_
        return getattr(
            self, f"weight_init_function_{weight_init_fn.lower()}", lambda: default
        )()

    # uniform
    def weight_init_function_uniform(self):
        return nn.init.uniform_

    # normal
    def weight_init_function_normal(self):
        return nn.init.normal_

    # xavier_uniform
    def weight_init_function_xavier_uniform(self):
        return nn.init.xavier_uniform_

    # xavier_normal
    def weight_init_function_xavier_normal_(self):
        return nn.init.xavier_normal_

    # kaiming_uniform
    def weight_init_function_kaiming_uniform(self):
        return nn.init.kaiming_uniform_

    # kaiming_normal
    def weight_init_function_kaiming_normal(self):
        return nn.init.kaiming_normal_

    # zeros
    def weight_init_function_zeros(self):
        return nn.init.zeros_

    # ones
    def weight_init_function_ones(self):
        return nn.init.ones_


def weight_bias_init(m, kernel_init_fn, bias_init_fn):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
        kernel_init_fn(m.weight)
        bias_init_fn(m.bias)


class LRScheduler:
    """
    Initialise the learning rate scheduler
    """

    def __init__(self, optimizer, scheduler_type, params=None):
        self.optimizer = optimizer
        scheduler_type = scheduler_type.lower()

        if scheduler_type == "step":
            self.scheduler = lr_scheduler.StepLR(optimizer, **params)
        elif scheduler_type == "multistep":
            self.scheduler = lr_scheduler.MultiStepLR(optimizer, **params)
        elif scheduler_type == "exponential":
            self.scheduler = lr_scheduler.ExponentialLR(optimizer, **params)
        elif scheduler_type == "linear":
            self.scheduler = lr_scheduler.LinearLR(optimizer, **params)
        elif scheduler_type == "constant":
            self.scheduler = lr_scheduler.ConstantLR(optimizer, **params)
        else:
            raise ValueError(f"Invalid scheduler type: {scheduler_type}")

    def step(self):
        self.scheduler.step()


class OptimSwitch:
    def fn(self, opt_fn):
        default = optim.Adam
        return getattr(self, f"optim_function_{opt_fn.lower()}", lambda: default)()

    # Adam
    def optim_function_adam(self):
        return optim.Adam

    # Stochastic Gradient Descent
    def optim_function_sgd(self):
        return optim.SGD

    # RMSprop
    def optim_function_rmsprop(self):
        return optim.RMSprop


class EMDLoss(nn.Module):
    """
    Computes the Earth Mover or Wasserstein distance
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        loss = torch.mean(
            torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)),
            dim=-1,
        ).sum()
        return loss


class CosineSimilarityLoss(nn.Module):
    """
    Implements Cosine Similarity as loss function
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        loss = torch.mean(nn.CosineSimilarity()(y_pred, y_true))
        return loss


class CosineSimilarityLoss(nn.Module):
    """
    Implements Cosine Similarity as loss function
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        loss = torch.mean(nn.CosineSimilarity()(y_pred, y_true))
        return loss


class WCCLoss(nn.Module):
    """
    Computes the weighted cross-correlation loss between y_pred and y_true based on the
    method proposed in [1].
    Args:
        gaussianHWHM: Scalar value for full-width-at-half-maximum of Gaussian weight function.
    Reference:
    [1] Källman, E., Delcey, M.G., Guo, M., Lindh, R. and Lundberg, M., 2020.
        "Quantifying similarity for spectra with a large number of overlapping transitions: Examples
        from soft X-ray spectroscopy." Chemical Physics, 535, p.110786.
    """

    def __init__(self, gaussianHWHM):
        super().__init__()
        if gaussianHWHM is None:
            print(
                ">> WCC Loss Function Gaussian HWHM parameter not set in input yaml file. Setting equal to 10"
            )
            gaussianHWHM = 10
        self.gaussianHWHM = gaussianHWHM

    def forward(self, y_true, y_pred):
        n_features = y_true.shape[1]
        n_samples = y_true.shape[0]

        width2 = (self.gaussianHWHM / np.sqrt(2.0 * np.log(2))) * 2

        corr = nn.functional.conv1d(
            y_true.unsqueeze(0), y_pred.unsqueeze(1), padding="same", groups=n_samples
        )
        corr1 = nn.functional.conv1d(
            y_true.unsqueeze(0), y_true.unsqueeze(1), padding="same", groups=n_samples
        )
        corr2 = nn.functional.conv1d(
            y_pred.unsqueeze(0), y_pred.unsqueeze(1), padding="same", groups=n_samples
        )

        corr = corr.squeeze(0)
        corr1 = corr1.squeeze(0)
        corr2 = corr2.squeeze(0)

        dx = torch.ones(n_samples)
        de = ((n_features / 2 - torch.arange(0, n_features))[:, None] * dx[None, :]).T
        weight = np.exp(-de * de / (2 * width2))

        norm = torch.sum(corr * weight, 1)
        norm1 = torch.sum(corr1 * weight, 1)
        norm2 = torch.sum(corr2 * weight, 1)
        similarity = torch.clip(norm / torch.sqrt(norm1 * norm2), 0, 1)

        loss = 1 - torch.mean(similarity)
        return loss


def model_mode_error(
    model: object,
    mode: object,
    model_mode: object,
    xyz_shape: object,
    xanes_shape: object,
) -> object:
    for child in model.modules():
        if type(child).__name__ == "Linear":
            output_size = child.weight.shape[0]

    if mode == "predict_xyz":
        input_data = xanes_shape
        output_data = xyz_shape
    elif mode == "predict_xanes":
        input_data = xyz_shape
        output_data = xanes_shape

    if mode == "predict_xyz" or mode == "predict_xanes":
        if model_mode == "mlp" or model_mode == "cnn" or model_mode == "ae_cnn":
            assert (
                output_size == output_data
            ), "the model was not train for this, please swap your predict mode"
        if model_mode == "ae_mlp":
            assert (
                output_size == input_data
            ), "the model was not train for this, please swap your predict mode"

    parent_model_dir, predict_dir = make_output_dir()
    return parent_model_dir, predict_dir


def make_output_dir():
    from pathlib import Path

    from xanesnet.utils import unique_path

    parent_model_dir = "outputs/"
    Path(parent_model_dir).mkdir(parents=True, exist_ok=True)

    predict_dir = unique_path(Path(parent_model_dir), "predictions")
    predict_dir.mkdir()

    return predict_dir


def json_check(inp):
    # assert isinstance(
    #     inp["hyperparams"]["loss"], str
    # ), "wrong type for loss param in json"
    assert isinstance(
        inp["hyperparams"]["activation"], str
    ), "wrong type for activation param in json"


# def json_cnn_check(inp, model):
#     assert isinstance(
#         inp["hyperparams"]["loss"], str
#     ), "wrong type for loss param in json"
#     assert isinstance(
#         inp["hyperparams"]["activation"], str
#     ), "wrong type for activation param in json"


def montecarlo_dropout(model, input_data, n_mc):
    model.train()

    prob_output = []

    input_data = torch.from_numpy(input_data)
    input_data = input_data.float()

    for t in range(n_mc):
        output = model(input_data)
        prob_output.append(output)

    prob_mean = torch.mean(torch.stack(prob_output), dim=0)
    prob_var = torch.std(torch.stack(prob_output), dim=0)

    return prob_mean, prob_var


def montecarlo_dropout_ae(model, input_data, n_mc):
    model.train()

    prob_output = []
    prob_recon = []

    input_data = torch.from_numpy(input_data)
    input_data = input_data.float()

    for t in range(n_mc):
        recon, output = model(input_data)
        prob_output.append(output)
        prob_recon.append(recon)

    mean_output = torch.mean(torch.stack(prob_output), dim=0)
    var_output = torch.std(torch.stack(prob_output), dim=0)

    mean_recon = torch.mean(torch.stack(prob_recon), dim=0)
    var_recon = torch.std(torch.stack(prob_recon), dim=0)

    return mean_output, var_output, mean_recon, var_recon


def loss_reg_fn(model, loss_reg_type, device):
    """Computes L1 or L2 norm of model parameters for use in regularisation of loss function

    Args:
        model
        loss_reg_type (_str_): Regularisation type. L1 or
        device
    """
    l_reg = torch.tensor(0.0).to(device)
    if loss_reg_type.lower() == "l1":
        for param in model.parameters():
            l_reg += torch.norm(param, p=1)

    elif loss_reg_type.lower() == "l2":
        for param in model.parameters():
            l_reg += torch.norm(param)

    return l_reg


def get_conv_layers_output_size(
    input_size,
    num_conv_layers,
    channel_mul,
    kernel_size,
    stride,
    out_channel,
    include_pooling=False,
):
    """
    Calculates size of flattened output from N 1D Convolutional layers
    For use with CNN and AE_CNN models
    """
    in_channel = 1

    for block in range(num_conv_layers):
        out_length = int(np.floor((input_size - kernel_size) / stride + 1))
        in_channel = int(out_channel)
        out_channel = out_channel * (channel_mul)
        input_size = out_length

    if include_pooling:
        out_length = int(np.floor((input_size - kernel_size) / stride + 1))

    return in_channel * out_length
