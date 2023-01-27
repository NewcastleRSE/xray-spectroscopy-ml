import torch
from torch import nn
import numpy as np
import warnings

# Suppress non-significant warning for WCCLoss function
warnings.filterwarnings("ignore")


# Select activation function from hyperparams inputs
class ActivationSwitch:
    def fn(self, activation):
        default = nn.PReLU()
        return getattr(
            self, f"activation_function_{activation.lower()}", lambda: default
        )()

    def activation_function_relu(self):
        return nn.ReLU()

    def activation_function_prelu(self):
        return nn.PReLU()

    def activation_function_tanh(self):
        return nn.Tanh()

    def activation_function_sigmoid(self):
        return nn.Sigmoid()

    def activation_function_elu(self):
        return nn.ELU()

    def activation_function_leakyrelu(self):
        return nn.LeakyReLU()

    def activation_function_selu(self):
        return nn.SELU()


# Select loss function from hyperparams inputs
class LossSwitch:
    def fn(self, loss_fn, *args):
        default = nn.MSELoss()
        return getattr(self, f"loss_function_{loss_fn.lower()}", lambda: default)(*args)

    def loss_function_mse(self, *args):
        return nn.MSELoss(*args)

    def loss_function_bce(self, *args):
        return nn.BCEWithLogitsLoss(*args)

    def loss_function_emd(self, *args):
        return EMDLoss(*args)

    def loss_function_cosine(self, *args):
        return nn.CosineEmbeddingLoss(*args)

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


def model_mode_error(model, mode, model_mode, xyz_shape, xanes_shape):

    for child in model.modules():
        if type(child).__name__ == "Linear":
            output_size = child.weight.shape[0]
            # print(output_size)

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

    parent_model_dir, predict_dir = make_dir()
    return parent_model_dir, predict_dir


def make_dir():
    from utils import unique_path
    from pathlib import Path

    parent_model_dir = "outputs/"
    Path(parent_model_dir).mkdir(parents=True, exist_ok=True)

    predict_dir = unique_path(Path(parent_model_dir), "predictions")
    predict_dir.mkdir()

    return parent_model_dir, predict_dir


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


def montecarlo_dropout(model, input_data, output_shape):

    model.train()
    T = 10

    prob_output = np.zeros(output_shape)
    print(prob_output.shape)
    for t in range(T):
        output = model(input_data)
        prob_output = prob_output + output.cpu().detach().numpy()

    prob_pred = prob_output / T

    return prob_pred
