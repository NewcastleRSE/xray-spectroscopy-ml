from torch import nn

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
    def fn(self, loss_fn):
        default = nn.MSELoss()
        return getattr(self, f"loss_function_{loss_fn.lower()}", lambda: default)()

    def loss_function_mse(self):
        return nn.MSELoss()

    def loss_function_bce(self):
        return nn.BCEWithLogitsLoss()

    def loss_function_emd(self):
        return EMDLoss()

    def loss_function_cosine(self):
        return nn.CosineEmbeddingLoss()

    def loss_function_l1(self):
        return nn.L1Loss()


# Earth mover distance as loss function
class EMDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        loss = torch.mean(
            torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)),
            dim=-1,
        ).sum()
        return loss


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def model_mode_error(model, mode, model_mode, xyz_shape, xanes_shape):

    from utils import unique_path
    from pathlib import Path

    for child in model.modules():
        if type(child).__name__ == "Linear":
            output_size = child.weight.shape[0]
            print(output_size)

    if mode == "predict_xyz":
        input_data = xanes_shape
        output_data = xyz_shape
    elif mode == "predict_xanes":
        input_data = xyz_shape
        output_data = xanes_shape

    if model_mode == "mlp" or model_mode == "cnn" or model_mode == "ae_cnn":
        assert (
            output_size == output_data
        ), "the model was not train for this, please swap your predict mode"
    if model_mode == "ae_mlp":
        assert (
            output_size == input_data
        ), "the model was not train for this, please swap your predict mode"

    predict_dir = unique_path(Path("."), "predictions")
    predict_dir.mkdir()

    return predict_dir


def json_check(inp):
    assert isinstance(
        inp["hyperparams"]["loss"], str
    ), "wrong type for loss param in json"
    assert isinstance(
        inp["hyperparams"]["activation"], str
    ), "wrong type for activation param in json"
