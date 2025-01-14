from xanesnet.utils_model import (
    ActivationSwitch,
    LossSwitch,
    WeightInitSwitch,
    OptimSwitch,
)
from torch import nn, optim


class TestModelUtils:
    def test_activation_switch(self):
        switch = ActivationSwitch()
        # Check default
        act_fn = switch.fn("NonExistent")
        assert act_fn == nn.PReLU
        # Check specified
        act_fn = switch.fn("relu")
        assert act_fn == nn.ReLU
        act_fn = switch.fn("prelu")
        assert act_fn == nn.PReLU
        act_fn = switch.fn("leakyrelu")
        assert act_fn == nn.LeakyReLU

    def test_loss_switch(self):
        switch = LossSwitch()
        # Check default
        loss_fn = switch.fn("NonExistent")
        assert isinstance(loss_fn, nn.MSELoss)
        # Check specified
        loss_fn = switch.fn("mse")
        assert isinstance(loss_fn, nn.MSELoss)
        loss_fn = switch.fn("bce")
        assert isinstance(loss_fn, nn.BCEWithLogitsLoss)

    def test_weight_switch(self):
        switch = WeightInitSwitch()
        # Check default
        weight_fn = switch.fn("NonExistent")
        assert weight_fn == nn.init.xavier_uniform_
        # Check specified
        weight_fn = switch.fn("uniform")
        assert weight_fn == nn.init.uniform_
        weight_fn = switch.fn("normal")
        assert weight_fn == nn.init.normal_

    def test_optimizer_switch(self):
        switch = OptimSwitch()
        # Check default
        opt_fn = switch.fn("NonExistent")
        assert opt_fn == optim.Adam
        # Check specified
        opt_fn = switch.fn("sgd")
        assert opt_fn == optim.SGD
        opt_fn = switch.fn("adam")
        assert opt_fn == optim.Adam
