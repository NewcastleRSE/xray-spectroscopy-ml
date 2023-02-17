"""
XANESNET
Copyright (C) 2021  Conor D. Rankine

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

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import tensorflow as tf

from numpy.random import RandomState
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import register_keras_serializable

###############################################################################
################################### CLASSES ###################################
###############################################################################


@register_keras_serializable()
class Dropout_(Dropout):
    """
    A Dropout layer subclass of the default Keras Dropout layer that exposes
    the `training` argument of the `call()` function to the Keras
    `model.Sequential()` API; the `training` argument takes either 1 (True) or
    0 (False) to indicate to the layer whether to operate in training
    (i.e. dropout-active) or testing (i.e. dropout-inactive) mode, otherwise
    the operating mode is automatically determined by Keras if the `training`
    argument is left to default to None.

    If `always_on` is True, dropout will be active during training and
    prediction (useful, e.g., in Monte-Carlo experiments); if `always_on` is
    False, Keras will determine dropout activity automatically.

    The default Keras Dropout layer randomly sets input units to zero with a
    frequency of `rate` at each step during training time to help limit the
    propensity for overfitting; inputs that are not set to zero are scaled up
    by 1(1 - `rate`) such that the sum over all inputs is not changed.
    """

    def __init__(
        self,
        rate: float,
        always_on: bool = False,
        noise_shape: int = None,
        seed: int = None,
        **kwargs,
    ):
        """
        Args:
            rate (float): Fraction (0.0 -> 1.0) of the input units to drop.
            always_on (bool): Toggles whether the Dropout layer is always
                called in dropout-active mode; `training` is passed to
                `call()` with a value of 1 (True) if True, else None (and
                automatically determined by Keras as usual) if False.
                Defaults to False.
            noise_shape (int): 1D integer tensor representing the shape of
                the binary dropout mask to be multiplied with the input.
                Defaults to None.
            seed (int): An integer to use as a random seed.
                Defaults to None.
        """

        super().__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)

        self.always_on = always_on

    def call(self, inputs, training: bool = None):
        """
        Args:
            inputs: Input tensor of any rank.
            training: Toggles whether the Dropout layer is called in training
                (dropout-active) or testing (dropout-inactive) mode.
                Defaults to None.
        """

        output = super().call(inputs, training=1 if self.always_on else training)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({"always_on": self.always_on})

        return config


###############################################################################
################################## FUNCTIONS ##################################
###############################################################################


def check_gpu_support():
    # checks if TensorFlow/Keras can identify GPUs/CUDA-compatible devices on
    # the system; prints out a message and the identified devices

    dev_type = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"

    str_ = ">> detecting GPU/nVidia CUDA acceleration support: {}"
    print(str_.format("supported" if dev_type == "GPU" else "unsupported"))

    print(f">> listing available {dev_type} devices:")
    for i, device in enumerate(tf.config.list_physical_devices(dev_type)):
        print(f"  >> {i + 1}. {device[0]}")
    print("")

    return 0


def set_callbacks(**kwargs) -> list:
    # returns a list of tensorflow.keras.callbacks assembled from the **kwargs
    # passed to the function; expects dictionaries containing key/value pairs
    # to pass through to the appropriate tensorflow.keras.callbacks

    callbacks_ = {
        "csvlogger": CSVLogger,
        "earlystopping": EarlyStopping,
        "reducelronplateau": ReduceLROnPlateau,
    }

    callbacks = []

    for callback_label, callback_ in callbacks_.items():
        if callback_label in kwargs:
            callbacks.append(callback_(**kwargs[callback_label]))

    return callbacks


def build_mlp(
    out_dim: int,
    n_hl: int = 2,
    hl_ini_dim: int = 256,
    hl_shrink: float = 0.5,
    activation: str = "relu",
    loss: str = "mse",
    lr: float = 0.001,
    dropout: float = 0.2,
    dropout_always_on: bool = False,
    kernel_init: str = "he_uniform",
    bias_init: str = "zeros",
    random_state: RandomState = None,
) -> Sequential:
    # returns a tensorflow.keras.models.Sequential neural network with the deep
    # multilayer perceptron (MLP) model; the MLP has an output layer of
    # `out_dim` neurons; there are `n_hl` hidden layers between the input and
    # output layers, the first hidden layer has `hl_ini_dim` neurons, and
    # successive changes in size layer-on-layer are controlled via `hl_shrink`

    if random_state:
        tf.random.set_seed(random_state.randint(2**16))

    net = Sequential()

    ini_condition = {
        "kernel_initializer": kernel_init,
        "kernel_regularizer": None,
        "bias_initializer": bias_init,
        "bias_regularizer": None,
    }

    # print(ini_condition)
    net.add(Dense(hl_ini_dim, **ini_condition))
    net.add(Activation(activation))
    net.add(Dropout_(dropout, always_on=dropout_always_on))

    for i in range(n_hl - 1):
        hl_dim = int(hl_ini_dim * (hl_shrink ** (i + 1)))
        # print(hl_dim)
        net.add(Dense(hl_dim if (hl_dim > 1) else 1, **ini_condition))
        net.add(Activation(activation))
        net.add(Dropout_(dropout, always_on=dropout_always_on))
    # print(out_dim)
    net.add(Dense(out_dim))
    net.add(Activation("linear"))

    net.compile(loss=loss, optimizer=Adam(learning_rate=lr))

    # print(net.summary())

    return net
