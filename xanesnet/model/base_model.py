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

from torch import nn


class Model(nn.Module):
    """
    A base class for constructing neural network model
    with flags indicating different model types.

    Attributes:
        nn_flag (int): if nn_flag = 1, the model is configured as a classic type of Neural Network.
        ae_flag (int): if ae_flag = 1, the model is configured as Autoencoder network.
        aegan_flag (int): if aegan_flag = 1, the model is configured as an
            Autoencoder-Generative Adversarial Network
    """

    def __init__(self):
        super().__init__()

        self.nn_flag = 0
        self.ae_flag = 0
        self.aegan_flag = 0
