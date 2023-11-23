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
import numpy as np
import torch

from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error


class Predict(ABC):
    """
    Base class for prediction

    """

    def __init__(self, xyz_data, xanes_data, pred_mode, pred_eval, index, fourier):
        self.pred_mode = pred_mode
        self.pred_eval = pred_eval
        self.xyz_data = xyz_data
        self.xanes_data = xanes_data
        self.ids = index
        self.fourier = fourier

        self.pred_xyz = None
        self.pred_xanes = None
        self.recon_xyz = None
        self.recon_xanes = None

    @staticmethod
    def print_mse(name, name2, data, result):
        if isinstance(result, torch.Tensor):
            mse = mean_squared_error(data, result.detach().numpy())
        elif isinstance(result, np.ndarray):
            mse = mean_squared_error(data, result)

        print(f"MSE {name} data -> {name2}: {mse}")

    @abstractmethod
    def predict(self, model):
        pass

    @abstractmethod
    def predict_bootstrap(self, model_list):
        pass

    @abstractmethod
    def predict_ensemble(self, model_list):
        pass
