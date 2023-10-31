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

    def predict_dim(self, data):
        if data.ndim == 1:
            if len(self.ids) == 1:
                data = data.reshape(-1, data.size)
            else:
                data = data.reshape(data.size, -1)

        print(">> ...predicted Xanes data!\n")

        return data

    @staticmethod
    def print_mse(name, data, pred_data):
        mse = mean_squared_error(data, pred_data.detach().numpy())
        print(f"MSE {name} to {name} pred: {mse}")

    @abstractmethod
    def predict(self, model):
        pass

    @abstractmethod
    def predict_bootstrap(self, model_list):
        pass

    @abstractmethod
    def predict_ensemble(self, model_list):
        pass
