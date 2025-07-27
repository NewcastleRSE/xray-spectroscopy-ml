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
import logging
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error


class Predict(ABC):
    """
    Base class for prediction

    """

    def __init__(self, xyz_data, xanes_data, **kwargs):
        self.xyz_data = xyz_data
        self.xanes_data = xanes_data

        self.mode = kwargs.get("pred_mode")
        self.pred_eval = kwargs.get("pred_eval")
        self.scaler = kwargs.get("scaler")
        self.fourier = kwargs.get("fourier")
        self.fourier_concat = kwargs.get("fourier_concat")

        self.recon_flag = 0

    @abstractmethod
    def predict(self, model):
        pass

    @abstractmethod
    def predict_std(self, model):
        pass

    @abstractmethod
    def predict_bootstrap(self, model_list):
        pass

    @abstractmethod
    def predict_ensemble(self, model_list):
        pass

    @staticmethod
    def print_mse(
        source_name: str, target_name: str, data: np.ndarray, result: np.ndarray
    ):
        mse = mean_squared_error(data, result)
        logging.info(f"Mean Squared Error ({source_name} → {target_name}): {mse:.6f}")

    @staticmethod
    def setup_scaler(scaler, x_data, inverse: bool):
        if not inverse:
            scaler.fit(x_data)
            x_data = scaler.transform(x_data)
        else:
            x_data = scaler.inverse_transform(x_data)

        return x_data
