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
import torch

from typing import Optional, Tuple, List
from dataclasses import dataclass
from torch_geometric.data import DataLoader

from xanesnet.scheme.base_predict import Predict
from xanesnet.utils.fourier import inverse_fourier_transform


@dataclass
class Result:
    """Data class to hold prediction results, including mean and standard deviation."""

    xyz_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xanes_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None


class GNNPredict(Predict):
    def predict(self, model) -> np.ndarray:
        """
        Performs a single prediction with a given model.
        """
        model.eval()
        dataloader = DataLoader(self.xyz_data, batch_size=1, shuffle=False)

        with torch.no_grad():
            predictions = [model(data).squeeze().numpy() for data in dataloader]

        xanes_pred = np.array(predictions)

        if self.fourier:
            xanes_pred = inverse_fourier_transform(xanes_pred, self.fourier_concat)
        if self.pred_eval:
            Predict.print_mse("xanes", "xanes prediction", self.xanes_data, xanes_pred)

        return xanes_pred

    def predict_std(self, model):
        model_type = model.__class__.__name__.lower()
        logging.info(f"\n--- Starting prediction with model: {model_type} ---")

        xanes_pred = self.predict(model)
        # Create dummy STD
        xanes_sd = np.zeros_like(xanes_pred)

        return Result(xanes_pred=(xanes_pred, xanes_sd))

    def predict_bootstrap(self, model_list):
        """
        Performs predictions on multiple models (bootstrapping)
        """
        # Get all predictions and reconstructions from model_list
        all_preds = self._predict_from_models(model_list)

        # Calculate mean and std
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)

        # Print MSE of the mean prediction
        if self.pred_eval:
            target_data = (
                self.xyz_data if self.mode == "predict_xyz" else self.xanes_data
            )
            logging.info("-" * 55)
            Predict.print_mse("target", "mean prediction", target_data, mean_pred)

        return Result(xanes_pred=(mean_pred, std_pred))

    def predict_ensemble(self, model_list):
        """
        Performs predictions on multiple models (ensemble)
        """
        return self.predict_bootstrap(model_list)

    def _predict_from_models(self, model_list: List[torch.nn.Module]) -> np.ndarray:
        """
        Predictions for a list of models.
        """
        predictions = []
        for i, model in enumerate(model_list, start=1):
            model_type = model.__class__.__name__.lower()
            logging.info(
                f">> Predicting with model {model_type} ({i}/{len(model_list)})..."
            )
            prediction = self.predict(model)
            predictions.append(prediction)

        return np.array(predictions)
