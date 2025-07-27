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
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from xanesnet.scheme.base_predict import Predict
from xanesnet.utils.fourier import (
    fourier_transform,
    inverse_fourier_transform,
)


@dataclass
class Result:
    """Data class to hold prediction results, including mean and standard deviation."""

    xyz_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xanes_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xyz_recon: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xanes_recon: Optional[Tuple[np.ndarray, np.ndarray]] = None


class AEPredict(Predict):
    def __init__(self, xyz_data, xanes_data, **kwargs):
        super().__init__(xyz_data, xanes_data, **kwargs)
        self.recon_flag = 1

    def predict(self, model):
        """
        Performs a single prediction with a given model.
        """
        xyz_pred = xanes_pred = xyz_recon = xanes_recon = None
        model.eval()

        if self.mode == "predict_xyz":
            input_data = self.xanes_data

            if self.fourier:
                input_data = fourier_transform(input_data, self.fourier_concat)
            if self.scaler:
                scaler = StandardScaler()
                input_data = self.setup_scaler(scaler, input_data, inverse=False)

            input_tensor = torch.from_numpy(input_data).float()

            # Prediction and reconstruction
            xyz_pred = model.predict(input_tensor)
            xyz_pred = xyz_pred.detach().numpy()
            xanes_recon = model.reconstruct(input_tensor)
            xanes_recon = xanes_recon.detach().numpy()

            # Inverse Standardscaler and Fourier transform
            if self.scaler:
                xanes_recon = self.setup_scaler(scaler, xanes_recon, True)
            if self.fourier:
                xanes_recon = inverse_fourier_transform(
                    xanes_recon, self.fourier_concat
                )

            Predict.print_mse(
                "xanes", "xanes reconstruction", self.xanes_data, xanes_recon
            )
            if self.pred_eval:
                Predict.print_mse("xyz", "xyz prediction", self.xyz_data, xyz_pred)

        elif self.mode == "predict_xanes":
            # Predict xanes data
            input_data = self.xyz_data

            if self.scaler:
                scaler = StandardScaler()
                input_data = self.setup_scaler(scaler, input_data, False)

            input_tensor = torch.from_numpy(input_data).float()

            # Prediction and reconstruction
            xanes_pred = model.predict(input_tensor)
            xanes_pred = xanes_pred.detach().numpy()
            xyz_recon = model.reconstruct(input_tensor)
            xyz_recon = xyz_recon.detach().numpy()

            # Standardscaler inverse transform
            if self.scaler:
                xyz_recon = self.setup_scaler(scaler, xyz_recon, True)
            if self.fourier:
                xanes_pred = inverse_fourier_transform(xanes_pred, self.fourier_concat)

            # print MSE
            Predict.print_mse("xyz", "xyz reconstruction", self.xyz_data, xyz_recon)
            if self.pred_eval:
                Predict.print_mse(
                    "xanes", "xanes prediction", self.xanes_data, xanes_pred
                )

        return xyz_pred, xanes_pred, xyz_recon, xanes_recon

    def predict_std(self, model):
        """
        Performs a single prediction and returns the result with a zero (dummy)
        standard deviation array.
        """
        xyz_sd = xanes_sd = xyz_recon_sd = xanes_recon_sd = None
        model_type = model.__class__.__name__.lower()
        logging.info(f"\n--- Starting prediction with model: {model_type} ---")

        # Get all predictions and reconstructions
        xyz_pred, xanes_pred, xyz_recon, xanes_recon = self.predict(model)

        # Create dummy array for STD
        if self.mode == "predict_xyz":
            xyz_sd = np.zeros_like(xyz_pred)
            xanes_recon_sd = np.zeros_like(xanes_recon)
        elif self.mode == "predict_xanes":
            xanes_sd = np.zeros_like(xanes_pred)
            xyz_recon_sd = np.zeros_like(xyz_recon)

        return Result(
            xyz_pred=(xyz_pred, xyz_sd),
            xanes_pred=(xanes_pred, xanes_sd),
            xyz_recon=(xyz_recon, xyz_recon_sd),
            xanes_recon=(xanes_recon, xanes_recon_sd),
        )

    def predict_bootstrap(self, model_list: List[torch.nn.Module]) -> Result:
        """
        Predictions and reconstructions on multiple autoencoder models
        (bootstrapping) to calculate the mean and standard deviation.
        """
        # Get all predictions and reconstructions from model_list
        all_preds = self._predict_from_models(model_list)
        all_xyz_p, all_xanes_p, all_xyz_r, all_xanes_r = all_preds

        xyz_pred = xanes_pred = xyz_recon = xanes_recon = None

        logging.info("-" * 55)

        if self.mode == "predict_xyz":
            # Calculate mean and std for xyz predictions
            mean_xyz_pred = np.mean(all_xyz_p, axis=0)
            std_xyz_pred = np.std(all_xyz_p, axis=0)
            xyz_pred = (mean_xyz_pred, std_xyz_pred)

            # Calculate mean and std for xanes reconstructions
            mean_xanes_recon = np.mean(all_xanes_r, axis=0)
            std_xanes_recon = np.std(all_xanes_r, axis=0)
            xanes_recon = (mean_xanes_recon, std_xanes_recon)

            # Print MSE for mean prediction and reconstruction
            Predict.print_mse(
                "xanes", "mean xanes reconstruction", self.xanes_data, mean_xanes_recon
            )
            if self.pred_eval:
                Predict.print_mse(
                    "xyz", "mean xyz prediction", self.xyz_data, mean_xyz_pred
                )

        elif self.mode == "predict_xanes":
            # Calculate mean and std for xanes predictions
            mean_xanes_pred = np.mean(all_xanes_p, axis=0)
            sd_xanes_pred = np.std(all_xanes_p, axis=0)
            xanes_pred = (mean_xanes_pred, sd_xanes_pred)

            # Calculate mean and std for xyz reconstructions
            mean_xyz_recon = np.mean(all_xyz_r, axis=0)
            sd_xyz_recon = np.std(all_xyz_r, axis=0)
            xyz_recon = (mean_xyz_recon, sd_xyz_recon)

            # Print MSE for mean prediction and reconstruction
            Predict.print_mse(
                "xyz", "mean xyz reconstruction", self.xyz_data, mean_xyz_recon
            )
            if self.pred_eval:
                Predict.print_mse(
                    "xanes", "mean xanes prediction", self.xanes_data, mean_xanes_pred
                )

        # Return the comprehensive results
        return Result(
            xyz_pred=xyz_pred,
            xanes_pred=xanes_pred,
            xyz_recon=xyz_recon,
            xanes_recon=xanes_recon,
        )

    def predict_ensemble(self, model_list):
        """
        Performs predictions on an ensemble of models.
        Same to bootstrap
        """
        return self.predict_bootstrap(model_list)

    def _predict_from_models(
        self, model_list: List[torch.nn.Module]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predictions for a list of models.
        """
        xyz_preds, xanes_preds, xyz_recons, xanes_recons = [], [], [], []

        for i, model in enumerate(model_list, start=1):
            model_type = model.__class__.__name__.lower()
            logging.info(
                f">> Predicting with model {model_type} ({i}/{len(model_list)})..."
            )

            with torch.no_grad():
                xyz_p, xanes_p, xyz_r, xanes_r = self.predict(model)

            # Append results based on prediction mode
            if self.mode == "predict_xyz":
                xyz_preds.append(xyz_p)
                xanes_recons.append(xanes_r)
            elif self.mode == "predict_xanes":
                xanes_preds.append(xanes_p)
                xyz_recons.append(xyz_r)

        return (
            np.array(xyz_preds),
            np.array(xanes_preds),
            np.array(xyz_recons),
            np.array(xanes_recons),
        )
