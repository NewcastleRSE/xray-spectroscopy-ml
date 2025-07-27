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
from sklearn.preprocessing import StandardScaler

from xanesnet.scheme.base_predict import Predict
from xanesnet.utils.fourier import fourier_transform, inverse_fourier_transform


@dataclass
class Result:
    """Data class to hold prediction results, including mean and standard deviation."""

    xyz_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xanes_pred: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xyz_recon: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xanes_recon: Optional[Tuple[np.ndarray, np.ndarray]] = None


class AEGANPredict(Predict):
    def __init__(self, xyz_data, xanes_data, **kwargs):
        super().__init__(xyz_data, xanes_data, **kwargs)
        self.recon_flag = 1

    def predict(self, model):
        """
        Performs a single prediction with a given model.
        """
        xyz_pred = xanes_pred = xyz_recon = xanes_recon = None
        model.eval()

        if self.mode in ["predict_xyz", "predict_all"]:
            input_data = self.xanes_data

            if self.fourier:
                input_data = fourier_transform(input_data, self.fourier_concat)
            if self.scaler:
                scaler = StandardScaler()
                input_data = self.setup_scaler(scaler, input_data, inverse=False)

            # Prediction and reconstruction
            input_tensor = torch.from_numpy(input_data).float()
            xyz_pred = model.predict_structure(input_tensor).detach().numpy()
            xanes_recon = model.reconstruct_spectrum(input_tensor).detach().numpy()

            # Inverse Standardscaler and Fourier transform
            if self.fourier:
                xanes_recon = inverse_fourier_transform(
                    xanes_recon, self.fourier_concat
                )
            if self.scaler:
                xanes_recon = self.setup_scaler(scaler, xanes_recon, inverse=True)

            Predict.print_mse(
                "xanes", "xanes reconstruction", self.xanes_data, xanes_recon
            )
            if self.pred_eval:
                Predict.print_mse("xyz", "xyz prediction", self.xyz_data, xyz_pred)

        if self.mode in ["predict_xanes", "predict_all"]:
            input_data = self.xyz_data
            if self.scaler:
                scaler = StandardScaler()
                input_data = self.setup_scaler(scaler, input_data, inverse=False)

            input_tensor = torch.from_numpy(input_data).float()
            xanes_pred = model.predict_spectrum(input_tensor).detach().numpy()
            xyz_recon = model.reconstruct_structure(input_tensor).detach().numpy()

            if self.fourier:
                xanes_pred = inverse_fourier_transform(xanes_pred, self.fourier_concat)
            if self.scaler:
                xyz_recon = self.setup_scaler(scaler, xyz_recon, inverse=True)

            Predict.print_mse("xyz", "xyz reconstruction", self.xyz_data, xyz_recon)
            if self.pred_eval:
                Predict.print_mse(
                    "xanes", "xanes prediction", self.xanes_data, xanes_pred
                )

        return xyz_pred, xanes_pred, xyz_recon, xanes_recon

    def predict_std(self, model: torch.nn.Module) -> Result:
        """
        Performs a single prediction and returns the result with a zero (dummy)
        standard deviation array.
        """
        xyz_sd = xanes_sd = xyz_recon_sd = xanes_recon_sd = None
        model_type = model.__class__.__name__.lower()
        logging.info(f"\n--- Starting prediction with model: {model_type} ---")

        # Get all predictions and reconstructions
        xyz_pred, xanes_pred, xyz_recon, xanes_recon = self.predict(model)

        if self.mode in ["predict_xyz", "predict_all"]:
            xyz_sd = np.zeros_like(xyz_pred)
            xanes_recon_sd = np.zeros_like(xanes_recon)

        if self.mode in ["predict_xanes", "predict_all"]:
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
        Performs predictions on multiple models (bootstrapping) to calculate
        the mean and standard deviation for predictions and reconstructions.
        """
        all_preds = self._predict_from_models(model_list)
        all_xyz_p, all_xanes_p, all_xyz_r, all_xanes_r = all_preds

        xyz_pred_res = xanes_pred_res = xyz_recon_res = xanes_recon_res = None

        logging.info("-" * 55)

        if self.mode in ["predict_xyz", "predict_all"]:
            mean = np.mean(all_xyz_p, axis=0)
            sd = np.std(all_xyz_p, axis=0)
            xyz_pred_res = (mean, sd)
            if self.pred_eval:
                Predict.print_mse("xyz", "mean xyz prediction", self.xyz_data, mean)

            mean = np.mean(all_xanes_r, axis=0)
            sd = np.std(all_xanes_r, axis=0)
            xanes_recon_res = (mean, sd)
            Predict.print_mse(
                "xanes", "mean xanes reconstruction", self.xanes_data, mean
            )
        if self.mode in ["predict_xanes", "predict_all"]:
            mean = np.mean(all_xanes_p, axis=0)
            sd = np.std(all_xanes_p, axis=0)
            xanes_pred_res = (mean, sd)
            if self.pred_eval:
                Predict.print_mse(
                    "xanes", "mean xanes prediction", self.xanes_data, mean
                )

            mean = np.mean(all_xyz_r, axis=0)
            sd = np.std(all_xyz_r, axis=0)
            xyz_recon_res = (mean, sd)
            Predict.print_mse("xyz", "mean xyz reconstruction", self.xyz_data, mean)

        return Result(
            xyz_pred=xyz_pred_res,
            xanes_pred=xanes_pred_res,
            xyz_recon=xyz_recon_res,
            xanes_recon=xanes_recon_res,
        )

    def predict_ensemble(self, model_list: List[torch.nn.Module]) -> Result:
        """
        Performs predictions on an ensemble of models.
        Same to bootstrap
        """
        return self.predict_bootstrap(model_list)

    def _predict_from_models(self, model_list: List[torch.nn.Module]) -> Tuple:
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

            if self.mode in ["predict_xyz", "predict_all"]:
                xyz_preds.append(xyz_p)
                xanes_recons.append(xanes_r)

            if self.mode in ["predict_xanes", "predict_all"]:
                xanes_preds.append(xanes_p)
                xyz_recons.append(xyz_r)

        return (
            np.array(xyz_preds),
            np.array(xanes_preds),
            np.array(xyz_recons),
            np.array(xanes_recons),
        )
