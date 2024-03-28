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

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from xanesnet.scheme.base_predict import Predict
from xanesnet.data_transform import (
    fourier_transform,
    inverse_fourier_transform,
)


@dataclass
class Result:
    xyz_pred: (np.ndarray, np.ndarray)
    xanes_pred: (np.ndarray, np.ndarray)
    xyz_recon: (np.ndarray, np.ndarray)
    xanes_recon: (np.ndarray, np.ndarray)


class AEPredict(Predict):
    def __init__(
        self,
        xyz_data,
        xanes_data,
        pred_mode,
        pred_eval,
        index,
        scaler,
        fourier,
        fourier_param,
    ):
        super().__init__(
            xyz_data,
            xanes_data,
            pred_mode,
            pred_eval,
            index,
            scaler,
            fourier,
            fourier_param,
        )
        self.recon_flag = 1

    def predict(self, model):
        xyz_pred = None
        xanes_pred = None
        xyz_recon = None
        xanes_recon = None

        model.eval()

        if self.pred_mode == "predict_xyz":
            if self.fourier:
                xanes = fourier_transform(self.xanes_data, self.fourier_concat)
            else:
                xanes = self.xanes_data

            # Apply standardscaler to the training dataset
            if self.scaler:
                scaler = StandardScaler()
                xanes = self.setup_scaler(scaler, xanes, False)

            xanes = torch.from_numpy(xanes).float()

            # Prediction and reconstruction
            xyz_pred = model.predict(xanes)
            xyz_pred = xyz_pred.detach().numpy()
            xanes_recon = model.reconstruct(xanes)
            xanes_recon = xanes_recon.detach().numpy()

            # Standardscaler inverse transform
            if self.scaler:
                xanes_recon = self.setup_scaler(scaler, xanes_recon, True)

            # Fourier inverse transform
            if self.fourier:
                xanes_recon = inverse_fourier_transform(
                    xanes_recon, self.fourier_concat
                )

            # Print MSE
            Predict.print_mse(
                "xanes", "xanes reconstruction", self.xanes_data, xanes_recon
            )
            if self.pred_eval:
                Predict.print_mse("xyz", "xyz prediction", self.xyz_data, xyz_pred)

        elif self.pred_mode == "predict_xanes":
            xyz = self.xyz_data
            # Apply standardscaler to training dataset
            if self.scaler:
                scaler = StandardScaler()
                xyz = self.setup_scaler(scaler, xyz, False)

            xyz = torch.from_numpy(xyz).float()

            # Prediction and reconstruction
            xanes_pred = model.predict(xyz)
            xanes_pred = xanes_pred.detach().numpy()
            xyz_recon = model.reconstruct(xyz)
            xyz_recon = xyz_recon.detach().numpy()

            # Standardscaler inverse transform
            if self.scaler:
                xyz_recon = self.setup_scaler(scaler, xyz_recon, True)

            # Fourier inverse transform
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
        xyz_std = None
        xanes_std = None
        xyz_recon_std = None
        xanes_recon_std = None

        xyz_pred, xanes_pred, xyz_recon, xanes_recon = self.predict(model)

        # Create dummy array for STD
        if self.pred_mode == "predict_xyz":
            xyz_std = np.zeros_like(xyz_pred)
            xanes_recon_std = np.zeros_like(xanes_recon)
        elif self.pred_mode == "predict_xanes":
            xanes_std = np.zeros_like(xanes_pred)
            xyz_recon_std = np.zeros_like(xyz_recon)

        return Result(
            xyz_pred=(xyz_pred, xyz_std),
            xanes_pred=(xanes_pred, xanes_std),
            xyz_recon=(xyz_recon, xyz_recon_std),
            xanes_recon=(xanes_recon, xanes_recon_std),
        )

    def predict_bootstrap(self, model_list):
        xyz_pred_score = []
        xyz_recon_score = []
        xanes_pred_score = []
        xanes_recon_score = []

        xyz_pred_list = []
        xanes_pred_list = []
        xyz_recon_list = []
        xanes_recon_list = []

        for i, model in enumerate(model_list, start=1):
            print(f">> Predicting with model {i}...")
            xyz_pred, xanes_pred, xyz_recon, xanes_recon = self.predict(model)

            if self.pred_mode == "predict_xyz":
                mse = mean_squared_error(self.xanes_data, xanes_recon)
                xanes_recon_score.append(mse)

                if self.pred_eval:
                    mse = mean_squared_error(self.xyz_data, xyz_pred)
                    xyz_pred_score.append(mse)

                xyz_pred_list.append(xyz_pred)
                xanes_recon_list.append(xanes_recon)

            elif self.pred_mode == "predict_xanes":
                mse = mean_squared_error(self.xyz_data, xyz_recon)
                xyz_recon_score.append(mse)

                if self.pred_eval:
                    mse = mean_squared_error(self.xanes_data, xanes_pred)
                    xanes_pred_score.append(mse)

                xanes_pred_list.append(xanes_pred)
                xyz_recon_list.append(xyz_recon)

        if len(xyz_pred_score) > 0:
            xyz_pred_mean = np.mean(xyz_pred_score)
            xyz_pred_std = np.std(xyz_pred_score)
            print(
                f"Mean score xyz prediction: {xyz_pred_mean:.4f}, Std: {xyz_pred_std:.4f}"
            )

        if len(xanes_pred_score) > 0:
            xanes_pred_mean = np.mean(xanes_pred_score)
            xanes_pred_std = np.std(xanes_pred_score)
            print(
                f"Mean score xanes prediction: {xanes_pred_mean:.4f}, Std: {xanes_pred_std:.4f}"
            )

        if len(xyz_recon_score) > 0:
            xyz_recon_mean = np.mean(xyz_recon_score)
            xyz_recon_std = np.std(xyz_recon_score)
            print(
                f"Mean score xyz reconstruction: {xyz_recon_mean:.4f}, Std: {xyz_recon_std:.4f}"
            )

        if len(xanes_recon_score) > 0:
            xanes_recon_mean = np.mean(xanes_recon_score)
            xanes_recon_std = np.std(xanes_recon_score)
            print(
                f"Mean score xanes reconstruction: {xanes_recon_mean:.4f}, Std: {xanes_recon_std:.4f}"
            )

        xyz_mean = np.mean(xyz_pred_list, axis=0)
        xyz_std = np.std(xyz_pred_list, axis=0)

        xanes_mean = np.mean(xanes_pred_list, axis=0)
        xanes_std = np.std(xanes_pred_list, axis=0)

        xyz_recon_mean = np.mean(xyz_recon_list, axis=0)
        xyz_recon_std = np.std(xyz_recon_list, axis=0)

        xanes_recon_mean = np.mean(xanes_recon_list, axis=0)
        xanes_recon_std = np.std(xanes_recon_list, axis=0)

        return Result(
            xyz_pred=(xyz_mean, xyz_std),
            xanes_pred=(xanes_mean, xanes_std),
            xyz_recon=(xyz_recon_mean, xyz_recon_std),
            xanes_recon=(xanes_recon_mean, xanes_recon_std),
        )

    def predict_ensemble(self, model_list):
        xyz_pred_list = []
        xyz_recon_list = []
        xanes_pred_list = []
        xanes_recon_list = []

        for i, model in enumerate(model_list, start=1):
            print(f">> Predicting with model {i}...")
            xyz_pred, xanes_pred, xyz_recon, xanes_recon = self.predict(model)

            if self.pred_mode == "predict_xyz":
                xyz_pred_list.append(xyz_pred)
                xanes_recon_list.append(xanes_recon)

            elif self.pred_mode == "predict_xanes":
                xanes_pred_list.append(xanes_pred)
                xyz_recon_list.append(xyz_recon)

        print(f"{'='*30}Ensemble Prediction Summary{'='*30}")
        if self.pred_mode == "predict_xyz":
            xanes_recon = sum(xanes_recon_list) / len(xanes_recon_list)
            Predict.print_mse(
                "Ensemble xanes", "xanes reconstruction", self.xanes_data, xanes_recon
            )
            if self.pred_eval:
                xyz_pred = sum(xyz_pred_list) / len(xyz_pred_list)
                Predict.print_mse(
                    "Ensemble xyz", "xyz prediction", self.xyz_data, xyz_pred
                )

        elif self.pred_mode == "predict_xanes":
            xyz_recon = sum(xyz_recon_list) / len(xyz_recon_list)
            Predict.print_mse(
                "Ensemble xyz", "xyz reconstruction", self.xyz_data, xyz_recon
            )
            if self.pred_eval:
                xanes_pred = sum(xanes_pred_list) / len(xanes_pred_list)
                Predict.print_mse(
                    "Ensemble xanes", "xanes prediction", self.xanes_data, xanes_pred
                )

        xyz_mean = np.mean(xyz_pred_list, axis=0)
        xyz_std = np.std(xyz_pred_list, axis=0)

        xanes_mean = np.mean(xanes_pred_list, axis=0)
        xanes_std = np.std(xanes_pred_list, axis=0)

        xyz_recon_mean = np.mean(xyz_recon_list, axis=0)
        xyz_recon_std = np.std(xyz_recon_list, axis=0)

        xanes_recon_mean = np.mean(xanes_recon_list, axis=0)
        xanes_recon_std = np.std(xanes_recon_list, axis=0)

        return Result(
            xyz_pred=(xyz_mean, xyz_std),
            xanes_pred=(xanes_mean, xanes_std),
            xyz_recon=(xyz_recon_mean, xyz_recon_std),
            xanes_recon=(xanes_recon_mean, xanes_recon_std),
        )
