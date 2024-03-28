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
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler

from xanesnet.scheme.base_predict import Predict
from xanesnet.data_transform import fourier_transform, inverse_fourier_transform


@dataclass
class Result:
    xyz_pred: (np.ndarray, np.ndarray)
    xanes_pred: (np.ndarray, np.ndarray)


class NNPredict(Predict):
    def predict(self, model) -> tuple[np.ndarray, np.ndarray]:
        xyz_pred = None
        xanes_pred = None

        model.eval()

        if self.pred_mode == "predict_xyz":
            if self.fourier:
                xanes_fourier = fourier_transform(self.xanes_data, self.fourier_concat)
                xanes = xanes_fourier
            else:
                xanes = self.xanes_data

            # Apply standardscaler to the training dataset
            if self.scaler:
                scaler = StandardScaler()
                xanes = self.setup_scaler(scaler, xanes, False)

            # Model prediction
            xanes = torch.from_numpy(xanes).float()
            xyz_pred = model(xanes)
            xyz_pred = xyz_pred.detach().numpy()

            # Print MSE if evaluation data is provided
            if self.pred_eval:
                Predict.print_mse("xyz", "xyz prediction", self.xyz_data, xyz_pred)

        elif self.pred_mode == "predict_xanes":
            xyz = self.xyz_data
            # Apply standardscaler to training dataset
            if self.scaler:
                scaler = StandardScaler()
                xyz = self.setup_scaler(scaler, xyz, False)

            # Model prediction
            xyz = torch.from_numpy(xyz).float()
            xanes_pred = model(xyz)
            xanes_pred = xanes_pred.detach().numpy()

            if self.fourier:
                xanes_pred = inverse_fourier_transform(xanes_pred, self.fourier_concat)

            # Print MSE if evaluation data is provided
            if self.pred_eval:
                Predict.print_mse(
                    "xanes", "xanes prediction", self.xanes_data, xanes_pred
                )

        return xyz_pred, xanes_pred

    def predict_std(self, model):
        xyz_std = None
        xanes_std = None

        print(f">> Predicting ...")
        xyz_pred, xanes_pred = self.predict(model)

        # Create dummy STD
        if self.pred_mode == "predict_xyz":
            xyz_std = np.zeros_like(xyz_pred)
        elif self.pred_mode == "predict_xanes":
            xanes_std = np.zeros_like(xanes_pred)

        return Result(xyz_pred=(xyz_pred, xyz_std), xanes_pred=(xanes_pred, xanes_std))

    def predict_bootstrap(self, model_list):
        predict_score = []
        xyz_pred_list = []
        xanes_pred_list = []

        # Iterate over models to perform predicting
        for i, model in enumerate(model_list, start=1):
            print(f">> Predicting with model {i}...")
            xyz_pred, xanes_pred = self.predict(model)

            if self.pred_mode == "predict_xyz":
                if self.pred_eval:
                    mse = mean_squared_error(self.xyz_data, xyz_pred)
                    predict_score.append(mse)

                xyz_pred_list.append(xyz_pred)

            elif self.pred_mode == "predict_xanes":
                if self.pred_eval:
                    mse = mean_squared_error(self.xanes_data, xanes_pred)
                    predict_score.append(mse)

                xanes_pred_list.append(xanes_pred)

        # Print MSE if evaluation data is provided
        if self.pred_eval:
            mean_score = np.mean(predict_score)
            std_score = np.std(predict_score)
            print(f"Mean score prediction: {mean_score:.4f}, Std: {std_score:.4f}")

        xyz_mean = np.mean(xyz_pred_list, axis=0)
        xyz_std = np.std(xyz_pred_list, axis=0)

        xanes_mean = np.mean(xanes_pred_list, axis=0)
        xanes_std = np.std(xanes_pred_list, axis=0)

        return Result(xyz_pred=(xyz_mean, xyz_std), xanes_pred=(xanes_mean, xanes_std))

    def predict_ensemble(self, model_list):
        xyz_pred_list = []
        xanes_pred_list = []

        # Iterate over models to perform predicting
        for i, model in enumerate(model_list, start=1):
            print(f">> Predicting with model {i}...")
            xyz_pred, xanes_pred = self.predict(model)

            if self.pred_mode == "predict_xyz":
                xyz_pred_list.append(xyz_pred)

            elif self.pred_mode == "predict_xanes":
                xanes_pred_list.append(xanes_pred)

        # Print MSE summary
        print(f"{'='*30}Ensemble Prediction Summary{'='*30}")
        if self.pred_mode == "predict_xyz":
            xyz_pred = sum(xyz_pred_list) / len(xyz_pred_list)
            Predict.print_mse("xyz", "xyz prediction", self.xyz_data, xyz_pred)

        elif self.pred_mode == "predict_xanes":
            xanes_pred = sum(xanes_pred_list) / len(xanes_pred_list)
            Predict.print_mse("xanes", "xanes prediction", self.xanes_data, xanes_pred)

        xyz_mean = np.mean(xyz_pred_list, axis=0)
        xyz_std = np.std(xyz_pred_list, axis=0)

        xanes_mean = np.mean(xanes_pred_list, axis=0)
        xanes_std = np.std(xanes_pred_list, axis=0)

        return Result(xyz_pred=(xyz_mean, xyz_std), xanes_pred=(xanes_mean, xanes_std))
