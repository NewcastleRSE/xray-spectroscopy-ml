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

from xanesnet.scheme.base_predict import Predict
from xanesnet.data_transform import (
    fourier_transform_data,
    inverse_fourier_transform_data,
)


@dataclass
class Result:
    xyz_pred: torch.Tensor
    xanes_pred: torch.Tensor


class NNPredict(Predict):
    def predict(self, model):
        xyz_pred = None
        xanes_pred = None
        model.eval()
        if self.pred_mode == "predict_xyz":
            if self.fourier:
                xanes_fourier = fourier_transform_data(self.xanes_data)
                xanes = torch.from_numpy(xanes_fourier)
            else:
                xanes = torch.from_numpy(self.xanes_data)

            xanes = xanes.float()
            xyz_pred = model(xanes)

            # print MSE if evaluation data is provided
            if self.pred_eval:
                Predict.print_mse("xyz", "xyz prediction", self.xyz_data, xyz_pred)

        elif self.pred_mode == "predict_xanes":
            xyz = torch.from_numpy(self.xyz_data)
            xyz = xyz.float()

            xanes_pred = model(xyz)

            if self.fourier:
                xanes_pred = inverse_fourier_transform_data(xanes_pred)

            # print MSE if evaluation data is provided
            if self.pred_eval:
                Predict.print_mse(
                    "xanes", "xanes prediction", self.xanes_data, xanes_pred
                )

        return Result(xyz_pred, xanes_pred)

    def predict_bootstrap(self, model_list):
        predict_score = []
        xyz_pred_list = []
        xanes_pred_list = []

        for i, model in enumerate(model_list, start=1):
            print(f">> Predict model {i}")
            if self.pred_mode == "predict_xyz":
                result = self.predict(model)
                if self.pred_eval:
                    mse = mean_squared_error(
                        self.xyz_data, result.xyz_pred.detach().numpy()
                    )
                    predict_score.append(mse)

                xyz_pred_list.append(result.xyz_pred.detach().numpy())

            elif self.pred_mode == "predict_xanes":
                result = self.predict(model)
                if self.pred_eval:
                    mse = mean_squared_error(
                        self.xanes_data, result.xanes_pred.detach().numpy()
                    )
                    predict_score.append(mse)

                xanes_pred_list.append(result.xanes_pred.detach().numpy())

        if self.pred_eval:
            mean_score = torch.mean(torch.tensor(predict_score))
            std_score = torch.std(torch.tensor(predict_score))
            print(f"Mean score prediction: {mean_score:.4f}, Std: {std_score:.4f}")

        xyz_pred_list = torch.tensor(np.asarray(xyz_pred_list)).float()
        xyz_pred = torch.mean(xyz_pred_list, dim=0)

        xanes_pred_list = torch.tensor(np.asarray(xanes_pred_list)).float()
        xanes_pred = torch.mean(xanes_pred_list, dim=0)

        return Result(xyz_pred, xanes_pred)

    def predict_ensemble(self, model_list):
        xyz_pred_list = []
        xanes_pred_list = []

        for i, model in enumerate(model_list, start=1):
            print(f">> Predict model {i}")
            if self.pred_mode == "predict_xyz":
                result = self.predict(model)
                xyz_pred_list.append(result.xyz_pred.detach().numpy())

            elif self.pred_mode == "predict_xanes":
                result = self.predict(model)
                xanes_pred_list.append(result.xanes_pred.detach().numpy())

        print(f"{'='*30}Summary{'='*30}")
        if len(xyz_pred_list) > 0:
            xyz_pred = sum(xyz_pred_list) / len(xyz_pred_list)
            Predict.print_mse("Ensemble xyz", "xyz_prediction", self.xyz_data, xyz_pred)

        if len(xanes_pred_list) > 0:
            xanes_pred = sum(xanes_pred_list) / len(xanes_pred_list)
            Predict.print_mse(
                "Ensemble xanes", "xanes_prediction", self.xanes_data, xanes_pred
            )

        xyz_pred_list = torch.tensor(np.asarray(xyz_pred_list)).float()
        xyz_pred = torch.mean(xyz_pred_list, dim=0)

        xanes_pred_list = torch.tensor(np.asarray(xanes_pred_list)).float()
        xanes_pred = torch.mean(xanes_pred_list, dim=0)

        return Result(xyz_pred, xanes_pred)
