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
from xanesnet.scheme.nn_predict import NNPredict
from xanesnet.data_transform import (
    fourier_transform_data,
    inverse_fourier_transform_data,
)


@dataclass
class Result:
    xyz_pred: (torch.Tensor, torch.Tensor)
    xanes_pred: (torch.Tensor, torch.Tensor)
    xyz_recon: (torch.Tensor, torch.Tensor)
    xanes_recon: (torch.Tensor, torch.Tensor)


class AEPredict(Predict):
    def __init__(self, xyz_data, xanes_data, pred_mode, pred_eval, index, fourier):
        super().__init__(xyz_data, xanes_data, pred_mode, pred_eval, index, fourier)
        self.recon_flag = 1

    def predict(self, model):
        xyz_mean = None
        xanes_mean = None
        xyz_recon_mean = None
        xanes_recon_mean = None

        xyz_std = None
        xanes_std = None
        xyz_recon_std = None
        xanes_recon_std = None

        model.eval()

        if self.pred_mode == "predict_xyz":
            xanes = torch.tensor(self.xanes_data).float()

            if self.fourier:
                xanes_fourier = fourier_transform_data(xanes)
                xanes_fourier = torch.tensor(xanes_fourier).float()
                xanes_recon_mean = model.reconstruct(xanes_fourier)
                xanes_recon_mean = inverse_fourier_transform_data(xanes_recon_mean)
                xyz_mean = model.predict(xanes_fourier)

            else:
                xanes_recon_mean = model.reconstruct(xanes)
                xyz_mean = model.predict(xanes)

            xyz_std = torch.zeros_like(xyz_mean)
            xanes_recon_std = torch.zeros_like(xanes_recon_mean)

            # print MSE
            Predict.print_mse(
                "xanes", "xanes reconstruction", self.xanes_data, xanes_recon_mean
            )
            if self.pred_eval:
                Predict.print_mse("xyz", "xyz prediction", self.xyz_data, xyz_mean)

        elif self.pred_mode == "predict_xanes":
            xyz = torch.tensor(self.xyz_data).float()

            xyz_recon_mean = model.reconstruct(xyz)
            xanes_mean = model.predict(xyz)

            if self.fourier:
                xanes_mean = inverse_fourier_transform_data(xanes_mean)

            xanes_std = torch.zeros_like(xanes_mean)
            xyz_recon_std = torch.zeros_like(xyz_recon_mean)

            # print MSE
            Predict.print_mse(
                "xyz", "xyz reconstruction", self.xyz_data, xyz_recon_mean
            )
            if self.pred_eval:
                Predict.print_mse(
                    "xanes", "xanes prediction", self.xanes_data, xanes_mean
                )

        return Result(
            xyz_pred=(xyz_mean, xyz_std),
            xanes_pred=(xanes_mean, xanes_std),
            xyz_recon=(xyz_recon_mean, xyz_recon_std),
            xanes_recon=(xanes_recon_mean, xanes_recon_std),
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
            print(f">> Predict model {i}")
            if self.pred_mode == "predict_xyz":
                result = self.predict(model)
                mse = mean_squared_error(
                    self.xanes_data, result.xanes_recon[0].detach().numpy()
                )
                xanes_recon_score.append(mse)
                if self.pred_eval:
                    mse = mean_squared_error(
                        self.xyz_data, result.xyz_pred[0].detach().numpy()
                    )
                    xyz_pred_score.append(mse)

                xyz_pred_list.append(result.xyz_pred[0].detach().numpy())
                xanes_recon_list.append(result.xanes_recon[0].detach().numpy())

            elif self.pred_mode == "predict_xanes":
                result = self.predict(model)
                mse = mean_squared_error(
                    self.xyz_data, result.xyz_recon[0].detach().numpy()
                )
                xyz_recon_score.append(mse)
                if self.pred_eval:
                    mse = mean_squared_error(
                        self.xanes_data, result.xanes_pred[0].detach().numpy()
                    )
                    xanes_pred_score.append(mse)

                xanes_pred_list.append(result.xanes_pred[0].detach().numpy())
                xyz_recon_list.append(result.xyz_recon[0].detach().numpy())

        if len(xyz_pred_score) > 0:
            xyz_pred_mean = torch.mean(torch.tensor(xyz_pred_score))
            xyz_pred_std = torch.std(torch.tensor(xyz_pred_score))
            print(
                f"Mean score xyz prediction: {xyz_pred_mean:.4f}, Std: {xyz_pred_std:.4f}"
            )

        if len(xanes_pred_score) > 0:
            xanes_pred_mean = torch.mean(torch.tensor(xanes_pred_score))
            xanes_pred_std = torch.std(torch.tensor(xanes_pred_score))
            print(
                f"Mean score xanes prediction: {xanes_pred_mean:.4f}, Std: {xanes_pred_std:.4f}"
            )

        if len(xyz_recon_score) > 0:
            xyz_recon_mean = torch.mean(torch.tensor(xyz_recon_score))
            xyz_recon_std = torch.std(torch.tensor(xyz_recon_score))
            print(
                f"Mean score xyz reconstruction: {xyz_recon_mean:.4f}, Std: {xyz_recon_std:.4f}"
            )

        if len(xanes_recon_score) > 0:
            xanes_recon_mean = torch.mean(torch.tensor(xanes_recon_score))
            xanes_recon_std = torch.std(torch.tensor(xanes_recon_score))
            print(
                f"Mean score xanes reconstruction: {xanes_recon_mean:.4f}, Std: {xanes_recon_std:.4f}"
            )

        xyz_pred_list = torch.tensor(np.asarray(xyz_pred_list)).float()
        xyz_mean = torch.mean(xyz_pred_list, dim=0)
        xyz_std = torch.std(xyz_pred_list, dim=0)

        xanes_pred_list = torch.tensor(np.asarray(xanes_pred_list)).float()
        xanes_mean = torch.mean(xanes_pred_list, dim=0)
        xanes_std = torch.std(xanes_pred_list, dim=0)

        xyz_recon_list = torch.tensor(np.asarray(xyz_recon_list)).float()
        xyz_recon_mean = torch.mean(xyz_recon_list, dim=0)
        xyz_recon_std = torch.std(xyz_recon_list, dim=0)

        xanes_recon_list = torch.tensor(np.asarray(xanes_recon_list)).float()
        xanes_recon_mean = torch.mean(xanes_recon_list, dim=0)
        xanes_recon_std = torch.std(xanes_recon_list, dim=0)

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
            print(f">> Predict model {i}")
            if self.pred_mode == "predict_xyz":
                result = self.predict(model)
                xyz_pred_list.append(result.xyz_pred[0].detach().numpy())
                xanes_recon_list.append(result.xanes_recon[0].detach().numpy())

            elif self.pred_mode == "predict_xanes":
                result = self.predict(model)
                xanes_pred_list.append(result.xanes_pred[0].detach().numpy())
                xyz_recon_list.append(result.xyz_recon[0].detach().numpy())

        print(f"{'='*30}Summary{'='*30}")
        if self.pred_mode == "predict_xyz":
            xanes_recon = sum(xanes_recon_list) / len(xanes_recon_list)
            Predict.print_mse(
                "Ensemble xanes", "xanes reconstruction", self.xanes_data, xanes_recon
            )
            if self.pred_eval:
                xyz_pred = sum(xyz_pred_list) / len(xyz_pred_list)
                Predict.print_mse(
                    "Ensemble xyz", "xyz_prediction", self.xyz_data, xyz_pred
                )
        elif self.pred_mode == "predict_xanes":
            if len(xyz_recon_list) > 0:
                xyz_recon = sum(xyz_recon_list) / len(xyz_recon_list)
                Predict.print_mse(
                    "Ensemble xyz", "xyz reconstruction", self.xyz_data, xyz_recon
                )
            if self.pred_eval:
                xanes_pred = sum(xanes_pred_list) / len(xanes_pred_list)
                Predict.print_mse(
                    "Ensemble xanes", "xanes_prediction", self.xanes_data, xanes_pred
                )

        xyz_pred_list = torch.tensor(np.asarray(xyz_pred_list)).float()
        xyz_mean = torch.mean(xyz_pred_list, dim=0)
        xyz_std = torch.std(xyz_pred_list, dim=0)

        xanes_pred_list = torch.tensor(np.asarray(xanes_pred_list)).float()
        xanes_mean = torch.mean(xanes_pred_list, dim=0)
        xanes_std = torch.std(xanes_pred_list, dim=0)

        xyz_recon_list = torch.tensor(np.asarray(xyz_recon_list)).float()
        xyz_recon_mean = torch.mean(xyz_recon_list, dim=0)
        xyz_recon_std = torch.std(xyz_recon_list, dim=0)

        xanes_recon_list = torch.tensor(np.asarray(xanes_recon_list)).float()
        xanes_recon_mean = torch.mean(xanes_recon_list, dim=0)
        xanes_recon_std = torch.std(xanes_recon_list, dim=0)

        return Result(
            xyz_pred=(xyz_mean, xyz_std),
            xanes_pred=(xanes_mean, xanes_std),
            xyz_recon=(xyz_recon_mean, xyz_recon_std),
            xanes_recon=(xanes_recon_mean, xanes_recon_std),
        )
