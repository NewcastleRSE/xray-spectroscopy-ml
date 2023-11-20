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

from xanesnet.scheme import Predict
from xanesnet.scheme.nn_predict import NNPredict
from xanesnet.data_transform import (
    fourier_transform_data,
    inverse_fourier_transform_data,
)


class AEPredict(NNPredict):
    def predict(self, model):
        model.eval()
        if self.pred_mode == "predict_xyz":
            xanes = torch.tensor(self.xanes_data).float()

            if self.fourier:
                xanes_fourier = fourier_transform_data(xanes)
                xanes_fourier = torch.tensor(xanes_fourier).float()
                xanes_recon = model.reconstruct(xanes_fourier)
                xanes_recon = inverse_fourier_transform_data(xanes_recon)
                xyz_pred = model.predict(xanes_fourier)

            else:
                xanes_recon = model.reconstruct(xanes)
                xyz_pred = model.predict(xanes)

            result_pred = xyz_pred
            result_recon = xanes_recon

            # print MSE
            Predict.print_mse(
                "xanes",
                "xanes reconstruction",
                self.xanes_data,
                xanes_recon.detach().numpy(),
            )
            if self.pred_eval:
                Predict.print_mse(
                    "xyz", "xyz prediction", self.xyz_data, xyz_pred.detach().numpy()
                )

        elif self.pred_mode == "predict_xanes":
            xyz = torch.tensor(self.xyz_data).float()

            xyz_recon = model.reconstruct(xyz)
            xanes_pred = model.predict(xyz)

            if self.fourier:
                xanes_pred = inverse_fourier_transform_data(xanes_pred)

            result_pred = xanes_pred
            result_recon = xyz_recon

            # print MSE
            Predict.print_mse(
                "xyz", "xyz reconstruction", self.xyz_data, xyz_recon.detach().numpy()
            )
            if self.pred_eval:
                Predict.print_mse(
                    "xanes",
                    "xanes prediction",
                    self.xanes_data,
                    xanes_pred.detach().numpy(),
                )

        return result_pred, result_recon

    def predict_bootstrap(self, model_list):
        predict_score = []
        recon_score = []
        result_list = []

        for model in model_list:
            if self.pred_mode == "predict_xyz":
                xyz_pred, xanes_recon = self.predict(model)
                mse = mean_squared_error(self.xanes_data, xanes_recon.detach().numpy())
                recon_score.append(mse)
                if self.pred_eval:
                    mse = mean_squared_error(self.xyz_data, xyz_pred.detach().numpy())
                    predict_score.append(mse)

                result_list.append(xyz_pred.detach().numpy())

            elif self.pred_mode == "predict_xanes":
                xanes_pred, xyz_recon = self.predict(model)
                mse = mean_squared_error(self.xyz_data, xyz_recon.detach().numpy())
                recon_score.append(mse)
                if self.pred_eval:
                    mse = mean_squared_error(
                        self.xanes_data, xanes_pred.detach().numpy()
                    )
                    predict_score.append(mse)

                result_list.append(xanes_pred.detach().numpy())

        mean_score = torch.mean(torch.tensor(predict_score))
        std_score = torch.std(torch.tensor(predict_score))
        print(f"Mean score prediction: {mean_score:.4f}, Std: {std_score:.4f}")
        mean_score = torch.mean(torch.tensor(recon_score))
        std_score = torch.std(torch.tensor(recon_score))
        print(f"Mean score reconstruction: {mean_score:.4f}, Std: {std_score:.4f}")

        result_list = np.asarray(result_list)
        result_mean = np.mean(result_list, axis=0)
        result_std = np.std(result_list, axis=0)

        return result_mean, result_std

    def predict_ensemble(self, model_list):
        predict_list = []
        recon_list = []

        for model in model_list:
            if self.pred_mode == "predict_xyz":
                xyz_pred, xanes_recon = self.predict(model)
                predict_list.append(xyz_pred.detach().numpy())
                recon_list.append(xanes_recon.detach().numpy())

            elif self.pred_mode == "predict_xanes":
                xanes_pred, xyz_recon = self.predict(model)
                predict_list.append(xanes_pred.detach().numpy())
                recon_list.append(xyz_recon.detach().numpy())

        predict_list = sum(predict_list) / len(predict_list)
        recon_list = sum(recon_list) / len(recon_list)

        if self.pred_mode == "predict_xyz":
            Predict.print_mse(
                "xanes", "xanes reconstruction", self.xanes_data, recon_list
            )
            if self.pred_eval:
                Predict.print_mse("xyz", "xyz_prediction", self.xyz_data, predict_list)

        elif self.pred_mode == "predict_xanes":
            Predict.print_mse("xyz", "xyz reconstruction", self.xyz_data, recon_list)
            if self.pred_eval:
                Predict.print_mse(
                    "xanes", "xanes_prediction", self.xanes_data, predict_list
                )

        result_list = torch.tensor(np.asarray(predict_list)).float()
        result_mean = torch.mean(result_list, dim=0)
        result_std = torch.std(result_list, dim=0)

        return result_mean, result_std
