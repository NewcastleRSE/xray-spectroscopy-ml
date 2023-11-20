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

from xanesnet.scheme.base_predict import Predict
from xanesnet.data_transform import (
    fourier_transform_data,
    inverse_fourier_transform_data,
)


class NNPredict(Predict):
    def predict(self, model):
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
                Predict.print_mse(
                    "xyz", "xyz prediction", self.xyz_data, xyz_pred.detach().numpy()
                )

            result = self.reshape(xyz_pred)

        elif self.pred_mode == "predict_xanes":
            xyz = torch.from_numpy(self.xyz_data)
            xyz = xyz.float()

            xanes_pred = model(xyz)

            if self.fourier:
                xanes_pred = inverse_fourier_transform_data(xanes_pred)

            # print MSE if evaluation data is provided
            if self.pred_eval:
                Predict.print_mse(
                    "xanes",
                    "xanes prediction",
                    self.xanes_data,
                    xanes_pred.detach().numpy(),
                )

            result = self.reshape(xanes_pred)

        return result

    def predict_bootstrap(self, model_list):
        predict_score = []
        result_list = []

        for model in model_list:
            if self.pred_mode == "predict_xyz":
                xyz_pred = self.predict(model)
                if self.pred_eval:
                    mse = mean_squared_error(self.xyz_data, xyz_pred.detach().numpy())
                    predict_score.append(mse)

                result_list.append(xyz_pred.detach().numpy())

            elif self.pred_mode == "predict_xanes":
                xanes_pred = self.predict(model)
                if self.pred_eval:
                    mse = mean_squared_error(
                        self.xanes_data, xanes_pred.detach().numpy()
                    )
                    predict_score.append(mse)

                result_list.append(xanes_pred.detach().numpy())

        if self.pred_eval:
            mean_score = torch.mean(torch.tensor(predict_score))
            std_score = torch.std(torch.tensor(predict_score))
            print(f"Mean score prediction: {mean_score:.4f}, Std: {std_score:.4f}")

        result_list = np.asarray(result_list)
        result_mean = np.mean(result_list, axis=0)
        result_std = np.std(result_list, axis=0)

        return result_mean, result_std

    def predict_ensemble(self, model_list):
        result_list = []

        for model in model_list:
            if self.pred_mode == "predict_xyz":
                xyz_pred = self.predict(model)
                result_list.append(xyz_pred.detach().numpy())

            elif self.pred_mode == "predict_xanes":
                xanes_pred = self.predict(model)
                result_list.append(xanes_pred.detach().numpy())

        result_list = sum(result_list) / len(result_list)

        if self.pred_mode == "predict_xyz" and self.pred_eval:
            Predict.print_mse("xyz", "xyz prediction", self.xyz_data, result_list)
        elif self.pred_mode == "predict_xanes" and self.pred_eval:
            Predict.print_mse("xanes", "xanes prediction", self.xanes_data, result_list)

        result_list = torch.tensor(np.asarray(result_list)).float()
        result_mean = torch.mean(result_list, dim=0)
        result_std = torch.std(result_list, dim=0)

        return result_mean, result_std
