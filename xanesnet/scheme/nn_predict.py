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
            pred = model(xanes)

            if self.pred_eval:
                Predict.print_mse("xyz", self.xyz_data, pred)

            return pred

        elif self.pred_mode == "predict_xanes":
            xyz = torch.from_numpy(self.xyz_data)
            xyz = xyz.float()

            pred = model(xyz)

            if self.fourier:
                pred = inverse_fourier_transform_data(pred)

            if self.pred_eval:
                Predict.print_mse("xanes", self.xanes_data, pred)

            return pred

    def predict_bootstrap(self, model_list):
        predict_score = []
        pred_all = []

        for model in model_list:
            model.eval()
            if self.pred_mode == "predict_xyz":
                xyz_pred = self.predict(model)

                if self.pred_eval:
                    mse = mean_squared_error(self.xyz_data, xyz_pred.detach().numpy())
                    predict_score.append(mse)

                pred_all.append(xyz_pred.detach().numpy())

            elif self.pred_mode == "predict_xanes":
                xanes_pred = self.predict(model)
                if self.pred_eval:
                    mse = mean_squared_error(
                        self.xanes_data, xanes_pred.detach().numpy()
                    )
                    predict_score.append(mse)

                pred_all.append(xanes_pred.detach().numpy())

        if self.pred_eval:
            mean_score = torch.mean(torch.tensor(predict_score))
            std_score = torch.std(torch.tensor(predict_score))
            print(f"Mean score: {mean_score:.4f}, Std score: {std_score:.4f}")

        pred_all = np.asarray(pred_all)
        pred_mean = np.mean(pred_all, axis=0)
        pred_std = np.std(pred_all, axis=0)

        return pred_mean, pred_std

    def predict_ensemble(self, model_list):
        pred_list = []

        for model in model_list:
            model.eval()
            if self.pred_mode == "predict_xyz":
                xyz_pred = self.predict(model)
                pred_list.append(xyz_pred.detach().numpy())

            elif self.pred_mode == "predict_xanes":
                xanes_pred = self.predict(model)
                pred_list.append(xanes_pred.detach().numpy())

        pred = sum(pred_list) / len(pred_list)

        if self.pred_mode == "predict_xyz":
            Predict.print_mse("xyz", self.xyz_data, pred)
        elif self.pred_mode == "predict_xanes":
            Predict.print_mse("xanes", self.xanes_data, pred)

        pred_list = torch.tensor(np.asarray(pred_list)).float()
        pred_mean = torch.mean(pred_list, dim=0)
        pred_std = torch.std(pred_list, dim=0)

        return pred_std, pred_mean
