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
                fourier_xanes = fourier_transform_data(self.xanes_data)
                xanes = torch.from_numpy(fourier_xanes)
            else:
                xanes = torch.from_numpy(self.xanes_data)

            xanes = xanes.float()
            pred_xyz = model(xanes)

            if self.pred_eval:
                Predict.print_mse("xyz", self.xyz_data, pred_xyz)

            self.predict_dim(pred_xyz)

            return pred_xyz, self.xyz_data

        elif self.pred_mode == "predict_xanes":
            xyz = torch.from_numpy(self.xyz_data)
            xyz = xyz.float()

            pred_xanes = model(xyz)

            if self.fourier:
                pred_xanes = inverse_fourier_transform_data(pred_xanes)

            if self.pred_eval:
                Predict.print_mse("xanes", self.xanes_data, pred_xanes)

            self.predict_dim(pred_xanes)

            return pred_xanes

    def predict_bootstrap(self, model_list):
        predict_score = []
        predict_all = []

        for model in model_list:
            model.eval()
            if self.pred_mode == "predict_xyz":
                pred_xyz = self.predict(model)

                if self.pred_eval:
                    mse = mean_squared_error(self.xyz_data, pred_xyz.detach().numpy())
                    predict_score.append(mse)

                predict_all.append(pred_xyz.detach().numpy())

            elif self.pred_mode == "predict_xanes":
                pred_xanes = self.predict(model)
                if self.pred_eval:
                    mse = mean_squared_error(
                        self.xyz_xanes, pred_xanes.detach().numpy()
                    )
                    predict_score.append(mse)
                predict_all.append(pred_xanes.detach().numpy())

        if self.pred_eval:
            mean_score = torch.mean(torch.tensor(predict_score))
            std_score = torch.std(torch.tensor(predict_score))
            print(f"Mean score: {mean_score:.4f}, Std score: {std_score:.4f}")

            predict_all = np.asarray(predict_all)
            mean_predict = np.mean(predict_all, axis=0)
            std_predict = np.std(predict_all, axis=0)

        return mean_predict, std_predict

    def predict_ensemble(self, model_list):
        ensemble_preds = []

        for model in model_list:
            if self.pred_mode == "predict_xyz":
                pred_xyz = self.predict(model)
                ensemble_preds.append(pred_xyz.detach().numpy())
                y_data = self.xyz_data

            elif self.pred_mode == "predict_xanes":
                pred_xanes = self.predict(model)
                ensemble_preds.append(pred_xanes.detach().numpy())
                y_data = self.xanes_data

        ensemble_pred = sum(ensemble_preds) / len(ensemble_preds)

        if self.pred_eval:
            mse = mean_squared_error(y_data, ensemble_pred)

        ensemble_preds = torch.tensor(np.asarray(ensemble_preds)).float()
        mean_ensemble_pred = torch.mean(ensemble_preds, dim=0)
        std_ensemble_pred = torch.std(ensemble_preds, dim=0)

        return std_ensemble_pred, mean_ensemble_pred
