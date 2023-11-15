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

import torch
from sklearn.metrics import mean_squared_error

from xanesnet.scheme.base_predict import Predict
from xanesnet.data_transform import (
    fourier_transform_data,
    inverse_fourier_transform_data,
)


class AEGANPredict(Predict):
    def predict(self, model):
        xyz_recon = None
        xanes_pred = None
        xyz_pred = None
        xanes_recon = None

        if self.pred_mode == "predict_xyz":
            xanes = torch.tensor(self.xanes_data).float()

            if self.fourier:
                xanes_fourier = fourier_transform_data(xanes)
                xanes_fourier = torch.tensor(xanes_fourier).float()
                xanes_recon = model.reconstruct_spectrum(xanes_fourier)
                xanes_recon = inverse_fourier_transform_data(xanes_recon)
                xyz_pred = model.predict_structure(xanes_fourier)

            else:
                xanes_recon = model.reconstruct_spectrum(xanes)
                xyz_pred = model.predict_structure(xanes)

        elif self.pred_mode == "predict_xanes":
            xyz = torch.tensor(self.xyz_data).float()

            xyz_recon = model.reconstruct_structure(xyz)
            xanes_pred = model.predict_spectrum(xyz)

            if self.fourier:
                xanes_pred = inverse_fourier_transform_data(xanes_pred)

        elif self.pred_mode == "predict_all":
            xyz = torch.tensor(self.xyz_data).float()
            xanes = torch.tensor(self.xanes_data).float()

            xyz_recon = model.reconstruct_structure(xyz)
            xanes_pred = model.predict_spectrum(xyz)

            if self.fourier:
                # xyz -> xanes
                xanes_pred = inverse_fourier_transform_data(xanes_pred)

                # xanes -> xyz
                xanes_fourier = fourier_transform_data(xanes)
                xanes_fourier = torch.tensor(xanes_fourier).float()
                xanes_recon = model.reconstruct_spectrum(xanes_fourier)
                xanes_recon = inverse_fourier_transform_data(xanes_recon)
                xyz_pred = model.predict_structure(xanes_fourier)

            else:
                xanes_recon = model.reconstruct_spectrum(xanes)
                xyz_pred = model.predict_structure(xanes)

        return xyz_recon, xanes_pred, xanes_recon, xyz_pred

    def predict_bootstrap(self, model_list):
        xyz_pred_score = []
        xyz_recon_score = []
        xanes_pred_score = []
        xanes_recon_score = []

        for model in model_list:
            model.eval()
            xyz_recon, xanes_pred, xanes_recon, xyz_pred = self.predict(model)

            if self.pred_mode == "predict_xyz" or self.pred_mode == "predict_all":
                mse1 = mean_squared_error(self.xanes_data, xanes_recon.detach().numpy())
                mse2 = mean_squared_error(self.xyz_data, xyz_pred.detach().numpy())

                xanes_recon_score.append(mse1)
                xyz_pred_score.append(mse2)

            elif self.pred_mode == "predict_xanes" or self.pred_mode == "predict_all":
                mse1 = mean_squared_error(self.xyz_data, xyz_recon.detach().numpy())
                mse2 = mean_squared_error(self.xanes_data, xanes_pred.detach().numpy())

                xyz_recon_score.append(mse1)
                xanes_pred_score.append(mse2)

        return xyz_pred_score, xyz_recon_score, xanes_pred_score, xanes_recon_score

    def predict_ensemble(self, model_list):
        xyz_pred_list = []
        xyz_recon_list = []
        xanes_pred_list = []
        xanes_recon_list = []
        xyz_recon = None
        xyz_pred = None
        xanes_pred = None
        xanes_recon = None

        for model in model_list:
            model.eval()
            xyz_recon, xanes_pred, xanes_recon, xyz_pred = self.predict(model)

            if self.pred_mode == "predict_xyz" or self.pred_mode == "predict_all":
                xanes_recon_list.append(xanes_recon)
                xyz_pred_list.append(xyz_pred)

            elif self.pred_mode == "predict_xanes" or self.pred_mode == "predict_all":
                xyz_recon_list.append(xyz_recon)
                xanes_pred_list.append(xanes_pred)

        if len(xyz_recon_list) > 0:
            xyz_recon = sum(xyz_recon_list) / len(xyz_recon_list)
        if len(xanes_recon_list) > 0:
            xanes_recon = sum(xanes_recon_list) / len(xanes_recon_list)
        if len(xyz_pred_list) > 0:
            xyz_pred = sum(xyz_pred_list) / len(xyz_pred_list)
        if len(xanes_pred_list) > 0:
            xanes_pred = sum(xanes_pred_list) / len(xanes_pred_list)

        if self.pred_mode == "predict_xyz" or self.pred_mode == "predict_all":
            Predict.print_mse("xyz_pred", self.xyz_data, xyz_pred.detach().numpy())
            Predict.print_mse(
                "xanes_recon", self.xanes_data, xanes_recon.detach().numpy()
            )
        elif self.pred_mode == "predict_xanes" or self.pred_mode == "predict_all":
            Predict.print_mse("xyz_recon", self.xyz_data, xyz_recon.detach().numpy())
            Predict.print_mse(
                "xanes_pred", self.xanes_data, xanes_pred.detach().numpy()
            )

        return xyz_recon, xanes_recon, xyz_pred, xanes_pred
