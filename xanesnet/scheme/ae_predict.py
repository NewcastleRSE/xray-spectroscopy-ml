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

from xanesnet.scheme.nn_predict import NNPredict
from xanesnet.data_transform import (
    fourier_transform_data,
    inverse_fourier_transform_data,
)


class AEPredict(NNPredict):
    def predict(self):
        if self.pred_mode == "predict_xyz":
            xanes = torch.tensor(self.xanes_data).float()

            if self.fourier:
                fourier_xanes = fourier_transform_data(xanes)
                fourier_xanes = torch.tensor(fourier_xanes).float()
                recon_xanes = self.model.reconstruct(fourier_xanes)
                recon_xanes = inverse_fourier_transform_data(recon_xanes)
                pred_xyz = self.model.predict(fourier_xanes)

            else:
                recon_xanes = self.model.reconstruct(xanes)
                pred_xyz = self.model.predict(xanes)

            recon_xyz = None
            pred_xanes = None

        elif self.pred_mode == "predict_xanes":
            xyz = torch.tensor(self.xyz_data).float()

            recon_xyz = self.model.reconstruct(xyz)
            pred_xanes = self.model.predict(xyz)

            if self.fourier:
                pred_xanes = inverse_fourier_transform_data(pred_xanes)

            recon_xanes = None
            pred_xyz = None

    def predict_bootstrap(self):
        super().predict_bootstrap(self)

    def predict_ensemble(self):
        super().predict_ensemble(self)
