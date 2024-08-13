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
from torch_geometric.data import DataLoader

from sklearn.preprocessing import StandardScaler

from xanesnet.scheme.base_predict import Predict
from xanesnet.data_transform import fourier_transform, inverse_fourier_transform


@dataclass
class Result:
    xyz_pred: (np.ndarray, np.ndarray)
    xanes_pred: (np.ndarray, np.ndarray)


class GNNPredict(Predict):
    def predict(self, model) -> np.ndarray:
        model.eval()

        # Model prediction
        dataloader = DataLoader(self.xyz_data, batch_size=1, shuffle=False)
        xanes_pred = []
        for data in dataloader:
            out = model(
                data.x.float(), data.edge_attr.float(), data.edge_index, data.batch
            )
            out = torch.squeeze(out)
            xanes_pred.append(out.detach().numpy())

        if self.fourier:
            xanes_pred = inverse_fourier_transform(xanes_pred, self.fourier_concat)

        # Print MSE if evaluation data is provided
        if self.pred_eval:
            Predict.print_mse("xanes", "xanes prediction", self.xanes_data, xanes_pred)

        return xanes_pred

    def predict_std(self, model):
        print(f">> Predicting ...")
        xanes_pred = self.predict(model)

        # Create dummy STD
        xanes_std = np.zeros_like(xanes_pred)

        return Result(xyz_pred=(None, None), xanes_pred=(xanes_pred, xanes_std))

    def predict_bootstrap(self, model_list):
        pass

    def predict_ensemble(self, model_list):
        pass
