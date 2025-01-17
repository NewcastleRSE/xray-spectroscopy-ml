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

from xanesnet.scheme.base_predict import Predict
from xanesnet.data_transform import inverse_fourier_transform


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
            # reshape concatenated graph_attr to [batch_size, feat_size]
            nfeats = data[0].graph_attr.shape[0]
            graph_attr = data.graph_attr.reshape(len(data), nfeats)
            out = model(
                data.x.float(),
                data.edge_attr.float(),
                graph_attr.float(),
                data.edge_index,
                data.batch,
            )
            out = torch.squeeze(out)
            xanes_pred.append(out.detach().numpy())

        xanes_pred = np.array(xanes_pred)

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
        predict_score = []
        xanes_pred_list = []

        # Iterate over models to perform predicting
        for i, model in enumerate(model_list, start=1):
            print(f">> Predicting with model {i}...")
            xanes_pred = self.predict(model)
            
            if self.pred_eval:
                mse = mean_squared_error(self.xanes_data, xanes_pred)
                predict_score.append(mse)
                
            xanes_pred_list.append(xanes_pred)

        # Print MSE if evaluation data is provided
        if self.pred_eval and len(predict_score) > 0:
            mean_score = np.mean(predict_score)
            std_score = np.std(predict_score)
            print(f"Mean score prediction: {mean_score:.4f}, Std: {std_score:.4f}")

        # Calculate mean and standard deviation across all predictions
        xanes_mean = np.mean(xanes_pred_list, axis=0)
        xanes_std = np.std(xanes_pred_list, axis=0)

        return Result(xyz_pred=(None, None), xanes_pred=(xanes_mean, xanes_std))

    def predict_ensemble(self, model_list):
        xanes_pred_list = []

        # Iterate over models to perform predicting
        for i, model in enumerate(model_list, start=1):
            print(f">> Predicting with model {i}...")
            xanes_pred = self.predict(model)    
            xanes_pred_list.append(xanes_pred)

        # Print MSE summary
        print(f"{'='*30}Ensemble Prediction Summary{'='*30}")
        xanes_pred = sum(xanes_pred_list) / len(xanes_pred_list)
        Predict.print_mse("xanes", "xanes prediction", self.xanes_data, xanes_pred)

        xanes_mean = np.mean(xanes_pred_list, axis=0)
        xanes_std = np.std(xanes_pred_list, axis=0)

        return Result(xyz_pred=(None, None), xanes_pred=(xanes_mean, xanes_std))