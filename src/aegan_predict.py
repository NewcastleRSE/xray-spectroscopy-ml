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
from pathlib import Path

import numpy as np
import torch
import tqdm as tqdm
from sklearn.metrics import mean_squared_error

import data_transform
from inout import save_xanes
from spectrum.xanes import XANES
from utils import unique_path


def predict_aegan(xyz_path, xanes_path, x, y, model, fourier_transform):
    if xyz_path is not None:
        x_recon = model.reconstruct_structure(x).detach().numpy()
        y_pred = model.predict_spectrum(x)
        print(
            f">> Reconstruction error (structure) = {mean_squared_error(x,x_recon):.4f}"
        )
        if fourier_transform:
            y_pred = data_transform.inverse_fourier_transform_data(y_pred)

        y_pred = y_pred.detach().numpy()

    if xanes_path is not None:
        if fourier_transform:
            z = data_transform.fourier_transform_data(y)
            z = torch.tensor(z).float()
            y_recon = model.reconstruct_spectrum(z)
            y_recon = (
                data_transform.inverse_fourier_transform_data(
                    y_recon).detach().numpy()
            )
            x_pred = model.predict_structure(z).detach().numpy()
        else:
            y_recon = model.reconstruct_spectrum(y).detach().numpy()
            x_pred = model.predict_structure(y).detach().numpy()

        print(
            f">> Reconstruction error (spectrum) =  {mean_squared_error(y,y_recon):.4f}"
        )

    if xyz_path is None:
        x_recon = None
        y_pred = None

    if xanes_path is None:
        y_recon = None
        x_pred = None

    if xyz_path is not None and xanes_path is not None:  # Get prediction errors
        print(
            f">> Prediction error (structure) =     {mean_squared_error(x,x_pred):.4f}"
        )
        print(
            f">> Prediction error (spectrum) =      {mean_squared_error(y,y_pred):.4f}"
        )

    print(">> ...done!\n")

    return x_recon, y_pred, y_recon, x_pred


def main(
    config,
    x,
    y,
    model,
    fourier_transform,
    model_dir,
    predict_dir,
    ids,
    parent_model_dir,
    e,
):
    print(">> Reconstructing and predicting data with neural net...")

    x_recon, y_pred, y_recon, x_pred = predict_aegan(
        config["x_path"], config["y_path"], x, y, model, fourier_transform
    )

    print(">> Saving predictions and reconstructions...")

    if config["x_path"] is not None:

        for id_, y_pred_ in tqdm.tqdm(zip(ids, y_pred)):
            with open(predict_dir / f"spectrum_{id_}.txt", "w") as f:
                save_xanes(f, XANES(e.flatten(), y_pred_))

    # TODO: save structure in .xyz format?
    if config["y_path"] is not None:
        for id_, x_pred_ in tqdm.tqdm(zip(ids, x_pred)):
            with open(predict_dir / f"structure_{id_}.txt", "w") as f:
                np.savetxt(f, x_pred_)

    print(">> ...done!\n")

    if config["plot_save"]:
        print(">> Plotting reconstructions and predictions...")
        plots_dir = unique_path(Path(parent_model_dir), "plots_predictions")
        plots_dir.mkdir()

        if config["x_path"] is not None and config["y_path"] is not None:
            from plot import plot_aegan_predict

            plot_aegan_predict(ids, x, y, x_recon, y_recon,
                               x_pred, y_pred, plots_dir)

        elif config["x_path"] is not None:
            from plot import plot_aegan_spectrum

            plot_aegan_spectrum(ids, x, x_recon, y_pred, plots_dir)

        elif config["y_path"] is not None:
            from plot import plot_aegan_structure

            plot_aegan_structure(ids, y, y_recon, x_pred, plots_dir)

        if config["x_path"] is not None and config["y_path"] is not None:
            print(">> Plotting and saving cosine-similarity...")

            analysis_dir = unique_path(Path(parent_model_dir), "analysis")
            analysis_dir.mkdir()

            from plot import plot_cosine_similarity

            plot_cosine_similarity(
                x, y, x_recon, y_recon, x_pred, y_pred, analysis_dir)

            print("...saved!\n")

    return x_recon, y_pred, y_recon, x_pred
