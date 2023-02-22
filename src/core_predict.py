"""
XANESNET
Copyright (C) 2021  Conor D. Rankine

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

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import os
import numpy as np
import pickle as pickle
import tqdm as tqdm

from pathlib import Path

from inout import load_xyz
from inout import load_xanes
from inout import save_xanes
from utils import unique_path
from utils import list_filestems
from utils import linecount
from structure.rdc import RDC
from structure.wacsf import WACSF
from spectrum.xanes import XANES

import torch
from sklearn.metrics import mean_squared_error

from model_utils import run_shap_analysis
from model_utils import model_mode_error
import plot


def average(lst):
    for lstNum in range(len(lst)):
        print(lstNum)
        for sublistItem in range(len(lst[lstNum])):
            lst[lstNum] / lst[sublistItem]  # <-- ??
    print(type(lst))
    return lst


def y_predict_dim(y_predict, ids, model_dir):
    if y_predict.ndim == 1:
        if len(ids) == 1:
            y_predict = y_predict.reshape(-1, y_predict.size)
        else:
            y_predict = y_predict.reshape(y_predict.size, -1)
    print(">> ...predicted Y data!\n")

    with open(model_dir / "dataset.npz", "rb") as f:
        e = np.load(f)["e"]

    return y_predict, e.flatten()


def predict_xyz(xanes_data, model):
    print("predict xyz structure")
    xanes = torch.from_numpy(xanes_data)
    xanes = xanes.float()

    pred_xyz = model(xanes)

    return pred_xyz


def predict_xanes(xyz_data, model):
    print("predict xanes spectrum")
    xyz = torch.from_numpy(xyz_data)
    xyz = xyz.float()

    pred_xanes = model(xyz)

    return pred_xanes


###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################


def main(
    mode: str,
    model_mode: str,
    run_shap: bool,
    shap_nsamples: int,
    model_dir: str,
    x_path: str,
    y_path: str,
    monte_carlo: dict = {},
    bootstrap: dict = {},
):
    """
    PREDICT. The model state is restored from a model directory containing
    serialised scaling/pipeline objects and the serialised model, .xyz (X)
    data are loaded and transformed, and the model is used to predict XANES
    spectral (Y) data; convolution of the Y data is also possible if
    {conv_params} are provided (see xanesnet/convolute.py).

    Args:
        model_dir (str): The path to a model.[?] directory created by
            the LEARN routine.
        x_path (str): The path to the .xyz (X) data; expects a directory
            containing .xyz files.
    """

    model_dir = Path(model_dir)

    xyz_path = Path(x_path) if x_path is not None else None
    xanes_path = Path(y_path) if y_path is not None else None

    if xyz_path is not None and xanes_path is not None:
        ids = list(set(list_filestems(xyz_path)) & set(list_filestems(xanes_path)))
    elif xyz_path is None:
        ids = list(set(list_filestems(xanes_path)))
    elif xanes_path is None:
        ids = list(set(list_filestems(xyz_path)))

    ids.sort()

    with open(model_dir / "descriptor.pickle", "rb") as f:
        descriptor = pickle.load(f)

    n_samples = len(ids)

    if xyz_path is not None:
        n_x_features = descriptor.get_len()
        xyz_data = np.full((n_samples, n_x_features), np.nan)
        print(">> preallocated {}x{} array for X data...".format(*xyz_data.shape))

    if xanes_path is not None:
        n_y_features = linecount(xanes_path / f"{ids[0]}.txt") - 2
        xanes_data = np.full((n_samples, n_y_features), np.nan)
        print(">> preallocated {}y{} array for Y data...".format(*xanes_data.shape))

    print(">> ...everything preallocated!\n")

    print(">> loading data into array(s)...")
    if xyz_path is not None:
        for i, id_ in enumerate(tqdm.tqdm(ids)):
            with open(xyz_path / f"{id_}.xyz", "r") as f:
                atoms = load_xyz(f)
            xyz_data[i, :] = descriptor.transform(atoms)

    if xanes_path is not None:
        for i, id_ in enumerate(tqdm.tqdm(ids)):
            with open(xanes_path / f"{id_}.txt", "r") as f:
                xanes = load_xanes(f)
                e, xanes_data[i, :] = xanes.spectrum

    print(">> ...loaded!\n")

    if bootstrap["fn"] == "True":
        n_boot = len(next(os.walk(model_dir))[1])
        # print(n_boot)
        for i in range(n_boot):
            n_dir = f"{model_dir}/model_00{i+1}/model.pt"

            model = torch.load(n_dir, map_location=torch.device("cpu"))
            model.eval()
            print("Loaded model from disk")

            parent_model_dir, predict_dir = model_mode_error(
                model, mode, model_mode, xyz_data.shape[1], xanes_data.shape[1]
            )

            if model_mode == "mlp" or model_mode == "cnn":
                if mode == "predict_xyz":
                    xyz_predict = predict_xyz(xanes_data, model)

                    x = xanes_data
                    y = xyz_data
                    y_predict = xyz_predict

                elif mode == "predict_xanes":
                    xanes_predict = predict_xanes(xyz_data, model)

                    x = xyz_data
                    y = xanes_data
                    y_predict = xanes_predict

                print(
                    "MSE y to y pred : ",
                    mean_squared_error(y, y_predict.detach().numpy()),
                )
                y_predict, e = y_predict_dim(y_predict, ids, model_dir)
                plot.plot_predict(ids, y, y_predict, e, predict_dir, mode)

            elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
                if mode == "predict_xyz":
                    recon_xanes, pred_xyz = predict_xyz(xanes_data, model)

                    x = xanes_data
                    x_recon = recon_xanes
                    y = xyz_data
                    y_predict = pred_xyz

                elif mode == "predict_xanes":
                    recon_xyz, pred_xanes = predict_xanes(xyz_data, model)

                    x = xyz_data
                    x_recon = recon_xyz
                    y = xanes_data
                    y_predict = pred_xanes

                print(
                    "MSE x to x recon : ",
                    mean_squared_error(x, x_recon.detach().numpy()),
                )
                print(
                    "MSE y to y pred : ",
                    mean_squared_error(y, y_predict.detach().numpy()),
                )

                from plot import plot_ae_predict

                plot_ae_predict(ids, y, y_predict, x, x_recon, e, predict_dir, mode)

    else:
        model = torch.load(model_dir / "model.pt", map_location=torch.device("cpu"))
        model.eval()
        print("Loaded model from disk")

        if xyz_path is not None and xanes_path is not None:
            from model_utils import model_mode_error

            parent_model_dir, predict_dir = model_mode_error(
                model, mode, model_mode, xyz_data.shape[1], xanes_data.shape[1]
            )
        else:
            from model_utils import make_dir

            parent_model_dir, predict_dir = make_dir()

        if model_mode == "mlp" or model_mode == "cnn":
            if mode == "predict_xyz":
                xyz_predict = predict_xyz(xanes_data, model)

                x = xanes_data
                y = xyz_data
                y_predict = xyz_predict

            elif mode == "predict_xanes":
                xanes_predict = predict_xanes(xyz_data, model)

                x = xyz_data
                y = xanes_data
                y_predict = xanes_predict

            print(
                "MSE y to y pred : ",
                mean_squared_error(y, y_predict.detach().numpy()),
            )
            y_predict, e = y_predict_dim(y_predict, ids, model_dir)

            if monte_carlo["mc_fn"] == "True":
                from model_utils import montecarlo_dropout
                from plot import plot_mc_predict

                prob_mean, prob_var = montecarlo_dropout(
                    model, x, monte_carlo["mc_iter"]
                )
                print(
                    "MSE y to y prob : ",
                    mean_squared_error(y, prob_mean.detach().numpy()),
                )
                # confidence interval
                plot_mc_predict(
                    ids,
                    y,
                    y_predict,
                    prob_mean.detach().numpy(),
                    prob_var.detach().numpy(),
                    e,
                    predict_dir,
                    mode,
                )
            else:
                from plot import plot_predict

                plot_predict(ids, y, y_predict, e, predict_dir, mode)

        elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
            if mode == "predict_xyz":
                recon_xanes, pred_xyz = predict_xyz(xanes_data, model)

                x = xanes_data
                x_recon = recon_xanes
                y = xyz_data
                y_predict = pred_xyz

            elif mode == "predict_xanes":
                recon_xyz, pred_xanes = predict_xanes(xyz_data, model)

                x = xyz_data
                x_recon = recon_xyz
                y = xanes_data
                y_predict = pred_xanes

            print(
                "MSE x to x recon : ",
                mean_squared_error(x, x_recon.detach().numpy()),
            )
            print(
                "MSE y to y pred : ",
                mean_squared_error(y, y_predict.detach().numpy()),
            )

            if monte_carlo["mc_fn"] == "True":
                from model_utils import montecarlo_dropout_ae
                from plot import plot_mc_ae_predict

                mean_output, var_output, mean_recon, var_recon = montecarlo_dropout_ae(
                    model, x, monte_carlo["mc_iter"]
                )
                print(
                    "MSE x to x prob : ",
                    mean_squared_error(x, mean_recon.detach().numpy()),
                )
                print(
                    "MSE y to y prob : ",
                    mean_squared_error(y, mean_output.detach().numpy()),
                )
                # confidence interval

                plot_mc_ae_predict(
                    ids,
                    y,
                    y_predict,
                    x,
                    x_recon,
                    mean_output.detach().numpy(),
                    var_output.detach().numpy(),
                    mean_recon.detach().numpy(),
                    var_recon.detach().numpy(),
                    e,
                    predict_dir,
                    mode,
                )
            else:
                y_predict, e = y_predict_dim(y_predict, ids, model_dir)
                from plot import plot_ae_predict

                plot_ae_predict(ids, y, y_predict, x, x_recon, e, predict_dir, mode)

        elif model_mode == "aegan_mlp" or model_mode == "aegan_cnn":
            # Convert to float
            if xyz_path is not None:
                x = torch.tensor(xyz_data).float()
            if xanes_path is not None:
                y = torch.tensor(xanes_data).float()

            print(">> Reconstructing and predicting data with neural net...")

            if xyz_path is not None:
                x_recon = model.reconstruct_structure(x).detach().numpy()
                y_pred = model.predict_spectrum(x).detach().numpy()
                print(
                    f">> Reconstruction error (structure) = {mean_squared_error(x,x_recon):.4f}"
                )

            if xanes_path is not None:
                y_recon = model.reconstruct_spectrum(y).detach().numpy()
                x_pred = model.predict_structure(y).detach().numpy()
                print(
                    f">> Reconstruction error (spectrum) =  {mean_squared_error(y,y_recon):.4f}"
                )

            if xyz_path is not None and xanes_path is not None:  # Get prediction errors
                print(
                    f">> Prediction error (structure) =     {mean_squared_error(x,x_pred):.4f}"
                )
                print(
                    f">> Prediction error (spectrum) =      {mean_squared_error(y,y_pred):.4f}"
                )

            print(">> ...done!\n")

            print(">> Saving predictions and reconstructions...")

            if xyz_path is not None:
                with open(model_dir / "dataset.npz", "rb") as f:
                    e = np.load(f)["e"]

                for id_, y_pred_ in tqdm.tqdm(zip(ids, y_pred)):
                    with open(predict_dir / f"spectrum_{id_}.txt", "w") as f:
                        save_xanes(f, XANES(e.flatten(), y_pred_))

            # TODO: save structure in .xyz format?
            if xanes_path is not None:
                for id_, x_pred_ in tqdm.tqdm(zip(ids, x_pred)):
                    with open(predict_dir / f"structure_{id_}.txt", "w") as f:
                        np.savetxt(f, x_pred_)

            print(">> ...done!\n")

            print(">> Plotting reconstructions and predictions...")

            plots_dir = unique_path(Path(parent_model_dir), "plots_predictions")
            plots_dir.mkdir()

            if xyz_path is not None and xanes_path is not None:
                from plot import plot_aegan_predict

                plot_aegan_predict(
                    ids, x, y, x_recon, y_recon, x_pred, y_pred, plots_dir
                )

            elif x_path is not None:
                from plot import plot_aegan_spectrum

                plot_aegan_spectrum(ids, x, x_recon, y_pred, plots_dir)

            elif y_path is not None:
                from plot import plot_aegan_structure

                plot_aegan_structure(ids, y, y_recon, x_pred, plots_dir)

            if x_path is not None and y_path is not None:
                print(">> Plotting and saving cosine-similarity...")

                analysis_dir = unique_path(Path(parent_model_dir), "analysis")
                analysis_dir.mkdir()

                from plot import plot_cosine_similarity

                plot_cosine_similarity(
                    x, y, x_recon, y_recon, x_pred, y_pred, analysis_dir
                )

                print("...saved!\n")

        if run_shap:
            if model_mode == "mlp" or model_mode == "cnn":
                if mode == "predict_xanes":
                    data = xyz_data

                elif mode == "predict_xyz":
                    data = xanes_data

                data = torch.from_numpy(data).float()

                print(">> Performing SHAP analysis on predicted data...")
                run_shap_analysis(model, predict_dir, data, ids, shap_nsamples)

            elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
                if mode == "predict_xanes":
                    # Redefine forward function
                    print(">> Performing SHAP analysis on predicted data...")
                    model.forward = model.predict
                    data = xyz_data
                    data = torch.from_numpy(data).float()
                    run_shap_analysis(
                        model,
                        predict_dir,
                        data,
                        ids,
                        shap_nsamples,
                        shap_mode="predict",
                    )

                    print(">> Performing SHAP analysis on reconstructed data...")
                    model.forward = model.reconstruct
                    data = xyz_data
                    data = torch.from_numpy(data).float()
                    run_shap_analysis(
                        model,
                        predict_dir,
                        data,
                        ids,
                        shap_nsamples,
                        shap_mode="reconstruct",
                    )

                elif mode == "predict_xyz":
                    print(">> Performing SHAP analysis on predicted data...")
                    model.forward = model.predict
                    data = xanes_data
                    data = torch.from_numpy(data).float()
                    run_shap_analysis(
                        model,
                        predict_dir,
                        data,
                        ids,
                        shap_nsamples,
                        shap_mode="predict",
                    )

                    print(">> Performing SHAP analysis on reconstructed data...")
                    model.foward = model.reconstruct
                    data = xanes_data
                    data = torch.from_numpy(data).float()
                    run_shap_analysis(
                        model,
                        predict_dir,
                        data,
                        ids,
                        shap_nsamples,
                        shap_mode="reconstruct",
                    )

            elif model_mode == "aegan_mlp" or model_mode == "aegan_cnn":
                if mode == "predict_xanes":
                    print(">> Performing SHAP analysis on predicted data...")
                    model.forward = model.predict_spectrum
                    data = xyz_data
                    data = torch.from_numpy(data).float()
                    run_shap_analysis(
                        model,
                        predict_dir,
                        data,
                        ids,
                        shap_nsamples,
                        shap_mode="predict",
                    )

                    print(">> Performing SHAP analysis on reconstructed data...")
                    model.forward = model.reconstruct_structure
                    data = xyz_data
                    data = torch.from_numpy(data).float()
                    run_shap_analysis(
                        model,
                        predict_dir,
                        data,
                        ids,
                        shap_nsamples,
                        shap_mode="reconstruct",
                    )

                elif mode == "predict_xyz":
                    print(">> Performing SHAP analysis on predicted data...")
                    model.forward = model.predict_structure
                    data = xanes_data
                    data = torch.from_numpy(data).float()
                    run_shap_analysis(
                        model,
                        predict_dir,
                        data,
                        ids,
                        shap_nsamples,
                        shap_mode="predict",
                    )

                    print(">> Performing SHAP analysis on reconstructed data...")
                    model.forward = model.reconstruct_spectrum
                    data = xanes_data
                    data = torch.from_numpy(data).float()
                    run_shap_analysis(
                        model,
                        predict_dir,
                        data,
                        ids,
                        shap_nsamples,
                        shap_mode="reconstruct",
                    )
