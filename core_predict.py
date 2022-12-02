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


def y_predict_dim(y_predict, ids, model_dir):
    if y_predict.ndim == 1:
        if len(ids) == 1:
            y_predict = y_predict.reshape(-1, y_predict.size)
        else:
            y_predict = y_predict.reshape(y_predict.size, -1)
    print(">> ...predicted Y data!\n")

    with open(model_dir / "dataset.npz", "rb") as f:
        e = np.load(f)["e"]

    return y_predict, e

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################


def main(mode: str, model_mode: str, model_dir: str, x_path: str, y_path: str):
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

    xyz_path = Path(x_path)
    xanes_path = Path(y_path)

    predict_dir = unique_path(Path("."), "predictions")
    predict_dir.mkdir()

    ids = list(set(list_filestems(xyz_path)) & set(list_filestems(xanes_path)))

    ids.sort()

    with open(model_dir / "descriptor.pickle", "rb") as f:
        descriptor = pickle.load(f)

    n_samples = len(ids)
    n_x_features = descriptor.get_len()
    n_y_features = linecount(xanes_path / f"{ids[0]}.txt") - 2

    xyz_data = np.full((n_samples, n_x_features), np.nan)
    print(">> preallocated {}x{} array for X data...".format(*xyz_data.shape))
    xanes_data = np.full((n_samples, n_y_features), np.nan)
    print(">> preallocated {}x{} array for Y data...".format(*xanes_data.shape))
    print(">> ...everything preallocated!\n")

    print(">> loading data into array(s)...")
    for i, id_ in enumerate(tqdm.tqdm(ids)):

        with open(xyz_path / f"{id_}.xyz", "r") as f:
            atoms = load_xyz(f)
        xyz_data[i, :] = descriptor.transform(atoms)
        with open(xanes_path / f"{id_}.txt", "r") as f:
            xanes = load_xanes(f)
        e, xanes_data[i, :] = xanes.spectrum
    print(">> ...loaded!\n")

    model = torch.load(model_dir / "model.pt", map_location=torch.device("cpu"))
    model.eval()
    print("Loaded model from disk")

    if model_mode == 'mlp' or model_mode == 'cnn':
        
        if mode == "predict_xyz":

            print("predict xyz structure")

            xanes = torch.from_numpy(xanes_data)
            xanes = xanes.float()

            pred_xyz = model(xanes)

            y = xyz_data
            y_predict = pred_xyz

        elif mode == "predict_xanes":

            print("predict xanes structure")

            xyz = torch.from_numpy(xyz_data)
            xyz = xyz.float()

            pred_xanes = model(xyz)

            y = xanes_data
            y_predict = pred_xanes

        print("MSE y to y pred : ", mean_squared_error(y, y_predict.detach().numpy()))
        
        y_predict, e = y_predict_dim(y_predict, ids, model_dir)
        from plot import plot_predict
        plot_predict(ids, y, y_predict, e, predict_dir, mode)

    
    elif model_mode == 'ae_mlp' or model_mode == 'ae_cnn':

        if mode == "predict_xyz":

            print("predict xyz structure")

            xanes = torch.from_numpy(xanes_data)
            xanes = xanes.float()

            recon_xanes, pred_xyz = model(xanes)

            x = xanes
            x_recon = recon_xanes
            y = xyz_data
            y_predict = pred_xyz

        elif mode == "predict_xanes":

            print("predict xyz structure")

            xyz = torch.from_numpy(xyz_data)
            xyz = xyz.float()

            recon_xyz, pred_xanes = model(xyz)

            x = xyz
            x_recon = recon_xyz
            y = xanes_data
            y_predict = pred_xanes

        print("MSE x to x recon : ", mean_squared_error(x, x_recon.detach().numpy()))
        print("MSE y to y pred : ", mean_squared_error(y, y_predict.detach().numpy()))

        y_predict, e = y_predict_dim(y_predict, ids, model_dir)
        from plot import plot_ae_predict
        plot_ae_predict(ids, y, y_predict, x, x_recon, e, predict_dir, mode)

    print("...saved!\n")

    return 0
