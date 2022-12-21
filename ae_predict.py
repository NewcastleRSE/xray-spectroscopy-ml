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
from glob import glob

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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################


def main(aemode: str, model_dir: str, x_path: str, y_path: str):
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

    if aemode == "predict_xyz":

        print("predict xyz structure")

        xanes = torch.from_numpy(xanes_data)
        xanes = xanes.float()

        recon_xanes, pred_xyz = model(xanes)

        x = xanes
        x_recon = recon_xanes
        y = xyz_data
        y_pred = pred_xyz

    elif aemode == "predict_xanes":

        print("predict xyz structure")

        xyz = torch.from_numpy(xyz_data)
        xyz = xyz.float()

        recon_xyz, pred_xanes = model(xyz)

        x = xyz
        x_recon = recon_xyz
        y = xanes_data
        y_pred = pred_xanes

    print("MSE x to x recon : ", mean_squared_error(x, x_recon.detach().numpy()))
    print("MSE y to y pred : ", mean_squared_error(y, y_pred.detach().numpy()))

    total_y = []
    total_y_pred = []
    total_x = []
    total_x_recon = []

    for id_, y_predict_, y_, x_recon_, x_ in tqdm.tqdm(zip(ids, y_pred, y, x_recon, x)):
        sns.set()
        fig, (ax1, ax2) = plt.subplots(2)

        ax1.plot(y_predict_.detach().numpy(), label="prediction")
        ax1.set_title("prediction")
        ax1.plot(y_, label="target")
        ax1.legend(loc="upper right")

        ax2.plot(x_recon_.detach().numpy(), label="prediction")
        ax2.set_title("reconstruction")
        ax2.plot(x_, label="target")
        ax2.legend(loc="upper right")
        # print(type(x_))
        total_y.append(y_)
        total_y_pred.append(y_predict_.detach().numpy())

        total_x.append(x_.detach().numpy())
        total_x_recon.append(x_recon_.detach().numpy())
        # with open(predict_dir / f"{id_}.txt", "w") as f:
        np.save(
            predict_dir / f"{id_}.npy",
            y_predict_.detach().numpy(),
        )
        plt.savefig(predict_dir / f"{id_}.pdf")
        fig.clf()
        plt.close(fig)

    total_y = np.asarray(total_y)
    total_y_pred = np.asarray(total_y_pred)
    total_x = np.asarray(total_x)
    total_x_recon = np.asarray(total_x_recon)

    # plotting the average loss
    sns.set_style("dark")
    fig, (ax1, ax2) = plt.subplots(2)

    mean_y = np.mean(total_y, axis=0)
    stddev_y = np.std(total_y, axis=0)

    ax1.plot(mean_y, label="target")
    ax1.fill_between(
        np.arange(mean_y.shape[0]),
        mean_y + stddev_y,
        mean_y - stddev_y,
        alpha=0.4,
        linewidth=0,
    )

    mean_y_pred = np.mean(total_y_pred, axis=0)
    stddev_y_pred = np.std(total_y_pred, axis=0)

    ax1.plot(mean_y_pred, label="prediction")
    ax1.fill_between(
        np.arange(mean_y_pred.shape[0]),
        mean_y_pred + stddev_y_pred,
        mean_y_pred - stddev_y_pred,
        alpha=0.4,
        linewidth=0,
    )

    ax1.legend(loc="best")
    ax1.grid()

    mean_x = np.mean(total_x, axis=0)
    stddev_x = np.std(total_x, axis=0)

    ax2.plot(mean_x, label="target")
    ax2.fill_between(
        np.arange(mean_x.shape[0]),
        mean_x + stddev_x,
        mean_x - stddev_x,
        alpha=0.4,
        linewidth=0,
    )

    mean_x = np.mean(total_x_recon, axis=0)
    stddev_x = np.std(total_x_recon, axis=0)

    ax2.plot(mean_x, label="reconstruction")
    ax2.fill_between(
        np.arange(mean_x.shape[0]),
        mean_x + stddev_x,
        mean_x - stddev_x,
        alpha=0.4,
        linewidth=0,
    )

    ax2.legend(loc="best")
    ax2.grid()

    plt.savefig(predict_dir / "plot.pdf")

    plt.show()
    fig.clf()
    plt.close(fig)

    print("...saved!\n")

    return 0
