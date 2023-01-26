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
from inout import save_xyz
from inout import load_xanes
from inout import save_xanes

# from inout import load_pipeline
# from inout import save_pipeline
from utils import unique_path
from utils import list_filestems
from utils import linecount
from structure.rdc import RDC
from structure.wacsf import WACSF
from spectrum.xanes import XANES

# from tensorflow.keras.models import model_from_json

import torch
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from pyemd import emd_samples

import AEGAN
from sklearn.metrics.pairwise import cosine_similarity

###############################################################################
############################### PREDICT FUNCTION ##############################
###############################################################################


def main(model_dir: str, x_path: str, y_path: str):
    """
    PREDICT ALL. The model state is restored from a model directory containing
    serialised scaling/pipeline objects and the serialised model, input data are
    loaded and transformed, and the model is used to reconstruct and predict
    structural and spectral data.
    Args:
        model_dir (str): The path to a model directory created by
            the LEARN routine.
        x_path (str): The path to the .xyz (X) data; expects a directory
            containing .xyz files.
        y_path (str): The path to the .xanes (Y) data; expects a directory
            containing xanes files.
    """

    model_dir = Path(model_dir)

    x_path = Path(x_path) if x_path is not None else None
    y_path = Path(y_path) if y_path is not None else None

    if x_path is not None and y_path is not None:
        ids = list(set(list_filestems(x_path)) & set(list_filestems(y_path)))
    elif x_path is None:
        ids = list(set(list_filestems(y_path)))
    elif y_path is None:
        ids = list(set(list_filestems(x_path)))

    ids.sort()

    with open(model_dir / "descriptor.pickle", "rb") as f:
        descriptor = pickle.load(f)

    n_samples = len(ids)

    if x_path is not None:
        n_x_features = descriptor.get_len()
        x = np.full((n_samples, n_x_features), np.nan)
        print(">> preallocated {}x{} array for X data...".format(*x.shape))

    if y_path is not None:
        n_y_features = linecount(y_path / f"{ids[0]}.txt") - 2
        y = np.full((n_samples, n_y_features), np.nan)
        print(">> preallocated {}y{} array for Y data...".format(*y.shape))

    print(">> ...everything preallocated!\n")

    print(">> loading data into array(s)...")
    if x_path is not None:
        for i, id_ in enumerate(tqdm.tqdm(ids)):
            with open(x_path / f"{id_}.xyz", "r") as f:
                atoms = load_xyz(f)
            x[i, :] = descriptor.transform(atoms)

    if y_path is not None:
        for i, id_ in enumerate(tqdm.tqdm(ids)):
            with open(y_path / f"{id_}.txt", "r") as f:
                xanes = load_xanes(f)
                e, y[i, :] = xanes.spectrum
    print(">> ...loaded!\n")

    # Convert to float
    if x_path is not None:
        x = torch.tensor(x).float()
    if y_path is not None:
        y = torch.tensor(y).float()

    # load the model
    # model_file = open(model_dir / 'model.pt', 'r')
    model = torch.load(model_dir / "model.pt", map_location=torch.device("cpu"))
    model.eval()
    print("Loaded model from disk")

    print(">> Reconstructing and predicting data with neural net...")

    if x_path is not None:
        x_recon = model.reconstruct_structure(x).detach().numpy()
        y_pred = model.predict_spectrum(x).detach().numpy()
        print(
            f">> Reconstruction error (structure) = {mean_squared_error(x,x_recon):.4f}"
        )

    if y_path is not None:
        y_recon = model.reconstruct_spectrum(y).detach().numpy()
        x_pred = model.predict_structure(y).detach().numpy()
        print(
            f">> Reconstruction error (spectrum) =  {mean_squared_error(y,y_recon):.4f}"
        )

    if x_path is not None and y_path is not None:  # Get prediction errors
        print(
            f">> Prediction error (structure) =     {mean_squared_error(x,x_pred):.4f}"
        )
        print(
            f">> Prediction error (spectrum) =      {mean_squared_error(y,y_pred):.4f}"
        )

    print(">> ...done!\n")

    print(">> Saving predictions and reconstructions...")
    predict_dir = unique_path(Path("."), "predictions")
    predict_dir.mkdir()

    if x_path is not None:
        with open(model_dir / "dataset.npz", "rb") as f:
            e = np.load(f)["e"]
        for id_, y_pred_ in tqdm.tqdm(zip(ids, y_pred)):
            with open(predict_dir / f"spectrum_{id_}.txt", "w") as f:
                save_xanes(f, XANES(e, y_pred_))

    # TODO: save structure in .xyz format?
    if y_path is not None:
        for id_, x_pred_ in tqdm.tqdm(zip(ids, x_pred)):
            with open(predict_dir / f"structure_{id_}.txt", "w") as f:
                np.savetxt(f, x_pred_)

    print(">> ...done!\n")

    print(">> Plotting reconstructions and predictions...")

    plots_dir = unique_path(Path("."), "plots_predictions")
    plots_dir.mkdir()

    if x_path is not None and y_path is not None:
        for id_, x_, y_, x_recon_, y_recon_, x_pred_, y_pred_ in tqdm.tqdm(
            zip(ids, x, y, x_recon, y_recon, x_pred, y_pred)
        ):
            sns.set()
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(20, 20))

            ax1.plot(x_recon_, label="Reconstruction")
            ax1.set_title(f"Structure Reconstruction")
            ax1.plot(x_, label="target")
            ax1.legend(loc="upper left")

            ax2.plot(y_recon_, label="Reconstruction")
            ax2.set_title(f"Spectrum Reconstruction")
            ax2.plot(y_, label="target")
            ax2.legend(loc="upper left")

            ax3.plot(y_pred_, label="Prediction")
            ax3.set_title(f"Spectrum Prediction")
            ax3.plot(y_, label="target")
            ax3.legend(loc="upper left")

            ax4.plot(x_pred_, label="Prediction")
            ax4.set_title(f"Structure Prediction")
            ax4.plot(x_, label="target")
            ax4.legend(loc="upper left")

            plt.savefig(plots_dir / f"{id_}.pdf")
            fig.clf()
            plt.close(fig)
    elif x_path is not None:
        for id_, x_, x_recon_, y_pred_ in tqdm.tqdm(zip(ids, x, x_recon, y_pred)):
            sns.set()
            fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 20))

            ax1.plot(x_recon_, label="Reconstruction")
            ax1.set_title(f"Structure Reconstruction")
            ax1.plot(x_, label="target")
            ax1.legend(loc="upper left")

            ax2.plot(y_pred_, label="Prediction")
            ax2.set_title(f"Spectrum Prediction")
            ax2.legend(loc="upper left")

            plt.savefig(plots_dir / f"{id_}.pdf")
            fig.clf()
            plt.close(fig)
    elif y_path is not None:
        for id_, y_, y_recon_, x_pred_ in tqdm.tqdm(zip(ids, y, y_recon, x_pred)):
            sns.set()
            fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 20))

            ax1.plot(y_recon_, label="Reconstruction")
            ax1.set_title(f"Sprectum Reconstruction")
            ax1.plot(y_, label="target")
            ax1.legend(loc="upper left")

            ax2.plot(x_pred_, label="Prediction")
            ax2.set_title(f"Structure Prediction")
            ax2.legend(loc="upper left")

            plt.savefig(plots_dir / f"{id_}.pdf")
            fig.clf()
            plt.close(fig)

    print("...saved!\n")

    if x_path is not None and y_path is not None:
        print(">> Plotting and saving cosine-similarity...")

        analysis_dir = unique_path(Path("."), "analysis")
        analysis_dir.mkdir()

        cosine_x_x_pred = np.diagonal(cosine_similarity(x, x_pred))
        cosine_y_y_pred = np.diagonal(cosine_similarity(y, y_pred))
        cosine_x_x_recon = np.diagonal(cosine_similarity(x, x_recon))
        cosine_y_y_recon = np.diagonal(cosine_similarity(y, y_recon))

        sns.set()
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 15))
        ax1.plot(cosine_x_x_recon, cosine_y_y_pred, "o", color=cycle[0])
        ax1.set(xlabel="Reconstructed Structure", ylabel="Predicted Spectrum")
        ax2.plot(cosine_y_y_recon, cosine_x_x_pred, "o", color=cycle[1])
        ax2.set(xlabel="Reconstructed Spectrum", ylabel="Predicted Structure")
        ax3.plot(
            cosine_x_x_recon + cosine_y_y_recon,
            cosine_x_x_pred + cosine_y_y_pred,
            "o",
            color=cycle[2],
        )
        ax3.set(xlabel="Reconstruction", ylabel="Prediction")
        plt.savefig(f"{analysis_dir}/cosine_similarity.pdf")
        fig.clf()
        plt.close(fig)

        print("...saved!\n")

    return 0
