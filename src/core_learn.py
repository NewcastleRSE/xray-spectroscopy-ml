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
import time


from pathlib import Path
from glob import glob
from numpy.random import RandomState
from sklearn.model_selection import RepeatedKFold
from sklearn.utils import shuffle

from inout import load_xyz
from inout import load_xanes

from utils import unique_path
from utils import linecount
from utils import list_filestems
from utils import print_cross_validation_scores
from structure.rdc import RDC
from structure.wacsf import WACSF

import torch
from torchinfo import summary
from sklearn.metrics import mean_squared_error

import model_utils
from utils import print_cross_validation_scores

import random

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################


def main(
    mode: str,
    model_mode: str,
    x_path: str,
    y_path: str,
    descriptor_type: str,
    descriptor_params: dict = {},
    data_params: dict = {},
    kfold_params: dict = {},
    hyperparams: dict = {},
    max_samples: int = None,
    variance_threshold: float = 0.0,
    epochs: int = 100,
    callbacks: dict = {},
    seed: int = None,
    save: bool = True,
):
    """
    LEARN. The .xyz (X) and XANES spectral (Y) data are loaded and transformed;
    a neural network is set up and fit to these data to find an Y <- X mapping.
    K-fold cross-validation is possible if {kfold_params} are provided.

    Args:
            x_path (str): The path to the .xyz (X) data; expects either a directory
                    containing .xyz files or a .npz archive file containing an 'x' key,
                    e.g. the `dataset.npz` file created when save == True. If a .npz
                    archive is provided, save is toggled to False, and the data are not
                    preprocessed, i.e. they are expected to be ready to be passed into
                    the neural net.
            y_path (str): The path to the XANES spectral (Y) data; expects either a
                    directory containing .txt FDMNES output files or a .npz archive
                    file containing 'y' and 'e' keys, e.g. the `dataset.npz` file
                    created when save == True. If a .npz archive is provided, save is
                    toggled to False, and the data are not preprocessed, i.e. they are
                    expected to be ready to be passed into the neural net.
            descriptor_type (str): The type of descriptor to use; the descriptor
                    transforms molecular systems into fingerprint feature vectors
                    that encodes the local environment around absorption sites. See
                    xanesnet.descriptors for additional information.
            descriptor_params (dict, optional): A dictionary of keyword
                    arguments passed to the descriptor on initialisation.
                    Defaults to {}.
            kfold_params (dict, optional): A dictionary of keyword arguments
                    passed to a scikit-learn K-fold splitter (KFold or RepeatedKFold).
                    If an empty dictionary is passed, no K-fold splitting is carried
                    out, and all available data are exposed to the neural network.
                    Defaults to {}.
            hyperparams (dict, optional): A dictionary of hyperparameter
                    definitions used to configure a Sequential Keras neural network.
                    Defaults to {}.
            max_samples (int, optional): The maximum number of samples to select
                    from the X/Y data; the samples are chosen according to a uniform
                    distribution from the full X/Y dataset.
                    Defaults to None.
            variance_threshold (float, optional): The minimum variance threshold
                    tolerated for input features; input features with variances below
                    the variance threshold are eliminated.
                    Defaults to 0.0.
            epochs (int, optional): The maximum number of epochs/cycles.
                    Defaults to 100.
            callbacks (dict, optional): A dictionary of keyword arguments passed
                    to set up Keras neural network callbacks; each argument is
                    expected to be dictionary of arguments for the defined callback,
                    e.g. "earlystopping": {"patience": 10, "verbose": 1}
                    Defaults to {}.
            seed (int, optional): A random seed used to initialise a Numpy
                    RandomState random number generator; set the seed explicitly for
                    reproducible results over repeated calls to the `learn` routine.
                    Defaults to None.
            save (bool, optional): If True, a model directory (containing data,
                    serialised scaling/pipeline objects, and the serialised model)
                    is created; this is required to restore the model state later.
                    Defaults to True.
    """

    rng = RandomState(seed=seed)

    xyz_path = [Path(p) for p in glob(x_path)]
    xanes_path = [Path(p) for p in glob(y_path)]

    xyz_list = []
    xanes_list = []
    e_list = []
    element_label = []

    for n_element in range(0, len(xyz_path)):

        element_name = str(xyz_path[n_element]).split("/")[-3]

        for path in (xyz_path[n_element], xanes_path[n_element]):
            if not path.exists():
                err_str = f"path to X/Y data ({path}) doesn't exist"
                raise FileNotFoundError(err_str)

        if xyz_path[n_element].is_dir() and xanes_path[n_element].is_dir():
            print(">> loading data from directories...\n")

            ids = list(
                set(list_filestems(xyz_path[n_element]))
                & set(list_filestems(xanes_path[n_element]))
            )

            ids.sort()

            descriptors = {"rdc": RDC, "wacsf": WACSF}

            descriptor = descriptors.get(descriptor_type)(**descriptor_params)

            n_samples = len(ids)
            n_x_features = descriptor.get_len()
            n_y_features = linecount(xanes_path[n_element] / f"{ids[0]}.txt") - 2

            xyz_data = np.full((n_samples, n_x_features), np.nan)
            print(">> preallocated {}x{} array for X data...".format(*xyz_data.shape))
            xanes_data = np.full((n_samples, n_y_features), np.nan)
            print(">> preallocated {}x{} array for Y data...".format(*xanes_data.shape))
            print(">> ...everything preallocated!\n")

            print(">> loading data into array(s)...")
            for i, id_ in enumerate(tqdm.tqdm(ids)):
                element_label.append(element_name)
                with open(xyz_path[n_element] / f"{id_}.xyz", "r") as f:
                    atoms = load_xyz(f)
                xyz_data[i, :] = descriptor.transform(atoms)
                with open(xanes_path[n_element] / f"{id_}.txt", "r") as f:
                    xanes = load_xanes(f)
                e, xanes_data[i, :] = xanes.spectrum
            print(">> ...loaded into array(s)!\n")

            xyz_list.append(xyz_data)
            xanes_list.append(xanes_data)
            e_list.append(e)

        elif x_path[n_element].is_file() and y_path[n_element].is_file():
            print(">> loading data from .npz archive(s)...\n")

            with open(x_path[n_element], "rb") as f:
                xyz_data = np.load(f)["x"]
            print(">> ...loaded {}x{} array of X data".format(*xyz_data.shape))
            with open(y_path[n_element], "rb") as f:
                xanes_data = np.load(f)["y"]
                e = np.load(f)["e"]
            print(">> ...loaded {}x{} array of Y data".format(*xanes_data.shape))
            print(">> ...everything loaded!\n")

            xyz_list.append(xyz_data)
            xanes_list.append(xanes_data)
            e_list.append(e)

            if save:
                print(">> overriding save flag (running in `--no-save` mode)\n")
                save = False

        else:

            err_str = (
                "paths to X/Y data are expected to be either a) both "
                "files (.npz archives), or b) both directories"
            )
            raise TypeError(err_str)

    xyz_data = np.vstack(xyz_list)
    xanes_data = np.vstack(xanes_list)
    e = np.vstack(e_list)
    element_label = np.asarray(element_label)

    # DATA AUGMENTATION
    if data_params:
        if data_params["augment"]:
            data_aug_params = data_params["augment_params"]
            n_aug_samples = (
                np.multiply(n_samples, data_params["augment_mult"]) - n_samples
            )
            print(">> ...AUGMENTING DATA...\n")
            if data_params["augment_type"].lower() == "random_noise":
                # augment data as random data point + noise

                rand = random.choices(range(n_samples), k=n_aug_samples)
                noise1 = np.random.normal(
                    data_aug_params["normal_mean"],
                    data_aug_params["normal_sd"],
                    (n_aug_samples, n_x_features),
                )
                noise2 = np.random.normal(
                    data_aug_params["normal_mean"],
                    data_aug_params["normal_sd"],
                    (n_aug_samples, n_y_features),
                )

                data1 = xyz_data[rand, :] + noise1
                data2 = xanes_data[rand, :] + noise2

                element_label = np.append(element_label, element_label[rand])

            elif data_params["augment_type"].lower() == "random_combination":

                rand1 = random.choices(range(n_samples), k=n_aug_samples)
                rand2 = random.choices(range(n_samples), k=n_aug_samples)

                data1 = 0.5 * (xyz_data[rand1, :] + xyz_data[rand2, :])
                data2 = 0.5 * (xanes_data[rand1, :] + xanes_data[rand2, :])

                element_label = np.append(element_label, element_label[rand1])
            else:
                raise ValueError("augment_type not found")

            xyz_data = np.vstack((xyz_data, data1))
            xanes_data = np.vstack((xanes_data, data2))

            print(">> ...FINISHED AUGMENTING DATA...\n")

    print(xyz_data.shape)
    print(element_label.shape)

    if save:
        parent_model_dir = "model/"
        Path(parent_model_dir).mkdir(parents=True, exist_ok=True)

        model_dir = unique_path(Path(parent_model_dir), "model")
        model_dir.mkdir()

        with open(model_dir / "descriptor.pickle", "wb") as f:
            pickle.dump(descriptor, f)
        with open(model_dir / "dataset.npz", "wb") as f:
            np.savez_compressed(f, ids=ids, x=xyz_data, y=xanes_data, e=e)

    print(">> shuffling and selecting data...")
    xyz, xanes, element = shuffle(
        xyz_data, xanes_data, element_label, random_state=rng, n_samples=max_samples
    )
    print(">> ...shuffled and selected!\n")

    # Setup K-fold Cross Validation variables
    if kfold_params:
        kfold_spooler = RepeatedKFold(
            n_splits=kfold_params["n_splits"],
            n_repeats=kfold_params["n_repeats"],
            random_state=rng,
        )
        fit_time = []
        prev_score = 1e6
        loss_fn = kfold_params["loss"]["loss_fn"]
        loss_args = kfold_params["loss"]["loss_args"]
        kfold_loss_fn = model_utils.LossSwitch().fn(loss_fn, loss_args)

    # getting exp name for mlflow
    exp_name = f"{mode}_{model_mode}"

    if mode == "train_xyz":
        print("training xyz structure")

        if model_mode == "mlp" or model_mode == "cnn":
            from learn import train

            if kfold_params:
                # K-fold Cross Validation model evaluation
                train_score = []
                test_score = []
                for fold, (train_index, test_index) in enumerate(
                    kfold_spooler.split(xyz)
                ):
                    print(">> fitting neural net...")
                    # Training
                    start = time.time()
                    model, score = train(
                        xyz[train_index],
                        xanes[train_index],
                        exp_name,
                        model_mode,
                        hyperparams,
                        epochs,
                    )
                    train_score.append(score)
                    fit_time.append(time.time() - start)
                    # Testing
                    model.eval()
                    xyz_test = torch.from_numpy(xyz[test_index]).float()
                    pred_xanes = model(xyz_test)
                    pred_score = kfold_loss_fn(
                        torch.tensor(xanes[test_index]), pred_xanes
                    ).item()
                    test_score.append(pred_score)
                    if pred_score < prev_score:
                        best_model = model
                    prev_score = pred_score
                result = {
                    "fit_time": fit_time,
                    "train_score": train_score,
                    "test_score": test_score,
                }
                print_cross_validation_scores(result, model_mode)
            else:
                print(">> fitting neural net...")
                model, score = train(
                    xyz, xanes, exp_name, model_mode, hyperparams, epochs
                )
                summary(model, (1, xyz.shape[1]))

        elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
            from ae_learn import train

            if kfold_params:
                # K-fold Cross Validation model evaluation
                train_score = []
                test_recon_score = []
                test_pred_score = []
                for fold, (train_index, test_index) in enumerate(
                    kfold_spooler.split(xyz)
                ):
                    print(">> fitting neural net...")
                    # Training
                    start = time.time()
                    model, score = train(
                        xyz[train_index],
                        xanes[train_index],
                        model_mode,
                        hyperparams,
                        epochs,
                    )
                    train_score.append(score)
                    fit_time.append(time.time() - start)
                    # Testing
                    model.eval()
                    xyz_test = torch.from_numpy(xyz[test_index]).float()
                    xanes_test = torch.from_numpy(xanes[test_index]).float()
                    recon_xyz, pred_xanes = model(xyz_test)
                    recon_score = kfold_loss_fn(xyz_test, recon_xyz).item()
                    pred_score = kfold_loss_fn(xanes_test, pred_xanes).item()
                    test_recon_score.append(recon_score)
                    test_pred_score.append(pred_score)
                    mean_score = np.mean([recon_score, pred_score])
                    if mean_score < prev_score:
                        best_model = model
                    prev_score = mean_score
                result = {
                    "fit_time": fit_time,
                    "train_score": train_score,
                    "test_recon_score": test_recon_score,
                    "test_pred_score": test_pred_score,
                }
                print_cross_validation_scores(result, model_mode)
            else:
                print(">> fitting neural net...")
                model, score = train(xyz, xanes, model_mode, hyperparams, epochs)
                summary(model, (1, xyz.shape[1]))

    elif mode == "train_xanes":
        print("training xanes spectrum")

        print(">> fitting neural net...")

        if model_mode == "mlp" or model_mode == "cnn":
            from learn import train

            if kfold_params:
                train_score = []
                test_score = []
                # K-fold Cross Validation model evaluation
                for fold, (train_index, test_index) in enumerate(
                    kfold_spooler.split(xyz)
                ):
                    print(">> fitting neural net...")
                    # Training
                    start = time.time()
                    model, score = train(
                        xanes[train_index],
                        xyz[train_index],
                        exp_name,
                        model_mode,
                        hyperparams,
                        epochs,
                    )
                    train_score.append(score)
                    fit_time.append(time.time() - start)
                    # Testing
                    model.eval()
                    xanes_test = torch.from_numpy(xanes[test_index]).float()
                    pred_xyz = model(xanes_test)
                    pred_score = kfold_loss_fn(
                        torch.tensor(xyz[test_index]), pred_xyz
                    ).item()
                    test_score.append(pred_score)
                    if pred_score < prev_score:
                        best_model = model
                    prev_score = pred_score
                result = {
                    "fit_time": fit_time,
                    "train_score": train_score,
                    "test_score": test_score,
                }
                print_cross_validation_scores(result, model_mode)
            else:
                print(">> fitting neural net...")
                model, score = train(
                    xanes, xyz, exp_name, model_mode, hyperparams, epochs
                )
                summary(model, (1, xanes.shape[1]))

        elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
            from ae_learn import train

            if kfold_params:
                # K-fold Cross Validation model evaluation
                train_score = []
                test_recon_score = []
                test_pred_score = []
                for fold, (train_index, test_index) in enumerate(
                    kfold_spooler.split(xyz)
                ):
                    print(">> fitting neural net...")
                    # Training
                    start = time.time()
                    model, score = train(
                        xanes[train_index],
                        xyz[train_index],
                        model_mode,
                        hyperparams,
                        epochs,
                    )
                    train_score.append(score)
                    fit_time.append(time.time() - start)
                    # Testing
                    model.eval()
                    xyz_test = torch.from_numpy(xyz[test_index]).float()
                    xanes_test = torch.from_numpy(xanes[test_index]).float()
                    recon_xanes, pred_xyz = model(xanes_test)
                    recon_score = kfold_loss_fn(xanes_test, recon_xanes).item()
                    pred_score = kfold_loss_fn(xyz_test, pred_xyz).item()
                    test_recon_score.append(recon_score)
                    test_pred_score.append(pred_score)
                    mean_score = np.mean([recon_score, pred_score])
                    if mean_score < prev_score:
                        best_model = model
                    prev_score = mean_score
                result = {
                    "fit_time": fit_time,
                    "train_score": train_score,
                    "test_recon_score": test_recon_score,
                    "test_pred_score": test_pred_score,
                }
                print_cross_validation_scores(result, model_mode)
            else:
                print(">> fitting neural net...")
                model, score = train(xanes, xyz, model_mode, hyperparams, epochs)
                summary(model, (1, xanes.shape[1]))

    elif mode == "train_aegan":
        from aegan_learn import train_aegan

        if kfold_params:
            # K-fold Cross Validation model evaluation
            train_score = []
            test_recon_xyz_score = []
            test_recon_xanes_score = []
            test_pred_xyz_score = []
            test_pred_xanes_score = []
            for fold, (train_index, test_index) in enumerate(kfold_spooler.split(xyz)):
                print(">> fitting neural net...")
                # Training
                start = time.time()
                model, score = train_aegan(
                    xyz[train_index], xanes[train_index], hyperparams, epochs
                )
                train_score.append(score["train_loss"][-1])
                fit_time.append(time.time() - start)
                # Testing
                model.eval()
                xyz_test = torch.from_numpy(xyz[test_index]).float()
                xanes_test = torch.from_numpy(xanes[test_index]).float()
                (
                    recon_xyz,
                    recon_xanes,
                    pred_xyz,
                    pred_xanes,
                ) = model.reconstruct_all_predict_all(xyz_test, xanes_test)
                recon_xyz_score = kfold_loss_fn(xyz_test, recon_xyz).item()
                recon_xanes_score = kfold_loss_fn(xanes_test, recon_xanes).item()
                pred_xyz_score = kfold_loss_fn(xyz_test, pred_xyz).item()
                pred_xanes_score = kfold_loss_fn(xanes_test, pred_xanes).item()
                test_recon_xyz_score.append(recon_xyz_score)
                test_recon_xanes_score.append(recon_xanes_score)
                test_pred_xyz_score.append(pred_xyz_score)
                test_pred_xanes_score.append(pred_xanes_score)
                mean_score = np.mean(
                    [
                        recon_xyz_score,
                        recon_xanes_score,
                        pred_xyz_score,
                        pred_xanes_score,
                    ]
                )
                if mean_score < prev_score:
                    best_model = model
                prev_score = mean_score

            result = {
                "fit_time": fit_time,
                "train_score": train_score,
                "test_recon_xyz_score": test_recon_xyz_score,
                "test_recon_xanes_score": test_recon_xanes_score,
                "test_pred_xyz_score": test_pred_xyz_score,
                "test_pred_xanes_score": test_pred_xanes_score,
            }
            print_cross_validation_scores(result, model_mode)
        else:
            print(">> fitting neural net...")
            model, score = train_aegan(xyz, xanes, hyperparams, epochs)
            summary(model)

        # from plot import plot_running_aegan

        # plot_running_aegan(losses, model_dir)

    if save:
        if kfold_params:
            torch.save(best_model, model_dir / f"model.pt")
            print("Saved best model to disk")
        else:
            torch.save(model, model_dir / f"model.pt")
            print("Saved model to disk")

    else:
        print("none")

    return
