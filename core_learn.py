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

    xyz_path = [Path(p) for p in glob (x_path)]
    xanes_path = [Path(p) for p in glob (y_path)]
    
    xyz_list = []
    xanes_list = []
    e_list = []  
    element_label = []

    for n_element in range (0, len(xyz_path)):
        
        element_name = str(xyz_path[n_element]).split("/")[-3]

        for path in (xyz_path[n_element], xanes_path[n_element]):
            if not path.exists():
                err_str = f'path to X/Y data ({path}) doesn\'t exist'
                raise FileNotFoundError(err_str)

        if xyz_path[n_element].is_dir() and xanes_path[n_element].is_dir():
            print('>> loading data from directories...\n')

            ids = list(
                set(list_filestems(xyz_path[n_element])) & set(list_filestems(xanes_path[n_element]))
            )

            ids.sort()

            descriptors = {
                'rdc': RDC,
                'wacsf': WACSF
            }
            
            descriptor = (
                descriptors.get(descriptor_type)(**descriptor_params)
            )

            n_samples = len(ids)
            n_x_features = descriptor.get_len()
            n_y_features = linecount(xanes_path[n_element] / f'{ids[0]}.txt') - 2

            xyz_data = np.full((n_samples, n_x_features), np.nan)
            print('>> preallocated {}x{} array for X data...'.format(*xyz_data.shape))
            xanes_data = np.full((n_samples, n_y_features), np.nan)
            print('>> preallocated {}x{} array for Y data...'.format(*xanes_data.shape))
            print('>> ...everything preallocated!\n')

            print('>> loading data into array(s)...')
            for i, id_ in enumerate(tqdm.tqdm(ids)):
                element_label.append(element_name)
                with open(xyz_path[n_element] / f'{id_}.xyz', 'r') as f:
                    atoms = load_xyz(f)
                xyz_data[i,:] = descriptor.transform(atoms)
                with open(xanes_path[n_element] / f'{id_}.txt', 'r') as f:
                    xanes = load_xanes(f)
                e, xanes_data[i,:] = xanes.spectrum
            print('>> ...loaded into array(s)!\n')

            xyz_list.append(xyz_data)
            xanes_list.append(xanes_data)
            e_list.append(e)

        elif x_path[n_element].is_file() and y_path[n_element].is_file():
            print('>> loading data from .npz archive(s)...\n')
            
            with open(x_path[n_element], 'rb') as f:
                xyz_data = np.load(f)['x']
            print('>> ...loaded {}x{} array of X data'.format(*xyz_data.shape))
            with open(y_path[n_element], 'rb') as f:
                xanes_data = np.load(f)['y']
                e = np.load(f)['e']
            print('>> ...loaded {}x{} array of Y data'.format(*xanes_data.shape))
            print('>> ...everything loaded!\n')

            xyz_list.append(xyz_data)
            xanes_list.append(xanes_data)
            e_list.append(e)

            if save:
                print('>> overriding save flag (running in `--no-save` mode)\n')
                save = False

        else:

            err_str = 'paths to X/Y data are expected to be either a) both ' \
                'files (.npz archives), or b) both directories'
            raise TypeError(err_str)

    xyz_data = np.vstack(xyz_list)
    xanes_data = np.vstack(xanes_list)
    e = np.vstack(e_list)
    element_label = np.asarray(element_label)

    print(xyz_data.shape)
    print(element_label.shape)

    if save:
        model_dir = unique_path(Path('.'), 'model')
        model_dir.mkdir()
        with open(model_dir / 'descriptor.pickle', 'wb') as f:
            pickle.dump(descriptor, f)
        with open(model_dir / 'dataset.npz', 'wb') as f:
            np.savez_compressed(f, ids = ids, x = xyz_data, y = xanes_data, e = e)

    print('>> shuffling and selecting data...')
    xyz, xanes, element = shuffle(xyz_data, xanes_data, element_label, random_state = rng, n_samples = max_samples)
    print('>> ...shuffled and selected!\n')

    # if kfold_params:

    #     kfold_spooler = RepeatedKFold(**kfold_params, random_state=rng)

    #     fit_time = []
    #     train_score = []
    #     test_score = []
    #     prev_score = 1

    #     for train_index, test_index in kfold_spooler.split(x):

    #         start = time.time()
    #         model, score = train_mlp(
    #             x[train_index], y[train_index], hyperparams, epochs
    #         )
    #         # model, score = train_cnn(
    #         #     x[train_index], y[train_index], hyperparams, epochs
    #         # )
    #         train_score.append(score)

    #         fit_time.append(time.time() - start)

    #         model.eval()
    #         x_test = torch.from_numpy(x[test_index])
    #         x_test = x_test.float()
    #         y_predict = model(x_test)
    #         y_score = mean_squared_error(y[test_index], y_predict.detach().numpy())

    #         test_score.append(y_score)

    #         if y_score < prev_score:
    #             best_model = model

    #         prev_score = y_score

    #     result = {
    #         "fit_time": fit_time,
    #         "train_score": train_score,
    #         "test_score": test_score,
    #     }
    #     print_cross_validation_scores(result)

    #     if save:
    #         torch.save(best_model, model_dir / f"model.pt")
    #         print("Saved model to disk")

    # else:

    if mode == "train_xyz":
        print("training xyz structure")

        print(">> fitting neural net...")

        if model_mode == "mlp" or model_mode == "cnn":
            from learn import train

            model, score = train(xyz, xanes, model_mode, hyperparams, epochs)

        elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
            from ae_learn import train

            model, score = train(xyz, xanes, model_mode, hyperparams, epochs)

        summary(model, (1, xyz.shape[1]))

    elif mode == "train_xanes":
        print("training xanes spectrum")

        print(">> fitting neural net...")

        if model_mode == "mlp" or model_mode == "cnn":
            from learn import train

            model, score = train(xanes, xyz, model_mode, hyperparams, epochs)

        elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
            from ae_learn import train

            model, score = train(xanes, xyz, model_mode, hyperparams, epochs)

        summary(model, (1, xanes.shape[1]))

    elif mode == "train_aegan":
        from aegan_learn import train_aegan

        losses, model = train_aegan(xyz, xanes, hyperparams, epochs)
        summary(model)

        from plot import plot_running_aegan

        plot_running_aegan(losses, model_dir)

    if save:

        torch.save(model, model_dir / f"model.pt")
        print("Saved model to disk")

    else:
        print("none")

    return
