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
from numpy.random import RandomState
from sklearn.model_selection import RepeatedKFold
from sklearn.utils import shuffle
# import tensorflow as tf

from inout import load_xyz
from inout import load_xanes
# from dnn import check_gpu_support
# from dnn import build_mlp
from utils import unique_path
from utils import linecount
from utils import list_filestems
from utils import print_cross_validation_scores
from structure.rdc import RDC
from structure.wacsf import WACSF

from AE import train_ae
import torch
from torchinfo import summary

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################

def main(
    aemode: str,
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

    rng = RandomState(seed = seed)

    x_path = Path(x_path)
    y_path = Path(y_path)
    
    for path in (x_path, y_path):
        if not path.exists():
            err_str = f'path to X/Y data ({path}) doesn\'t exist'
            raise FileNotFoundError(err_str)

    if x_path.is_dir() and y_path.is_dir():
        print('>> loading data from directories...\n')

        ids = list(
            set(list_filestems(x_path)) & set(list_filestems(y_path))
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
        n_y_features = linecount(y_path / f'{ids[0]}.txt') - 2

        x = np.full((n_samples, n_x_features), np.nan)
        print('>> preallocated {}x{} array for X data...'.format(*x.shape))
        y = np.full((n_samples, n_y_features), np.nan)
        print('>> preallocated {}x{} array for Y data...'.format(*y.shape))
        print('>> ...everything preallocated!\n')

        print('>> loading data into array(s)...')
        for i, id_ in enumerate(tqdm.tqdm(ids)):
            with open(x_path / f'{id_}.xyz', 'r') as f:
                atoms = load_xyz(f)
                # print(type(atoms))
            x[i,:] = descriptor.transform(atoms)
            with open(y_path / f'{id_}.txt', 'r') as f:
                xanes = load_xanes(f)
                # print(xanes.spectrum)
            e, y[i,:] = xanes.spectrum
        print('>> ...loaded into array(s)!\n')

        if save:
            model_dir = unique_path(Path('.'), 'model')
            model_dir.mkdir()
            with open(model_dir / 'descriptor.pickle', 'wb') as f:
                pickle.dump(descriptor, f)
            with open(model_dir / 'dataset.npz', 'wb') as f:
                np.savez_compressed(f, ids = ids, x = x, y = y, e = e)

    elif x_path.is_file() and y_path.is_file():
        print('>> loading data from .npz archive(s)...\n')
        
        with open(x_path, 'rb') as f:
            x = np.load(f)['x']
        print('>> ...loaded {}x{} array of X data'.format(*x.shape))
        with open(y_path, 'rb') as f:
            y = np.load(f)['y']
            e = np.load(f)['e']
        print('>> ...loaded {}x{} array of Y data'.format(*y.shape))
        print('>> ...everything loaded!\n')

        if save:
            print('>> overriding save flag (running in `--no-save` mode)\n')
            save = False

    else:

        err_str = 'paths to X/Y data are expected to be either a) both ' \
            'files (.npz archives), or b) both directories'
        raise TypeError(err_str)

    print('>> shuffling and selecting data...')
    x, y = shuffle(x, y, random_state = rng, n_samples = max_samples)
    print('>> ...shuffled and selected!\n')

    

    if aemode == 'ae_xyz':
        
        print('>> fitting neural net...')
    
        epoch, model, optimizer = train_ae(x, y, hyperparams, epochs)
        summary(model, (1, y.shape[1]))

        if save:
            # state = {
            #     "epoch": epoch,
            #     "state_dict": model.state_dict,
            #     "optimizer": optimizer
            # }
        
            # torch.save(model.state_dict(), model_dir / f"model.cpt")
            torch.save(model, model_dir / f"model.pt")
            print("Saved model to disk")

        # net.fit(
        #     x, y, epochs
        #     )
        # print(net.summary())
        # print('>> ...neural net fit!\n')

        # if save:
        #     model_json = net.to_json()
        #     with open(model_dir / f"model.json", "w") as json_file:
        #         json_file.write(model_json)
        #     net.save_weights(model_dir / f"model.h")
        #     print("Saved model to disk")
    else:
        print("none")
    
    return
