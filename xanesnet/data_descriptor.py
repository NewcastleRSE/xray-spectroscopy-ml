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
import tqdm as tqdm

from pathlib import Path

from xanesnet.utils import load_xyz, load_xanes, linecount, list_filestems, load_descriptor_direct


def encode_train(xyz_path, xanes_path, descriptor):
    xyz_path = Path(xyz_path)
    xanes_path = Path(xanes_path)

    for path in (xyz_path, xanes_path):
        if not path.exists():
            err_str = f"path to data ({path}) doesn't exist"
            raise FileNotFoundError(err_str)

    if xyz_path.is_dir() and xanes_path.is_dir():
        index = list(set(list_filestems(xyz_path)) & set(list_filestems(xanes_path)))
        index.sort()

        n_samples = len(index)
        
        if descriptor.__class__.__name__ == 'DIRECT':
            n_x_features = linecount(xyz_path / f"{index[0]}.dsc")
        else:
            n_x_features = descriptor.get_number_of_features()
        n_y_features = linecount(xanes_path / f"{index[0]}.txt") - 2

        xyz_data = np.full((n_samples, n_x_features), np.nan)
        print(">> preallocated {}x{} array for XYZ data...".format(*xyz_data.shape))
        xanes_data = np.full((n_samples, n_y_features), np.nan)
        print(">> preallocated {}x{} array for XANES data...".format(*xanes_data.shape))

        print(">> loading data into array(s)...")
        for i, id_ in enumerate(tqdm.tqdm(index)):
            if descriptor.__class__.__name__ == 'DIRECT':
               with open(xyz_path / f"{id_}.dsc", "r") as f:
                   xyz_data[i, :] = load_descriptor_direct(f)
            else:
               with open(xyz_path / f"{id_}.xyz", "r") as f:
                   atoms = load_xyz(f)
               xyz_data[i, :] = descriptor.process(atoms)
            with open(xanes_path / f"{id_}.txt", "r") as f:
                xanes = load_xanes(f)
            e, xanes_data[i, :] = xanes.spectrum

    elif xyz_path.is_file() and xanes_path.is_file():
        print(">> loading data from .npz archive(s)...\n")

        with open(xyz_path, "rb") as f:
            xyz_data = np.load(f)["x"]
        print(">> ...loaded {}x{} array of XYZ data".format(*xyz_data.shape))
        with open(xanes_path, "rb") as f:
            xanes_data = np.load(f)["y"]
            e = np.load(f)["e"]
        print(">> ...loaded {}x{} array of XANES data".format(*xanes_data.shape))

    else:
        err_str = (
            "paths to data are expected to be either a) both "
            "files (.npz archives), or b) both directories"
        )
        raise TypeError(err_str)

    return xyz_data, xanes_data, index, n_x_features, n_y_features


def encode_predict(xyz_path, xanes_path, descriptor, mode, pred_eval):
    if mode == "predict_all" or pred_eval:
        xyz_path = Path(xyz_path)
        xanes_path = Path(xanes_path)
        index = list(set(list_filestems(xyz_path)) & set(list_filestems(xanes_path)))
    elif mode == "predict_xyz" and not pred_eval:
        xanes_path = Path(xanes_path)
        index = list(set(list_filestems(xanes_path)))
    elif mode == "predict_xanes" and not pred_eval:
        xyz_path = Path(xyz_path)
        index = list(set(list_filestems(xyz_path)))
    else:
        raise ValueError("Unsupported prediction mode")

    index.sort()
    n_samples = len(index)

    print(">> loading data into array(s)...")
    if mode == "predict_all" or pred_eval:
        # Load both xyz and xanes data
        n_x_features = descriptor.get_number_of_features()
        n_y_features = linecount(xanes_path / f"{index[0]}.txt") - 2
        xanes_data = np.full((n_samples, n_y_features), np.nan)
        xyz_data = np.full((n_samples, n_x_features), np.nan)
        print(">> preallocated {}x{} array for XYZ data...".format(*xyz_data.shape))
        print(">> preallocated {}y{} array for XANES data...".format(*xanes_data.shape))

        print(">> loading data into array(s)...")
        for i, id_ in enumerate(tqdm.tqdm(index)):
            with open(xyz_path / f"{id_}.xyz", "r") as f:
                atoms = load_xyz(f)
            xyz_data[i, :] = descriptor.process(atoms)
            with open(xanes_path / f"{id_}.txt", "r") as f:
                xanes = load_xanes(f)
            e, xanes_data[i, :] = xanes.spectrum

    elif mode == "predict_xyz" and not pred_eval:
        # Load xanes data
        n_y_features = linecount(xanes_path / f"{index[0]}.txt") - 2
        xanes_data = np.full((n_samples, n_y_features), np.nan)
        print(">> preallocated {}y{} array for XANES data...".format(*xanes_data.shape))

        for i, id_ in enumerate(tqdm.tqdm(index)):
            with open(xanes_path / f"{id_}.txt", "r") as f:
                xanes = load_xanes(f)
            e, xanes_data[i, :] = xanes.spectrum
        xyz_data = None

    elif mode == "predict_xanes" and not pred_eval:
        # Load xyz data
        n_x_features = descriptor.get_number_of_features()
        xyz_data = np.full((n_samples, n_x_features), np.nan)
        print(">> preallocated {}x{} array for XYZ data...".format(*xyz_data.shape))

        for i, id_ in enumerate(tqdm.tqdm(index)):
            with open(xyz_path / f"{id_}.xyz", "r") as f:
                atoms = load_xyz(f)
            xyz_data[i, :] = descriptor.process(atoms)
        xanes_data = None
        e = None
    else:
        raise ValueError("Unsupported prediction mode")

    return xyz_data, xanes_data, index, e
