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
from ase.io import read
from mace.calculators import mace_mp

from pathlib import Path

from xanesnet.data_graph import GraphDataset
from xanesnet.utils import (
    load_xyz,
    load_xanes,
    linecount,
    list_filestems,
    load_descriptor_direct,
)


def encode_xyz(xyz_path: Path, index: list, descriptor_list: list):
    n_samples = len(index)
    # Feature length
    n_x_features = 0
    for descriptor in descriptor_list:
        if descriptor.get_type() == "direct":
            n_x_features += linecount(xyz_path / f"{index[0]}.dsc")
        elif descriptor.get_type() == "mace":
            env = read(xyz_path / f"{index[0]}.xyz")
            mace = mace_mp()
            tmp = mace.get_descriptors(env, num_layers=2)
            n_x_features += len(tmp[0, :])
        else:
            n_x_features += descriptor.get_nfeatures()

    # Feature array pre-allocation
    xyz_data = np.full((n_samples, n_x_features), np.nan)
    print(">> preallocated {}x{} array for XYZ data...".format(*xyz_data.shape))
    print(">> loading encoded data into array(s)...")
    for i, id_ in enumerate(tqdm.tqdm(index)):
        s = 0
        for descriptor in descriptor_list:
            if descriptor.get_type() == "direct":
                l = linecount(xyz_path / f"{index[0]}.dsc")
                with open(xyz_path / f"{id_}.dsc", "r") as f:
                    xyz_data[i, s : s + l] = load_descriptor_direct(f)
            elif descriptor.get_type() == "mace":
                env = read(xyz_path / f"{id_}.xyz")
                mace = mace_mp()
                tmp = mace.get_descriptors(env, num_layers=2)
                l = len(tmp[0, :])
                xyz_data[i, s : s + l] = tmp[0, :]
            else:
                l = descriptor.get_nfeatures()
                with open(xyz_path / f"{id_}.xyz", "r") as f:
                    atoms = load_xyz(f)
                xyz_data[i, s : s + l] = descriptor.transform(atoms)
            s += l
        if np.any(np.isnan(xyz_data[i, :])):
            print(f"Warning issue arising with transformation of {id_}.")
            continue

    return xyz_data


def encode_xanes(xanes_path: Path, index: list):
    n_samples = len(index)
    n_y_features = linecount(xanes_path / f"{index[0]}.txt") - 2
    xanes_data = np.full((n_samples, n_y_features), np.nan)

    print(">> preallocated {}x{} array for XANES data...".format(*xanes_data.shape))
    print(">> loading data into array(s)...")
    for i, id_ in enumerate(tqdm.tqdm(index)):
        with open(xanes_path / f"{id_}.txt", "r") as f:
            xanes = load_xanes(f)
        e, xanes_data[i, :] = xanes.spectrum

    return xanes_data, e


def data_learn(xyz_path: str, xanes_path: str, descriptor_list: list):
    """
    Process and encode data from given XYZ and xanes files using
    one or more descriptors.
    """
    xyz_path = Path(xyz_path)
    xanes_path = Path(xanes_path)

    if not xyz_path.exists() or not xanes_path.exists():
        raise FileNotFoundError("Path to data doesn't exist")

    if xyz_path.is_dir() and xanes_path.is_dir():
        # Dataset index by finding the common elements in XYZ and xanes files
        index = list(set(list_filestems(xyz_path)) & set(list_filestems(xanes_path)))
        index.sort()

        xyz_data = encode_xyz(xyz_path, index, descriptor_list)
        xanes_data, e = encode_xanes(xanes_path, index)

    elif xyz_path.is_file() and xanes_path.is_file():
        print(">> loading data from .npz archive(s)...\n")

        with open(xyz_path, "rb") as f:
            xyz_data = np.load(f)["x"]
        print(">> ...loaded {}x{} array of XYZ data".format(*xyz_data.shape))
        with open(xanes_path, "rb") as f:
            xanes_data = np.load(f)["y"]
        print(">> ...loaded {}x{} array of XANES data".format(*xanes_data.shape))
        with open(xyz_path, "rb") as f:
            index = np.load(f)["ids"]

    else:
        err_str = (
            "paths to data are expected to be either a) both "
            "files (.npz archives), or b) both directories"
        )
        raise TypeError(err_str)

    return xyz_data, xanes_data, index


def data_predict(
    xyz_path: str | None,
    xanes_path: str | None,
    descriptor_list: list,
    mode: str,
    pred_eval: bool,
):
    print(">> loading xanes data into array(s)...")
    if mode == "predict_all" or pred_eval:
        xyz_path = Path(xyz_path)
        xanes_path = Path(xanes_path)

        if not xyz_path.exists() or not xanes_path.exists():
            raise FileNotFoundError("Path to data doesn't exist")

        index = list(set(list_filestems(xyz_path)) & set(list_filestems(xanes_path)))
        index.sort()
        # Load both xyz and xanes data
        xyz_data = encode_xyz(xyz_path, index, descriptor_list)
        xanes_data, e = encode_xanes(xanes_path, index)

    elif mode == "predict_xyz":
        xanes_path = Path(xanes_path)

        if not xanes_path.exists():
            raise FileNotFoundError("Path to data doesn't exist")

        index = list(set(list_filestems(xanes_path)))
        index.sort()
        # Load xanes data
        xanes_data, e = encode_xanes(xanes_path, index)
        xyz_data = None

    elif mode == "predict_xanes":
        xyz_path = Path(xyz_path)

        if not xyz_path.exists():
            raise FileNotFoundError("Path to data doesn't exist")

        index = list(set(list_filestems(xyz_path)))
        index.sort()
        # Load xyz data
        xyz_data = encode_xyz(xyz_path, index, descriptor_list)
        xanes_data = None
        e = None
    else:
        raise ValueError("Unsupported prediction mode")

    return xyz_data, xanes_data, e, index


def data_gnn_learn(
    xyz_path: str,
    xanes_path: str,
    node_feats: dict,
    edge_feats: dict,
    descriptor_list: list,
):
    xyz_path = Path(xyz_path)
    xanes_path = Path(xanes_path)

    if not xyz_path.exists() or not xanes_path.exists():
        raise FileNotFoundError("Path to data doesn't exist")

    if xyz_path.is_dir() and xanes_path.is_dir():
        index = list(set(list_filestems(xyz_path)) & set(list_filestems(xanes_path)))
        index.sort()
        xanes_data, _ = encode_xanes(xanes_path, index)
        print(f"Converting {len(index)} data files from XYZ format to graphs...")
        graph_dataset = GraphDataset(
            root=str(xyz_path),
            index=index,
            node_feats=node_feats,
            edge_feats=edge_feats,
            descriptor_list=descriptor_list,
            xanes_data=xanes_data,
        )
    else:
        err_str = "paths to data are expected to be directories"
        raise TypeError(err_str)

    return graph_dataset


def data_gnn_predict(
    xyz_path: str | Path,
    xanes_path: str | Path,
    node_feats: dict,
    edge_feats: dict,
    descriptor_list: list,
    pred_eval: bool,
):
    xyz_path = Path(xyz_path)
    if not xyz_path.exists():
        raise FileNotFoundError("Path to data doesn't exist")

    if pred_eval:
        xanes_path = Path(xanes_path)
        if not xanes_path.exists():
            raise FileNotFoundError("Path to data doesn't exist")
        index = list(set(list_filestems(xyz_path)) & set(list_filestems(xanes_path)))
        index.sort()
        xanes_data, e = encode_xanes(xanes_path, index)
    else:
        index = list(set(list_filestems(xyz_path)))
        index.sort()
        xanes_data = None
        e = None

    print(f"Converting {len(index)} data files from XYZ format to graph format...")
    graph_dataset = GraphDataset(
        root=str(xyz_path),
        index=index,
        node_feats=node_feats,
        edge_feats=edge_feats,
        descriptor_list=descriptor_list,
        xanes_data=xanes_data,
    )

    return graph_dataset, index, xanes_data, e
