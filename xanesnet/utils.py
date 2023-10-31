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

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import torch
import yaml
import numpy as np
import pickle as pickle

from pathlib import Path
from ase import Atoms
from typing import TextIO
from tqdm import tqdm

from xanesnet.spectrum.xanes import XANES

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################


def unique_path(path: Path, base_name: str) -> Path:
    # returns a unique path from `p`/`base_name`_001, `p`/`base_name`_002,
    # `p`/`base_name`_003, etc.

    n = 0
    while True:
        n += 1
        unique_path = path / (base_name + f"_{n:03d}")
        if not unique_path.exists():
            return unique_path


def linecount(f: Path) -> int:
    # returns the linecount for a file (`f`)

    with open(f, "r") as f_:
        return len([l for l in f_])


def list_files(d: Path, with_ext: bool = True) -> list:
    # returns a list of files (as POSIX paths) found in a directory (`d`);
    # 'hidden' files are always omitted and, if with_ext == False, file
    # extensions are also omitted

    return [
        (f if with_ext else f.with_suffix(""))
        for f in d.iterdir()
        if f.is_file() and not f.stem.startswith(".")
    ]


def list_filestems(d: Path) -> list:
    # returns a list of file stems (as strings) found in a directory (`d`);
    # 'hidden' files are always omitted

    return [f.stem for f in list_files(d)]


def str_to_numeric(str_: str):
    # returns the numeric (floating-point or integer) cast of `str_` if
    # cast is allowed, otherwise returns `str_`

    try:
        return float(str_) if "." in str_ else int(str_)
    except ValueError:
        return str_


def print_nested_dict(dict_: dict, nested_level: int = 0):
    # prints the key:value pairs in a dictionary (`dict`) in the format
    # '>> key :: value'; iterates recursively through any subdictionaries,
    # indenting with two white spaces for each sublevel (`nested level`)

    for key, val in dict_.items():
        if not isinstance(val, dict):
            if isinstance(val, list):
                val = f"[{val[0]}, ..., {val[-1]}]"
            print("  " * nested_level + f">> {key} :: {val}")
        else:
            print("  " * nested_level + f">> {key}")
            print_nested_dict(val, nested_level=nested_level + 1)

    return 0


def save_model(path, model, descriptor, data_compress, metadata):
    Path(path).mkdir(parents=True, exist_ok=True)

    model_dir = unique_path(Path(path), "model")
    model_dir.mkdir()

    with open(model_dir / "descriptor.pickle", "wb") as f:
        pickle.dump(descriptor, f)
    with open(model_dir / "dataset.npz", "wb") as f:
        np.savez_compressed(
            f,
            ids=data_compress["ids"],
            x=data_compress["x"],
            y=data_compress["y"],
        )

    torch.save(model, model_dir / f"model.pt")
    print(f"saved model to disk: {model_dir}")

    metadata["mdl_dir"] = str(model_dir)
    with open(model_dir / "metadata.yaml", "w") as f:
        yaml.dump_all([metadata], f)


def save_models(path, models, descriptor, data_compress, metadata):
    for model in models:
        save_model(path, model, descriptor, data_compress, metadata)


def save_prediction(save_path, mode, pred_data, index, e):
    if e is None:
        e = np.arange(pred_data.shape[1])
    if mode == "predict_xanes":
        for id_, y_predict_ in tqdm.tqdm(zip(index, pred_data)):
            with open(save_path / f"{id_}.txt", "w") as f:
                save_xanes(f, XANES(e, y_predict_.detach().numpy()))

    elif mode == "predict_xyz":
        for id_, y_predict_ in tqdm.tqdm(zip(index, pred_data)):
            with open(save_path / f"{id_}.txt", "w") as f:
                f.write("\n".join(map(str, y_predict_.detach().numpy())))


def load_xyz(xyz_f: TextIO) -> Atoms:
    # loads an Atoms object from a .xyz file

    xyz_f_l = xyz_f.readlines()

    # pop the number of atoms `n_ats`
    n_ats = int(xyz_f_l.pop(0))

    # pop the .xyz comment block
    comment_block = xyz_f_l.pop(0)

    # pop the .xyz coordinate block
    coord_block = [xyz_f_l.pop(0).split() for _ in range(n_ats)]
    # atomic symbols or atomic numbers
    ats = np.array([l[0] for l in coord_block], dtype="str")
    # atomic coordinates in .xyz format
    xyz = np.array([l[1:] for l in coord_block], dtype="float32")

    try:
        info = dict(
            [
                [key, str_to_numeric(val)]
                for key, val in [
                    pair.split(" = ") for pair in comment_block.split(" | ")
                ]
            ]
        )
    except ValueError:
        info = dict()

    try:
        # return Atoms object, assuming `ats` contains atomic symbols
        return Atoms(ats, xyz, info=info)
    except KeyError:
        # return Atoms object, assuming `ats` contains atomic numbers
        return Atoms(ats.astype("uint8"), xyz, info=info)


def save_xyz(xyz_f: TextIO, atoms: Atoms):
    # saves an Atoms object in .xyz format

    # write the number of atoms in `atoms`
    xyz_f.write(f"{len(atoms)}\n")
    # write additional info ('key = val', '|'-delimited) from the `atoms.info`
    # dictionary to the .xyz comment block
    for i, (key, val) in enumerate(atoms.info.items()):
        if i < len(atoms.info) - 1:
            xyz_f.write(f"{key} = {val} | ")
        else:
            xyz_f.write(f"{key} = {val}")
    xyz_f.write("\n")
    # write atomic symbols and atomic coordinates in .xyz format
    for atom in atoms:
        fmt = "{:<4}{:>16.8f}{:>16.8f}{:>16.8f}\n"
        xyz_f.write(fmt.format(atom.symbol, *atom.position))

    return 0


def load_xanes(xanes_f: TextIO) -> XANES:
    # loads a XANES object from an FDMNES (.txt) output file

    xanes_f_l = xanes_f.readlines()

    # pop the FDMNES header block
    for _ in range(2):
        xanes_f_l.pop(0)

    # pop the XANES spectrum block
    xanes_block = [xanes_f_l.pop(0).split() for _ in range(len(xanes_f_l))]
    # absorption energies
    e = np.array([l[0] for l in xanes_block], dtype="float64")
    # absorption intensities
    m = np.array([l[1] for l in xanes_block], dtype="float64")

    return XANES(e, m)


def save_xanes(xanes_f: TextIO, xanes: XANES):
    # saves a XANES object in FDMNES (.txt) output format

    xanes_f.write(f'{"FDMNES":>10}\n{"energy":>10}{"<xanes>":>12}\n')
    for e_, m_ in zip(*xanes.spectrum):
        fmt = f"{e_:>10.2f}{m_:>15.7E}\n"
        xanes_f.write(fmt.format(e_, m_))

    return 0


def save_xanes_mean(xanes_f: TextIO, xanes: XANES, std):
    # saves a mean and sandard deviation of XANES object in FDMNES (.txt) output format
    xanes_f.write(
        f'{"FDMNES":>10}\n{"energy":>10}{"<mean_xanes>":>12}{"<std_xanes>":>12}\n'
    )
    for e_, m_, std_ in zip(*xanes.spectrum, std):
        fmt = f"{e_:>10.2f}{m_:>15.7E}{std_:>15.7E}\n"
        xanes_f.write(fmt.format(e_, m_, std_))

    return 0


def save_xyz_mean(xyz_f: TextIO, mean, std):
    # saves a mean and sandard deviation of XANES object in FDMNES (.txt) output format
    xyz_f.write(f'{"<mean_xyz>":>12}{"<std_xyz>":>12}\n')
    for m_, std_ in zip(mean, std):
        fmt = f"{m_:>15.7E}{std_:>15.7E}\n"
        xyz_f.write(fmt.format(m_, std_))

    return 0