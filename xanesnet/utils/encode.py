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

import logging
import numpy as np
import tqdm as tqdm

from typing import Tuple
from pathlib import Path

from xanesnet.utils.io import load_xanes, linecount, load_xyz


def encode_xyz(xyz_path: Path, index: list, descriptor_list: list) -> np.ndarray:
    """
    Encodes XYZ molecular data into numerical feature array,
    Multiple descriptors can be applied, and the results are concatenated.
    """
    n_samples = len(index)
    n_features = 0
    # Get total feature length
    for descriptor in descriptor_list:
        n_features += descriptor.get_nfeatures()

    # Feature array pre-allocation
    xyz_data = np.full((n_samples, n_features), np.nan)

    # Iterate each sample data to extract descriptor values,
    # and assign result to array.
    for i, id_ in enumerate(tqdm.tqdm(index)):
        s = 0
        for descriptor in descriptor_list:
            l = descriptor.get_nfeatures()
            if descriptor.get_type() == "direct":
                with open(xyz_path / f"{id_}.dsc", "r") as f:
                    xyz_data[i, s : s + l] = np.loadtxt(f)
            else:
                with open(xyz_path / f"{id_}.xyz", "r") as f:
                    atoms = load_xyz(f)
                xyz_data[i, s : s + l] = descriptor.transform(atoms)
            s += l

        # Check for any NaN values in the encoded data
        if np.any(np.isnan(xyz_data[i, :])):
            logging.info(f"Warning issue arising with transformation of {id_}.")
            continue

    return xyz_data


def encode_xanes(xanes_path: Path, index: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encodes XANES spectral data into a numerical feature array.
    """
    n_samples = len(index)
    n_y_features = linecount(xanes_path / f"{index[0]}.txt") - 2

    # Feature array pre-allocation
    xanes_data = np.full((n_samples, n_y_features), np.nan)

    # Iterate each sample and assign XANES spectra to array
    for i, id_ in enumerate(tqdm.tqdm(index)):
        with open(xanes_path / f"{id_}.txt", "r") as f:
            xanes = load_xanes(f)
        e, xanes_data[i, :] = xanes.spectrum

    return xanes_data, e
