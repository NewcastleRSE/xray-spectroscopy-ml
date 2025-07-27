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
import os
from pathlib import Path

import torch
import numpy as np
from torch import Tensor

from tqdm import tqdm
from typing import List, Union
from torch_geometric.data import Data

from xanesnet.datasets.base_dataset import BaseDataset, IndexType
from xanesnet.registry import register_dataset
from xanesnet.utils.encode import encode_xanes
from xanesnet.utils.fourier import fourier_transform
from xanesnet.utils.io import list_filestems
from xanesnet.utils.xyz2graph import MolGraph


@register_dataset("graph")
class GraphDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        xyz_path: List[str] | str | Path = None,
        xanes_path: List[str] | str | Path = None,
        descriptors: list = None,
        **kwargs,
    ):
        # Unpack kwargs
        self.n = kwargs.get("n", 16)
        self.r_min = kwargs.get("r_min", 0.0)
        self.r_max = kwargs.get("r_max", 4.0)
        self.fft = kwargs.get("fourier", False)
        self.fft_concat = kwargs.get("fourier_concat", False)

        # Graph dataset accepts only a single path
        if isinstance(xyz_path, List):
            self.xyz_path = Path(xyz_path[0]) if xyz_path else None
            if xyz_path and len(xyz_path) > 1:
                raise ValueError("Invalid dataset: xyz_path cannot be > 1")
        else:
            self.xyz_path = Path(xyz_path) if xyz_path else None

        if isinstance(xanes_path, List):
            self.xanes_path = Path(xanes_path[0]) if xanes_path else None
            if xanes_path and len(xanes_path) > 1:
                raise ValueError("Invalid dataset: xyz_paths cannot be > 1")
        else:
            self.xanes_path = Path(xanes_path) if xanes_path else None

        BaseDataset.__init__(
            self, root, self.xyz_path, self.xanes_path, descriptors, **kwargs
        )

        # Save configuration
        params = {
            "fourier": self.fft,
            "fourier_concat": self.fft_concat,
            "n": self.n,
            "r_min": self.r_min,
            "r_max": self.r_max,
        }
        self.register_config(locals(), type="graph")

        # Assign this dataset to xyz_data
        self.xyz_data = self

    def set_index(self):
        xyz_stems = set(list_filestems(self.xyz_path))
        if self.xyz_path and self.xanes_path:
            xanes_stems = set(list_filestems(self.xanes_path))
            # Find common files
            index = sorted(list(xyz_stems & xanes_stems))
        else:
            index = sorted(list(xyz_stems))

        if not index:
            raise ValueError("No matching files found in xyz_path and xanes_path.")

        self.index = index

    @property
    def processed_file_names(self) -> List[str]:
        """A list of all processed file names."""
        return [f"{i}_{stem}.pt" for i, stem in enumerate(self.index)]

    @property
    def processed_dir(self) -> str:
        """The directory containing processed graph datasets"""
        xyz_dirname = os.path.basename(self.xyz_path)
        return self.root / "processed" / xyz_dirname

    def process(self):
        """
        Processes raw XYZ and Xanes files to convert them into graph data objects.
        """
        logging.info("Encoding XANES spectra...")

        xanes_data = None
        if self.xanes_path:
            xanes_data, e = encode_xanes(self.xanes_path, self.index)
            self.e_data = e

        # Apply FFT to spectra training dataset if specified
        if self.fft:
            logging.info(">> Transforming spectra data using Fourier transform...")
            xanes_data = fourier_transform(xanes_data, self.fft_concat)

        logging.info(f"Converting {len(self.index)} XYZ files to graph data objects...")
        for idx, stem in tqdm(enumerate(self.index), total=len(self.index)):
            raw_path = os.path.join(self.xyz_path, f"{stem}.xyz")

            mg = MolGraph()
            mg.read_xyz(raw_path)

            y = torch.tensor(xanes_data[idx]) if xanes_data is not None else None

            data = Data(
                x=self._get_node_features(mg),
                edge_index=mg.edge_index,
                edge_attr=self._get_edge_features(mg),
                y=y,
                graph_attr=self._get_graph_features(mg),
                name=stem,
            )

            save_path = os.path.join(self.processed_dir, f"{idx}_{stem}.pt")
            torch.save(data, save_path)

    def __getitem__(
        self, idx: Union[int, np.integer, IndexType]
    ) -> Union["BaseDataset", Data]:
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            data = torch.load(self.processed_paths[idx])
            return data

        else:
            return self.index_select(idx)

    def _get_node_features(self, mg: MolGraph):
        """
        Return a 2d array of the shape [Number of Nodes, Node Feature size]
        """

        atomic_numbers = list(range(2, 21))

        mole_atomic_numbers = mg.atoms.get_atomic_numbers()

        one_hot_encoding = np.zeros(
            (len(mole_atomic_numbers), len(atomic_numbers) + 1), dtype=int
        )
        # Set one_hot_encoding for absorber
        one_hot_encoding[0, 0] = 1

        # Set one_hot_encoding for atomic number
        for i, atomic_number in enumerate(mole_atomic_numbers):
            if atomic_number in atomic_numbers:
                index = atomic_numbers.index(atomic_number)
                one_hot_encoding[i, index] = 1

        all_node_feats = one_hot_encoding

        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mg: MolGraph):
        """
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """

        num_edges = len(mg.edge_list)
        all_edge_feats = np.full((num_edges, self.n), np.nan)

        r_aux = np.linspace(self.r_min + 0.5, self.r_max - 0.5, self.n)
        dr = np.diff(r_aux)[0]
        width = np.array([1.0 / (2.0 * (dr**2)) for _ in r_aux])
        grid = np.array([i for i in r_aux])
        bond_lengths = np.array([mg.bond_lengths[i] for i in mg.edge_list])
        cutoffs = (np.cos((np.pi * bond_lengths) / self.r_max) + 1.0) / 2.0

        for i in range(num_edges):
            g2 = gaussian(bond_lengths[i], width, grid)
            all_edge_feats[i, :] = np.sum(g2 * cutoffs[i], axis=0)

        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_graph_features(self, mg: MolGraph):
        """
        This will return 1d vector of the shape
        [Feature size]
        """
        n_feats = 0
        # Feature array pre-allocation
        for descriptor in self.descriptor_list:
            n_feats += descriptor.get_nfeatures()
        all_graph_feats = np.full(n_feats, np.nan)

        s = 0
        for descriptor in self.descriptor_list:
            l = descriptor.get_nfeatures()
            all_graph_feats[s : s + l] = descriptor.transform(mg.atoms)
            s += l

        return torch.tensor(all_graph_feats, dtype=torch.float)


def gaussian(r: np.ndarray, h: float, m: float) -> np.ndarray:
    """returns a gaussian-like function defined over `r`"""
    return np.exp(-1.0 * h * (r - m) ** 2)
