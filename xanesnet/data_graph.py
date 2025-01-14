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

import os

import numpy as np
import torch

from torch_geometric.data import Dataset, Data
from tqdm import tqdm
from pathlib import Path

from xanesnet.xyz2graph import MolGraph


class GraphDataset(Dataset):
    def __init__(
        self,
        root: str,
        index: list[str],
        node_feats: dict,
        edge_feats: dict,
        descriptor_list: list,
        xanes_data: np.ndarray = None,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root (str): Root directory containing XYZ files
            index (list[str]): List of XYZ file stems
            node_feats (dict): Node feature parameters obtained from the input file.
            edge_feats (dict): Edge feature parameters obtained from the input file.
            descriptor_list(list): List of descriptors added to graph feature
            xanes_data (np.ndarray): graph label
        """

        self.index = index
        self.xanes_data = xanes_data
        self.descriptor_list = descriptor_list
        # Extracted edge feature parameters
        self.n = edge_feats["n"]
        self.r_min = edge_feats["r_min"]
        self.r_max = edge_feats["r_max"]

        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        """The directory containing processed graph datasets"""
        return os.path.join(self.root, "graph")

    @property
    def raw_file_names(self):
        """If this file exists in raw_dir, the download is not triggered.
        (The download func. is not implemented here)
        """
        return [f"{i}.xyz" for i in list(self.index)]

    @property
    def processed_file_names(self):
        """If these files are found in raw_dir, processing is skipped"""
        file_names = []
        idx = 0
        for file_name in self.index:
            file_names.append(str(idx) + "_" + Path(file_name).stem + ".pt")
            idx += 1

        return file_names

    def download(self):
        pass

    def process(self):
        """
        Processes raw XYZ files to convert them into graph data objects.
        """
        idx = 0
        for raw_path in tqdm(self.raw_paths):
            mg = MolGraph()
            mg.read_xyz(raw_path)
            # Node features
            node_feats = self._get_node_features(mg)
            # Edge features
            edge_feats = self._get_edge_features(mg)
            # Graph features to be added as the additional argument to Data()
            graph_feats = self._get_graph_features(mg)
            # Get adjacency info
            edge_index = mg.edge_index
            # Get graph-level labels info
            if self.xanes_data is not None:
                label = self._get_labels(self.xanes_data[idx])
            else:
                label = None

            name = Path(raw_path).stem
            data = Data(
                x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_feats,
                y=label,
                graph_attr=graph_feats,
            )

            torch.save(data, os.path.join(self.processed_dir, f"{idx}_{name}.pt"))
            idx += 1

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

    def _get_labels(self, xanes_data):
        return torch.from_numpy(xanes_data)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """Load a file with specified prefix from the processed directory."""
        file_list = os.listdir(self.processed_dir)
        for name in file_list:
            if name.startswith(f"{idx}_"):
                path = os.path.join(self.processed_dir, name)
                data = torch.load(path)
                return data
        # Raise an error if no matching file is found
        raise FileNotFoundError(f"File not found: index={idx}")


def gaussian(r: np.ndarray, h: float, m: float) -> np.ndarray:
    """returns a gaussian-like function defined over `r`"""
    return np.exp(-1.0 * h * (r - m) ** 2)
