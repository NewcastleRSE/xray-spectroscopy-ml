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
from mendeleev import element

from xanesnet.utils import load_xyz
from xanesnet.xyz2graph import MolGraph


class GraphDataset(Dataset):
    def __init__(
        self,
        root,
        index,
        xanes_data,
        node_descriptors: [],
        edge_descriptors: [],
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root (str): root directory containing XYZ files
            index (List[str]): list of XYF file stems
            xanes_data: graph-level labels [n_samples, 400]
            node_descriptors (List[Descriptors]): list of node feature descriptors
            edge_descriptors (List[Descriptors]) = list of edge feature descriptors
        """

        self.index = index
        self.xanes_data = xanes_data
        self.node_descriptors = node_descriptors
        self.edge_descriptors = edge_descriptors
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
            # Get node features
            node_feats = self._get_node_features(mg, raw_path)
            # Get edge features
            edge_feats = self._get_edge_features(mg, raw_path)
            # Get adjacency info
            edge_index = mg.edge_index
            # Get graph-level labels info
            label = self._get_labels(self.xanes_data[idx])

            name = Path(raw_path).stem
            data = Data(
                x=node_feats, edge_index=edge_index, edge_attr=edge_feats, y=label
            )

            torch.save(data, os.path.join(self.processed_dir, f"{idx}_{name}.pt"))
            idx += 1

    def _get_node_features(self, mg: MolGraph, raw_path: str):
        """
        Return a 2d array of the shape [Number of Nodes, Node Feature size]
        """
        all_node_feats = []
        # Generic node features
        for e in mg.elements:
            node_feats = []
            e = element(e)
            node_feats.append(e.atomic_weight)

            all_node_feats.append(node_feats)

        # Absorbing features from descriptors
        for i in self.node_descriptors:
            with open(raw_path, "r") as f:
                atoms = load_xyz(f)
            descriptor_feature = i.transform(atoms)
            # Append to the first (absorbing) element
            all_node_feats[0].extend(descriptor_feature)
            # Extend the rest of rows
            for row in all_node_feats[1:]:
                row.extend([0] * (i.get_nfeatures()))

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mg: MolGraph, raw_path: str):
        """
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []

        for i in mg.edge_list:
            edge_feats = [mg.bond_lengths[i]]

            # Append edge features to matrix
            all_edge_feats += [edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_labels(self, xanes_data):
        return torch.from_numpy(xanes_data)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """Load a file with specified prefix from the processed directory."""
        file_list = os.listdir(self.processed_dir)
        for name in file_list:
            if name.startswith(str(idx)):
                path = os.path.join(self.processed_dir, name)
                data = torch.load(path)
                return data
