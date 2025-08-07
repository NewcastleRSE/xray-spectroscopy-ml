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
from pathlib import Path

import torch
from torch import nn
from typing import List, Tuple
import numpy as np
from torch import Tensor

from tqdm import tqdm
from typing import List, Union
from xanesnet.datasets.base_dataset import BaseDataset, IndexType
from xanesnet.registry import register_dataset
from xanesnet.utils.encode import encode_xanes
from xanesnet.utils.io import list_filestems
from ase.data import atomic_masses
from xanesnet.utils.encode import encode_xyz
from xanesnet.utils.io import load_xyz

@register_dataset("transformer")
class TransformerDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        xyz_path: List[str] | str | Path = None,
        xanes_path: List[str] | str | Path = None,
        descriptors: list = None,
        **kwargs,
    ):

        # Transformer dataset accepts only a single path
        if isinstance(xyz_path, List):
            self.xyz_path = Path(xyz_path[0]) if xyz_path else None
            if xyz_path and len(xyz_path) > 1:
                raise ValueError("Invalid dataset: xyz_path cannot be > 1")
        else:
            self.xyz_path = Path(xyz_path) if xyz_path else None

        if isinstance(xanes_path, List):
            self.xanes_path = Path(xanes_path[0]) if xanes_path else None
            if xanes_path and len(xanes_path) > 1:
                raise ValueError("Invalid dataset: xanes_path cannot be > 1")
        else:
            self.xanes_path = Path(xanes_path) if xanes_path else None

        # Transformer dataset accepts exactly 1 descriptor of type mace and exactly 1 other descriptr
        count_mace = sum(1 for desc in descriptors if desc.get_type() == "mace")
        if len(descriptors) !=2:
            raise ValueError("Invalid dataset: Transformer dataset requires exactly 2 types of descriptors")
        count_mace = sum(1 for desc in descriptors if desc.get_type() == "mace")
        if (count_mace != 1):
            raise ValueError("Invalid dataset: Transformer dataset requires that exactly 1 descriptor is of type 'mace'")
        if descriptors[0].absorber_atom_only:
            raise ValueError("Invalid dataset: Transformer dataset with mace descriptor values for the absorber atom only is not implemented")

        BaseDataset.__init__(
            self, root, self.xyz_path, self.xanes_path, descriptors, **kwargs
        )

        # Save configuration
        params = {
        }
        self.register_config(locals(), type="transformer")
        # Load processed data into RAM
        self.set_datasets()

    def set_datasets(self):
        self.mace_descriptor_data = []
        self.other_descriptor_data = []
        self.atom_positions_data = []
        self.atom_weights_data = []
        self.atom_masks_data = []
        self.xanes_data = []
        self.xyz_data = []
        for idx, stem in enumerate(self.index):
            (
                mace_descriptor,
                other_descriptor,
                atom_positions,
                atom_weights,
                atom_masks,
                xanes
            ) =  torch.load(self.processed_paths[idx])
            self.mace_descriptor_data.append(mace_descriptor)
            self.other_descriptor_data.append(other_descriptor)
            self.atom_positions_data.append(atom_positions)
            self.atom_weights_data.append(atom_weights)
            self.atom_masks_data.append(atom_masks)
            self.xyz_data.append((
                mace_descriptor,
                other_descriptor,
                atom_positions,
                atom_weights,
                atom_masks,
            ))
            self.xanes_data.append(np.array(xanes))
        self.xanes_data = np.array(self.xanes_data)


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
        """The directory containing processed transformer datasets"""
        return self.root / "processed"

    def __getitem__(self, idx: Union[int, np.integer, IndexType]):
        data = {}
        # Dataset preloaded in RAM
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            if self.xyz_path:
                xyz_data = self.xyz_data[idx]
            if self.xanes_path:
                xanes_data = self.xanes_data[idx]
            return xyz_data, xanes_data
        else:
            return self.index_select(idx)

    def process(self):
        """
        Processes raw data and saves it to the processed_dir.
        """
        mace_descriptor_data = other_descriptor_data = atom_positions_data = atom_weights_data = atom_masks_data = xanes_data = None

        # Encode xyz data
        if self.xyz_path:
            #separate mace from other descriptors
            mace_descriptor = next(desc for desc in self.descriptor_list if desc.get_type() == "mace")
            other_descriptor = [desc for desc in self.descriptor_list if desc.get_type() != "mace"]

            # get mace encoding, atom positions and atom weights
            mace_descriptor_data = []
            atom_positions_data = []
            atom_weights_data = []
            atom_masks_data = []
            for idx, stem in tqdm(enumerate(self.index), total=len(self.index)):
                raw_path = os.path.join(self.xyz_path, f"{stem}.xyz")
                with open(raw_path, "r") as f:
                    atoms = load_xyz(f)
                mace_encoding_one_molecule = torch.tensor(mace_descriptor.transform(atoms), dtype=torch.float32)
                atom_positions_one_molecule = torch.tensor(atoms.get_positions(), dtype=torch.float32)
                atom_weights_one_molecule = torch.tensor([atomic_masses[Z] for Z in atoms.get_atomic_numbers()], dtype=torch.float32)
                atom_mask_one_molecule = torch.ones(mace_encoding_one_molecule.shape[0], dtype=torch.bool)
    
                mace_descriptor_data.append(mace_encoding_one_molecule)
                atom_positions_data.append(atom_positions_one_molecule)
                atom_weights_data.append(atom_weights_one_molecule)
                atom_masks_data.append(atom_mask_one_molecule)
    
            mace_data_all_atoms = torch.cat(mace_descriptor_data, dim=0)
            mean = mace_data_all_atoms.mean(dim=0, keepdim=True)
            std = mace_data_all_atoms.std(dim=0, keepdim=True) + 1e-8
            mace_descriptor_data = [(mdd - mean) / std for mdd in mace_descriptor_data]

            # get encoding from other descriptors
            other_descriptor_data = encode_xyz(Path(self.xyz_path), self.index, other_descriptor)
            other_descriptor_data = torch.from_numpy(other_descriptor_data).float()

        # Encode xanes data
        if self.xanes_path:
            xanes_data, _ = encode_xanes(self.xanes_path, self.index)
            xanes_data_array = np.stack(xanes_data)  # shape: (N, D)
            valid_mask = np.std(xanes_data_array, axis=0) > 1e-6  # shape: (D,)
            xanes_data = [xanes[valid_mask] for xanes in xanes_data_array]

        # Save dataset to disk
        for idx, stem in tqdm(enumerate(self.index), total=len(self.index)):
            data = (
                mace_descriptor_data[idx],
                other_descriptor_data[idx],
                atom_positions_data[idx],
                atom_weights_data[idx],
                atom_masks_data[idx],
                xanes_data[idx]
            )

            save_path = os.path.join(self.processed_dir, f"{idx}_{stem}.pt")
            torch.save(data, save_path)