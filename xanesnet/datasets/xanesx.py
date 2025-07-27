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
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from sklearn.utils import shuffle
from torch import Tensor

from xanesnet.datasets.base_dataset import BaseDataset, IndexType
from xanesnet.registry import register_dataset
from xanesnet.utils.encode import encode_xyz, encode_xanes
from xanesnet.utils.fourier import fourier_transform
from xanesnet.utils.io import list_filestems
from xanesnet.utils.switch import DataAugmentSwitch


@register_dataset("xanesx")
class XanesXDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        xyz_path: List[str] | str | Path = None,
        xanes_path: List[str] | str | Path = None,
        descriptors: list = None,
        shuffle: bool = False,
        **kwargs,
    ):
        # Unpack kwargs
        self.fft = kwargs.get("fourier", False)
        self.fft_concat = kwargs.get("fourier_concat", False)
        self.augment = kwargs.get("data_augment", False)
        self.aug_params = kwargs.get("augment_params", {})
        self.shuffle = shuffle

        # XYZXanes dataset accepts only a single path
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
            "data_augment": self.augment,
            "augment_params": self.aug_params,
        }
        self.register_config(locals(), type="xanesx")
        # Load processed data into RAM
        self.set_datasets()

    def set_datasets(self):
        processed_dir = Path(self.processed_dir)
        # Load structural data if the path exists
        if self.xyz_path:
            file_path = processed_dir / f"{self.xyz_path.name}.pt"
            self.xyz_data = torch.load(file_path)

        # Load spectral and energy data if the path exists
        if self.xanes_path:
            xanes_file_path = processed_dir / f"{self.xanes_path.name}.pt"
            self.xanes_data = torch.load(xanes_file_path)

            e_file_path = processed_dir / f"{self.xanes_path.name}_e.pt"
            self.e_data = torch.load(e_file_path)

    def set_index(self):
        xyz_path = self.xyz_path
        xanes_path = self.xanes_path

        if xyz_path and xanes_path:
            xyz_stems = set(list_filestems(xyz_path))
            xanes_stems = set(list_filestems(xanes_path))
            index = sorted(list(xyz_stems & xanes_stems))
        elif xyz_path:
            xyz_stems = set(list_filestems(xyz_path))
            index = sorted(list(xyz_stems))
        elif xyz_path:
            xanes_stems = set(list_filestems(xanes_path))
            index = sorted(list(xanes_stems))
        else:
            raise ValueError("At least one data dataset path must be provided.")

        if not index:
            raise ValueError("No matching files found in the provided paths.")

        self.index = index

    def processed_file_names(self) -> List[str]:
        """A list of all processed file names."""
        file_list = []

        # Conditionally add the xyz processed file name
        if self.xyz_path:
            file_list.append(f"{self.xyz_path.name}.pt")

        # Conditionally add the xanes processed file names
        if self.xanes_path:
            file_list.append(f"{self.xanes_path.name}.pt")
            file_list.append(f"{self.xanes_path.name}_e.pt")

        return file_list

    def __getitem__(self, idx: Union[int, np.integer, IndexType]):
        data = {}
        # Dataset preloaded in RAM
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            if self.xyz_path:
                data["xyz"] = self.xyz_data[idx]
            if self.xanes_path:
                data["xanes"] = self.xanes_data[idx]
            return data
        else:
            return self.index_select(idx)

    def process(self):
        """Processes raw data and saves it to the processed_dir."""
        processed_dir = Path(self.processed_dir)
        xyz_data = xanes_data = e_data = None

        # Encode xyz data
        if self.xyz_path:
            xyz_data = encode_xyz(self.xyz_path, self.index, self.descriptor_list)

        # Encode xanes data
        if self.xanes_path:
            xanes_data, e_data = encode_xanes(self.xanes_path, self.index)
            if self.fft:
                logging.info(">> Transforming spectra data using Fourier transform...")
                xanes_data = fourier_transform(xanes_data, self.fft_concat)

        if self.shuffle:
            xyz_data, xanes_data = shuffle(xyz_data, xanes_data)

        if self.augment:
            logging.info("Applying data augmentation...")
            xyz_data, xanes_data = DataAugmentSwitch().augment(
                xyz_data, xanes_data, **self.aug_params
            )

        # Save dataset to disk
        if self.xyz_path:
            torch.save(xyz_data, processed_dir / f"{self.xyz_path.name}.pt")

        if self.xanes_path:
            torch.save(xanes_data, processed_dir / f"{self.xanes_path.name}.pt")
            torch.save(e_data, processed_dir / f"{self.xanes_path.name}_e.pt")
