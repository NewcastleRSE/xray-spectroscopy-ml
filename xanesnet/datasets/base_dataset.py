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
import copy
import logging
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from collections.abc import Sequence
from pathlib import Path
from typing import Union, List, Any, Callable

from torch import Tensor
from torch_geometric.io import fs

IndexType = Union[slice, Tensor, np.ndarray, Sequence]

###############################################################################
################################## CLASSES ####################################
###############################################################################


class BaseDataset(Dataset):
    """An abstract base class for xanesnet datasets."""

    def __init__(
        self,
        root: str | Path,
        xyz_path: List[str] | str | Path = None,
        xanes_path: List[str] | str | Path = None,
        descriptors: List = None,
        **kwargs,
    ):
        super().__init__()

        self.root = Path(root)
        self.xyz_path = xyz_path
        self.xanes_path = xanes_path
        self.descriptor_list = descriptors

        self.config = {}
        self.index = None
        self.xyz_data = self.xanes_data = self.e_data = None
        self.X = self.y = None

        self.set_index()
        self._process()

    def indices(self) -> Sequence:
        return range(len(self.index))

    def set_index(self) -> List[str]:
        """List of identifiers for each data point."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the `self.processed_dir` folder."""
        raise NotImplementedError

    def __getitem__(self, idx: int | slice) -> None:
        raise NotImplementedError

    @property
    def processed_file_names(self) -> List[str]:
        """A list of all processed file names."""
        raise NotImplementedError

    @property
    def processed_dir(self) -> str:
        """The directory containing processed datasets"""
        return os.path.join(self.root, "processed")

    @property
    def processed_paths(self) -> List[str]:
        """A list of absolute paths to all processed files."""
        files = self.processed_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [os.path.join(self.processed_dir, f) for f in to_list(files)]

    def __len__(self) -> int:
        return len(self.index)

    def _process(self):
        """
        Checks if processing is complete and runs the process() if not.
        """
        if files_exist(self.processed_paths):
            logging.info(
                f">> Processed files exist in {self.processed_dir}, skipping data processing."
            )
            return

        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

    def index_select(self, idx: IndexType) -> "BaseDataset":
        r"""Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.
        """
        indices = self.indices()

        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            # Allow floating-point slicing, e.g., dataset[:0.9]
            if isinstance(start, float):
                start = round(start * len(self))
            if isinstance(stop, float):
                stop = round(stop * len(self))
            idx = slice(start, stop, step)

            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')"
            )

        dataset = copy.copy(self)
        dataset._indices = indices
        return dataset

    def register_config(self, args, **kwargs):
        """
        Assign arguments from the child class's constructor to self.config.

        Args:
            args: The dictionary of arguments from the child class's constructor
            awargs: Addition
            **kwargs: additional arguments to store
        """
        config = kwargs.copy()

        # Extract parameters from the local_vars, excluding 'self' and '__class__'
        args_dict = {
            key: val
            for key, val in args.items()
            if key not in ["self", "__class__", "descriptors", "kwargs"]
        }

        # config.update(params)
        config.update(args_dict)

        self.config = config


def files_exist(files: List[str]) -> bool:
    return len(files) != 0 and all([fs.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]
