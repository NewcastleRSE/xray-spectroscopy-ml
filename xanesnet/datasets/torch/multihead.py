# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   ...

"""Descriptor-based tensor dataset for multi-head models."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from xanesnet.datasources import DataSource
from xanesnet.descriptors import Descriptor, DescriptorRegistry
from xanesnet.serialization.config import Config

from ..base import SavePathFn, TorchDataset
from ..registry import DatasetRegistry
from .descriptor import SPECTRUM_KEYS


@dataclass
class MultiheadData:
    """Container for one multi-head sample or batch.

    Attributes:
        x: Model input tensor, commonly ``(n_features,)`` or ``(batch, n_features)``.
        y: Model target tensor, commonly ``(n_energies,)`` or ``(batch, n_energies)``.
        energies: Energy grid tensor with shape ``(n_energies,)`` or ``(batch, n_energies)``.
        sample_id: Sample identifier metadata for one sample or a batch.
        element: Absorber atomic number as a scalar tensor for one sample, or
            ``(batch,)`` for a batch. Consumed by element-aware spectra
            encodings.
        head_idx: Active multi-head index for one sample, or ``(batch,)`` for
            a batch. Populated from ``subdir_id`` on the datasource entry.
    """

    x: torch.Tensor | None = None
    y: torch.Tensor | None = None
    energies: torch.Tensor | None = None
    sample_id: str | list[Any] | None = None
    element: torch.Tensor | None = None
    head_idx: torch.Tensor | None = None

    def to(self, device: str | torch.device) -> "MultiheadData":
        """Move tensor attributes to ``device`` in place.

        Args:
            device: Target device accepted by ``torch.Tensor.to``.

        Returns:
            This data object after moving tensor attributes.
        """
        for attr in ["x", "y", "energies", "element", "head_idx"]:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.to(device))
        return self

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize this sample to a torch-saveable state dictionary.

        Returns:
            Dictionary containing tensor and metadata fields.
        """
        return {
            "x": self.x,
            "y": self.y,
            "energies": self.energies,
            "sample_id": self.sample_id,
            "element": self.element,
            "head_idx": self.head_idx,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "MultiheadData":
        """Create data from a state dictionary.

        Args:
            state: State dictionary produced by ``to_state_dict``.

        Returns:
            Reconstructed multi-head data object.
        """
        return cls(
            x=state.get("x"),
            y=state.get("y"),
            energies=state.get("energies"),
            sample_id=state.get("sample_id"),
            element=state.get("element"),
            head_idx=state.get("head_idx"),
        )

    def save(self, path: str) -> str:
        """Save this data object to disk.

        Args:
            path: Destination ``.pth`` path.

        Returns:
            The destination path.
        """
        torch.save(self.to_state_dict(), path)
        return path

    @classmethod
    def load(cls, path: str) -> "MultiheadData":
        """Load multi-head data from disk.

        Args:
            path: Source ``.pth`` path.

        Returns:
            Loaded multi-head data object.
        """
        state = torch.load(path, weights_only=True)
        return cls.from_state_dict(state)


@DatasetRegistry.register("multihead")
class MultiheadDataset(TorchDataset):
    """Dataset that converts structures to descriptor tensors for multi-head models.

    Like :class:`~xanesnet.datasets.torch.descriptor.DescriptorDataset`, but
    each sample also stores ``head_idx`` from the datasource ``subdir_id``
    property so batch processors can select the active prediction head.

    Args:
        dataset_type: Registered dataset type name (``"multihead_descriptor"``).
        datasource: Raw datasource of pymatgen structures or molecules.
        root: Directory that stores processed ``.pth`` files.
        preload: Whether to preload processed samples.
        skip_prepare: Whether to reuse existing processed files.
        split_ratios: Optional split ratios.
        split_indexfile: Optional path to split indices.
        descriptors: Descriptor configuration objects.
    """

    _INVERSE_MARKER = "_inverse"

    def __init__(
        self,
        dataset_type: str,
        datasource: DataSource,
        root: str,
        preload: bool,
        skip_prepare: bool,
        split_ratios: list[float] | None,
        split_indexfile: str | None,
        # params:
        descriptors: list[Config],
    ) -> None:
        """Initialize the multi-head dataset."""
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

        self._inverse = self._INVERSE_MARKER in dataset_type

        self.descriptor_configs = descriptors
        self.descriptor_list: list[Descriptor] = []
        descriptor_types = ", ".join(d.get_str("descriptor_type") for d in descriptors)
        logging.info(f"Initializing descriptors: {descriptor_types}")
        for descriptor_config in descriptors:
            descriptor_type = descriptor_config.get_str("descriptor_type")
            descriptor = DescriptorRegistry.create(descriptor_type, **descriptor_config.as_kwargs())
            self.descriptor_list.append(descriptor)

    def _prepare_single(self, idx: int, save_path_fn: SavePathFn) -> int:
        """Process one datasource item into multi-head samples.

        Args:
            idx: Datasource index to process.
            save_path_fn: Callback that maps per-item sample sequence numbers to output paths.

        Returns:
            Number of processed absorber samples written.
        """
        pmg_obj = self.datasource[idx]
        for key in SPECTRUM_KEYS:
            if key in pmg_obj.site_properties.keys():
                break
        else:
            logging.warning(f"No XANES spectrum found for sample {idx} ({pmg_obj.properties['sample_id']}); skipping.")
            return 0

        xanes = np.array(pmg_obj.site_properties[key], dtype=object)
        xanes_idxs: list[int] = np.where(xanes != None)[0].tolist()

        descriptor_features = []
        for descriptor in self.descriptor_list:
            feature = descriptor.transform_pmg(pmg_obj, site_index=xanes_idxs)
            descriptor_features.append(feature)
        descriptor_features = np.concatenate(descriptor_features, axis=1)

        head_idx = torch.tensor(pmg_obj.properties["subdir_id"], dtype=torch.int64)

        seq = 0
        for site_idx, df in zip(xanes_idxs, descriptor_features):
            df = torch.tensor(df, dtype=torch.float32)
            element = torch.tensor(pmg_obj.atomic_numbers[site_idx], dtype=torch.int64)

            spectrum = pmg_obj.site_properties[key][site_idx]
            energies = torch.tensor(spectrum["energies"], dtype=torch.float32)
            intensities = torch.tensor(spectrum["intensities"], dtype=torch.float32)

            if self._inverse:
                x = intensities
                y = df
            else:
                x = df
                y = intensities

            data = MultiheadData(
                x=x,
                y=y,
                energies=energies,
                sample_id=pmg_obj.properties["sample_id"],
                element=element,
                head_idx=head_idx,
            )
            data.save(save_path_fn(seq))
            seq += 1

        return seq

    def collate_fn(self, batch: list[MultiheadData]) -> MultiheadData:
        """Collate multi-head samples into a batch.

        Args:
            batch: Multi-head samples loaded by ``__getitem__``.

        Returns:
            Batched multi-head data with stacked tensor fields.
        """

        def _stack(tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
            """Stack tensors unless any field is absent for the batch."""
            if any(t is None for t in tensors):
                return None
            return torch.stack([tensor for tensor in tensors if tensor is not None])

        return MultiheadData(
            x=_stack([b.x for b in batch]),
            y=_stack([b.y for b in batch]),
            energies=_stack([b.energies for b in batch]),
            sample_id=[b.sample_id for b in batch],
            element=_stack([b.element for b in batch]),
            head_idx=_stack([b.head_idx for b in batch]),
        )

    def _load_item(self, path: str) -> MultiheadData:
        """Load one processed multi-head data sample.

        Args:
            path: Path to a processed ``.pth`` file.

        Returns:
            Loaded multi-head data object.
        """
        return MultiheadData.load(path)

    @property
    def signature(self) -> Config:
        """Dataset configuration signature.

        Returns:
            Configuration values that identify this multi-head dataset.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "descriptors": self.descriptor_configs,
            }
        )
        return signature
