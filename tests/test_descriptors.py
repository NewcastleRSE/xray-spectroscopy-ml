import numpy as np
import pytest

from xanesnet.creator import create_descriptor
from xanesnet.utils import load_xyz


class TestWACSF:
    config = {
        "type": "wacsf",
        "params": {"r_min": 0.5, "r_max": 6.5, "n_g2": 16, "n_g4": 32},
    }

    @pytest.fixture(scope="module")
    def descriptor(self):
        return create_descriptor(self.config["type"], **self.config["params"])

    def test_transform(self, descriptor):
        with open("tests/data/xyz/0853.xyz", "r") as f:
            atoms = load_xyz(f)
            out = descriptor.transform(atoms)
        # Check shape
        assert out.shape == (49,)
        # Check that there are no NaNs or infs
        assert np.all(np.isfinite(out))

    def test_get_type(self, descriptor):
        assert descriptor.get_type() == "wacsf"

    def test_get_nfeatures(self, descriptor):
        assert descriptor.get_nfeatures() == 49
