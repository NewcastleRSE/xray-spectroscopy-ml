import os
import shutil
from pathlib import Path

import torch
import yaml

from xanesnet.creator import create_predict_scheme
from xanesnet.data_encoding import data_predict, data_gnn_predict
from xanesnet.model import MLP, AE_MLP, AEGAN_MLP, GNN
from xanesnet.scheme import NNPredict, AEPredict, AEGANPredict, GNNPredict
from xanesnet.utils import load_descriptors, load_models

config = {
    "xyz_path": "tests/data/xyz_predict",
    "xanes_path": "tests/data/xanes_predict",
}


def init_scheme(model_dir: Path, mode):
    # Load metadata
    metadata_file = f"{model_dir}/metadata.yaml"
    with open(metadata_file, "r") as f:
        metadata = yaml.safe_load(f)
    model_name = metadata["model_type"]

    # Load descriptor list
    descriptor_list = load_descriptors(model_dir)

    # Encode prediction dataset with saved descriptors
    xyz, xanes, e, index = data_predict(
        config["xyz_path"], config["xanes_path"], descriptor_list, mode, False
    )
    # Initialise prediction scheme
    scheme = create_predict_scheme(
        model_name,
        xyz,
        xanes,
        mode,
        False,
        metadata["standardscaler"],
        metadata["fourier_transform"],
        metadata["fourier_param"],
    )

    return scheme


def init_scheme_gnn(model_dir: Path):
    # Load metadata
    metadata_file = f"{model_dir}/metadata.yaml"
    with open(metadata_file, "r") as f:
        metadata = yaml.safe_load(f)

    # Load descriptor list
    descriptor_list = load_descriptors(model_dir)

    # Remove existing graph data
    graph_path = os.path.join(config["xyz_path"], "graph")
    if os.path.exists(graph_path):
        shutil.rmtree(graph_path)

    # Encode prediction dataset with saved descriptor
    graph_dataset, index, xanes_data, e = data_gnn_predict(
        config["xyz_path"],
        config["xanes_path"],
        metadata["node_features"],
        metadata["edge_features"],
        descriptor_list,
        False,
    )

    # Initialise prediction scheme
    scheme = create_predict_scheme(
        "gnn",
        graph_dataset,
        xanes_data,
        "predict_xanes",
        False,
        metadata["standardscaler"],
        metadata["fourier_transform"],
        metadata["fourier_param"],
    )

    return scheme


def init_model(model_dir):
    return torch.load(model_dir / "model.pt", map_location=torch.device("cpu"))


def init_model_list(model_dir):
    return load_models(model_dir)


class TestNNPredict:
    def test_case_xyz(self):
        model_dir = Path("tests/models/model_mlp_xyz")
        scheme = init_scheme(model_dir, "predict_xanes")
        model = init_model(model_dir)
        # Check scheme and model instances
        assert isinstance(scheme, NNPredict)
        assert isinstance(model, MLP)
        # Check prediction result
        result = scheme.predict_std(model)
        assert result.xyz_pred[0] is None
        assert result.xyz_pred[1] is None
        assert result.xanes_pred[0].shape == (5, 400)
        assert result.xanes_pred[1].shape == (5, 400)

    def test_case_xanes(self):
        model_dir = Path("tests/models/model_mlp_xanes")
        scheme = init_scheme(model_dir, "predict_xyz")
        model = init_model(model_dir)
        # Check prediction result
        result = scheme.predict_std(model)
        assert result.xyz_pred[0].shape == (4, 49)
        assert result.xyz_pred[1].shape == (4, 49)
        assert result.xanes_pred[0] is None
        assert result.xanes_pred[1] is None


class TestAEPredict:
    def test_std_xyz(self):
        model_dir = Path("tests/models/model_ae_xyz")
        scheme = init_scheme(model_dir, "predict_xanes")
        model = init_model(model_dir)
        # Check scheme and model instances
        assert isinstance(scheme, AEPredict)
        assert isinstance(model, AE_MLP)
        # Check prediction result
        result = scheme.predict_std(model)
        assert result.xyz_pred[0] is None
        assert result.xyz_pred[1] is None

        assert result.xanes_pred[0].shape == (5, 400)
        assert result.xanes_pred[1].shape == (5, 400)

        assert result.xyz_recon[0].shape == (5, 49)
        assert result.xyz_recon[1].shape == (5, 49)

        assert result.xanes_recon[0] is None
        assert result.xanes_recon[1] is None

    def test_std_xanes(self):
        model_dir = Path("tests/models/model_ae_xanes")
        scheme = init_scheme(model_dir, "predict_xyz")
        model = init_model(model_dir)
        # Check prediction result
        result = scheme.predict_std(model)
        assert result.xyz_pred[0].shape == (4, 49)
        assert result.xyz_pred[1].shape == (4, 49)

        assert result.xanes_pred[0] is None
        assert result.xanes_pred[1] is None

        assert result.xyz_recon[0] is None
        assert result.xyz_recon[1] is None

        assert result.xanes_recon[0].shape == (4, 400)
        assert result.xanes_recon[1].shape == (4, 400)


class TestAEGANPredict:
    def test_case_all(self):
        model_dir = Path("tests/models/model_aegan")
        scheme = init_scheme(model_dir, "predict_all")
        model = init_model(model_dir)
        # Check scheme and model instances
        assert isinstance(scheme, AEGANPredict)
        assert isinstance(model, AEGAN_MLP)
        # Check prediction result
        result = scheme.predict_std(model)
        assert result.xyz_pred[0].shape == (4, 49)
        assert result.xyz_pred[1].shape == (4, 49)

        assert result.xanes_pred[0].shape == (4, 400)
        assert result.xanes_pred[1].shape == (4, 400)

        assert result.xyz_recon[0].shape == (4, 49)
        assert result.xyz_recon[1].shape == (4, 49)

        assert result.xanes_recon[0].shape == (4, 400)
        assert result.xanes_recon[1].shape == (4, 400)

    def test_std_xyz(self):
        model_dir = Path("tests/models/model_aegan")
        scheme = init_scheme(model_dir, "predict_xanes")
        model = init_model(model_dir)
        # Check prediction result
        result = scheme.predict_std(model)
        assert result.xyz_pred[0] is None
        assert result.xyz_pred[1] is None

        assert result.xanes_pred[0].shape == (5, 400)
        assert result.xanes_pred[1].shape == (5, 400)

        assert result.xyz_recon[0].shape == (5, 49)
        assert result.xyz_recon[1].shape == (5, 49)

        assert result.xanes_recon[0] is None
        assert result.xanes_recon[1] is None

    def test_std_xanes(self):
        model_dir = Path("tests/models/model_aegan")
        scheme = init_scheme(model_dir, "predict_xyz")
        model = init_model(model_dir)
        # Check prediction result
        result = scheme.predict_std(model)
        assert result.xyz_pred[0].shape == (4, 49)
        assert result.xyz_pred[1].shape == (4, 49)

        assert result.xanes_pred[0] is None
        assert result.xanes_pred[1] is None

        assert result.xyz_recon[0] is None
        assert result.xyz_recon[1] is None

        assert result.xanes_recon[0].shape == (4, 400)
        assert result.xanes_recon[1].shape == (4, 400)


class TestGNNPredict:
    def test_gnn(self):
        model_dir = Path("tests/models/model_gnn")
        scheme = init_scheme_gnn(model_dir)
        model = init_model(model_dir)
        # Check prediction result
        # Check scheme and model instances
        assert isinstance(scheme, GNNPredict)
        assert isinstance(model, GNN)
        # Check prediction result
        result = scheme.predict_std(model)
        assert result.xyz_pred[0] is None
        assert result.xyz_pred[1] is None
        assert result.xanes_pred[0].shape == (5, 400)
        assert result.xanes_pred[1].shape == (5, 400)
