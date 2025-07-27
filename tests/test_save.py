import shutil
from pathlib import Path

import torch
import yaml

import pytest
from xanesnet.creator import (
    create_descriptor,
    create_model,
    create_learn_scheme,
    create_predict_scheme,
)
from xanesnet.utils.encode import data_learn, data_predict
from xanesnet.utils import save_models, load_descriptors, save_predict


def init_descriptors(config):
    descriptor_list = []
    descriptors = config.get("descriptors")

    for d in descriptors:
        descriptor = create_descriptor(d["type"], **d["params"])
        descriptor_list.append(descriptor)

    return descriptor_list


def init_dataset(config, descriptors):
    xyz, xanes, index = data_learn(
        config["xyz_path"], config["xanes_path"], descriptors
    )

    return [xyz, xanes]


def init_model(config, xyz, xanes):
    config["model"]["params"]["x_data"] = xyz
    config["model"]["params"]["y_data"] = xanes

    model = create_model(config.get("model")["type"], **config.get("model")["params"])

    return model


def init_scheme(config, xyz, xanes):
    kwargs = {
        "model": config["model"],
        "hyper_params": config["hyperparams"],
        "kfold": config["kfold"],
        "kfold_params": config["kfold_params"],
        "bootstrap_params": config["bootstrap_params"],
        "ensemble_params": config["ensemble_params"],
        "scheduler": config["lr_scheduler"],
        "scheduler_params": config["scheduler_params"],
        "optuna": config["optuna"],
        "optuna_params": config["optuna_params"],
        "freeze": config["freeze"],
        "freeze_params": config["freeze_params"],
        "scaler": config["standardscaler"],
    }

    scheme = create_learn_scheme(xyz, xanes, **kwargs)

    return scheme


class TestSaveModels:
    with open("tests/inputs/in_mlp2.yaml", "r") as f:
        config = yaml.safe_load(f)

    path = Path("tests/save")

    @pytest.fixture(scope="class")
    def descriptors(self):
        return init_descriptors(self.config)

    @pytest.fixture(scope="class")
    def dataset(self, descriptors):
        return init_dataset(self.config, descriptors)

    @pytest.fixture(scope="class")
    def scheme(self, dataset):
        return init_scheme(self.config, dataset[0], dataset[1])

    def test_save_single(self, scheme, descriptors):
        metadata = {
            "mode": "train_xyz",
            "model_type": self.config["model"]["type"],
            "scheme": "std",
        }
        models = []
        models.append(scheme.train_std())
        save_models(self.path, models, descriptors, metadata)

        # Check directory
        save_dir = Path("tests/save/mlp_std_xyz_001")
        assert save_dir.exists()

        # Check files
        files = [
            save_dir / "descriptor0_wacsf.pickle",
            save_dir / "descriptor1_wacsf.pickle",
            save_dir / "metadata.yaml",
            save_dir / "model.pt",
        ]
        for file in files:
            assert file.exists()
        # Clean up
        shutil.rmtree(self.path)

    def test_save_multiple(self, scheme, descriptors):
        metadata = {
            "mode": "train_xyz",
            "model_type": self.config["model"]["type"],
            "scheme": "ensemble",
        }
        models = scheme.train_ensemble()
        save_models(self.path, models, descriptors, metadata)

        # Check directory
        save_dir = Path("tests/save/mlp_ensemble_xyz_001")
        assert save_dir.exists()
        model1_dir = Path("tests/save/mlp_ensemble_xyz_001/model_001")
        assert model1_dir.exists()
        model2_dir = Path("tests/save/mlp_ensemble_xyz_001/model_002")
        assert model2_dir.exists()
        model3_dir = Path("tests/save/mlp_ensemble_xyz_001/model_003")
        assert model3_dir.exists()

        # Check files
        files = [
            save_dir / "descriptor0_wacsf.pickle",
            save_dir / "descriptor1_wacsf.pickle",
            save_dir / "metadata.yaml",
            model1_dir / "model.pt",
            model2_dir / "model.pt",
            model3_dir / "model.pt",
        ]

        for file in files:
            assert file.exists()
        # Clean up
        shutil.rmtree(self.path)


config = {
    "xyz_path": "tests/data/xyz_predict",
    "xanes_path": "tests/data/xanes_predict",
}


def init_predict_scheme(model_dir, mode):
    # Load metadata
    metadata_file = Path(f"{model_dir}/metadata.yaml")
    with open(metadata_file, "r") as f:
        metadata = yaml.safe_load(f)
    model_name = metadata["model_type"]

    # Load descriptor list
    descriptor_list = load_descriptors(model_dir)

    # Encode prediction dataset with saved descriptors
    xyz, xanes, e, index = data_predict(
        config["xyz_path"], config["xanes_path"], descriptor_list, mode, False
    )

    kwargs = {
        "pred_mode": mode,
        "pred_eval": False,
        "scaler": metadata["standardscaler"],
        "fourier": metadata["fourier_transform"],
        "fourier_param": metadata["fourier_param"],
    }

    # Initialise prediction scheme using kwargs
    scheme = create_predict_scheme(model_name, xyz, xanes, **kwargs)

    return scheme, index, e


def load_model(model_dir: Path):
    return torch.load(model_dir / "model.pt", map_location=torch.device("cpu"))


class TestSavePredict:
    path = Path("tests/outputs")

    def test_save_xanes(self):
        model_dir = Path("tests/models/model_mlp_xyz")
        mode = "predict_xanes"
        scheme, index, e = init_predict_scheme(model_dir, mode)
        model = load_model(model_dir)
        result = scheme.predict_std(model)
        save_predict(self.path, mode, result, index, e, scheme.recon_flag)
        # Check directory
        save_dir = Path("tests/outputs/xanes_pred")
        assert save_dir.exists()

        # Check files
        files = [
            save_dir / "0924.txt",
            save_dir / "0925.txt",
            save_dir / "0929.txt",
            save_dir / "0932.txt",
            save_dir / "0935.txt",
        ]
        for file in files:
            assert file.exists()

        # Check file content
        expected_lines = [
            "FDMNES\n",
            "energy <xanes> <std>\n",
            "0.00      -2.7976468E-02 0.0000000E+00  \n",
        ]

        with open(save_dir / "0924.txt", "r") as file:
            # Read the first three lines
            first_three_lines = [file.readline() for _ in range(3)]

            # Check if the first three lines match the expected lines
        for expected, actual in zip(expected_lines, first_three_lines):
            assert expected == actual

        shutil.rmtree(self.path)

    def test_save_xyz(self):
        model_dir = Path("tests/models/model_mlp_xanes")
        mode = "predict_xyz"
        scheme, index, e = init_predict_scheme(model_dir, mode)
        model = load_model(model_dir)
        result = scheme.predict_std(model)
        save_predict(self.path, mode, result, index, e, scheme.recon_flag)
        # Check directory
        save_dir = Path("tests/outputs/xyz_pred")
        assert save_dir.exists()

        # Check files
        files = [
            save_dir / "0924.txt",
            save_dir / "0925.txt",
            save_dir / "0929.txt",
            save_dir / "0932.txt",
        ]
        for file in files:
            assert file.exists()

        # Check file content
        expected_lines = [
            "<xyz> <std>\n",
            "3.5448174E+00  0.0000000E+00  \n",
        ]

        with open(save_dir / "0924.txt", "r") as file:
            # Read the first three lines
            first_three_lines = [file.readline() for _ in range(3)]

            # Check if the first three lines match the expected lines
        for expected, actual in zip(expected_lines, first_three_lines):
            assert expected == actual

        shutil.rmtree(self.path)

    def test_save_all(self):
        model_dir = Path("tests/models/model_aegan")
        mode = "predict_all"
        scheme, index, e = init_predict_scheme(model_dir, mode)
        model = load_model(model_dir)
        result = scheme.predict_std(model)
        save_predict(self.path, mode, result, index, e, scheme.recon_flag)
        # Check directory
        xanes_pred_dir = Path("tests/outputs/xanes_pred")
        assert xanes_pred_dir.exists()
        xanes_recon_dir = Path("tests/outputs/xanes_recon")
        assert xanes_recon_dir.exists()
        xyz_pred_dir = Path("tests/outputs/xyz_pred")
        assert xyz_pred_dir.exists()
        xyz_recon_dir = Path("tests/outputs/xyz_recon")
        assert xyz_recon_dir.exists()

        # Check files
        files = [
            xanes_pred_dir / "0924.txt",
            xanes_pred_dir / "0925.txt",
            xanes_pred_dir / "0929.txt",
            xanes_pred_dir / "0932.txt",
        ]
        for file in files:
            assert file.exists()

        shutil.rmtree(self.path)
