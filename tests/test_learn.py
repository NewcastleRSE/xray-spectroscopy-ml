import os
import shutil

import pytest
import torch
import yaml

from xanesnet.data_encoding import data_gnn_learn, data_learn
from xanesnet.creator import create_descriptor, create_learn_scheme, create_model
from xanesnet.scheme import NNLearn, AELearn, AEGANLearn


def init_dataset(config):
    descriptor_list = []
    descriptors = config.get("descriptors")

    for d in descriptors:
        descriptor = create_descriptor(d["type"], **d["params"])
        descriptor_list.append(descriptor)

    xyz, xanes, index = data_learn(
        config["xyz_path"], config["xanes_path"], descriptor_list
    )

    return [xyz, xanes]


def init_dataset_gnn(config):
    # Remove existing graph data
    graph_path = os.path.join(config["xyz_path"], "graph")
    if os.path.exists(graph_path):
        shutil.rmtree(graph_path)

    descriptor_list = []
    for d in config["descriptors"]:
        descriptor = create_descriptor(d["type"], **d["params"])
        descriptor_list.append(descriptor)

    dataset = data_gnn_learn(
        config["xyz_path"],
        config["xanes_path"],
        config["model"]["node_features"],
        config["model"]["edge_features"],
        descriptor_list,
        config["fourier_transform"],
        config["fourier_params"],
    )
    return dataset


def init_model(config, xyz, xanes):
    config["model"]["params"]["x_data"] = xyz
    config["model"]["params"]["y_data"] = xanes

    model = create_model(config.get("model")["type"], **config.get("model")["params"])

    return model


def init_model_gnn(config, xyz):
    config.get("model")["params"]["x_data"] = xyz
    config.get("model")["params"]["mlp_feat_size"] = xyz[0].graph_attr.shape[0]

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


class TestNNLearn:
    with open("tests/inputs/in_mlp.yaml", "r") as f:
        config = yaml.safe_load(f)

    @pytest.fixture(scope="class")
    def dataset(self):
        return init_dataset(self.config)

    @pytest.fixture(scope="class")
    def scheme(self, dataset):
        return init_scheme(self.config, dataset[0], dataset[1])

    @pytest.fixture(scope="class")
    def model(self, dataset):
        return init_model(self.config, dataset[0], dataset[1])

    def test_scheme_instance(self, scheme):
        assert isinstance(scheme, NNLearn)

    def test_single_step(self, scheme, model):
        model.to(scheme.device)

        initial_weights = [p.clone() for p in model.parameters()]
        model = scheme.setup_weight(model, scheme.weight_seed)
        output, _ = scheme.train(model, scheme.x_data, scheme.y_data)

        # Check if the model is on the expected device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert next(output.parameters()).device == device

        # Check that weight tensor has changed
        for initial, updated in zip(initial_weights, model.parameters()):
            assert not torch.equal(initial, updated)

        # Check gradients are non-zero
        for param in model.parameters():
            assert param.grad is None or param.grad.abs().sum().item() > 0

    def test_std_train(self, scheme):
        model = scheme.train_std()
        assert model is not None

    def test_kfold_train(self, scheme):
        model = scheme.train_kfold()
        assert model is not None

    def test_bootstrap(self, scheme):
        model_list = scheme.train_bootstrap()
        # Check # of models
        assert len(model_list) == 3
        # Check that each model has different weights
        check_weight_diff(model_list)

    def test_ensemble(self, scheme):
        model_list = scheme.train_ensemble()
        # Check # of models
        assert len(model_list) == 3
        # Check that each model has different weights
        check_weight_diff(model_list)


class TestAELearn:
    with open("tests/inputs/in_ae_mlp.yaml", "r") as f:
        config = yaml.safe_load(f)

    @pytest.fixture(scope="class")
    def dataset(self):
        return init_dataset(self.config)

    @pytest.fixture(scope="class")
    def scheme(self, dataset):
        return init_scheme(self.config, dataset[0], dataset[1])

    @pytest.fixture(scope="class")
    def model(self, dataset):
        return init_model(self.config, dataset[0], dataset[1])

    def test_scheme_instance(self, scheme):
        assert isinstance(scheme, AELearn)

    def test_single_step(self, scheme, model):
        model.to(scheme.device)

        initial_weights = [p.clone() for p in model.parameters()]
        model = scheme.setup_weight(model, scheme.weight_seed)
        output, _ = scheme.train(model, scheme.x_data, scheme.y_data)

        # Check if the model is on the expected device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert next(output.parameters()).device == device

        # Check that weight tensor has changed
        for initial, updated in zip(initial_weights, model.parameters()):
            assert not torch.equal(initial, updated)

        # Check gradients are non-zero
        for param in model.parameters():
            assert param.grad is None or param.grad.abs().sum().item() > 0

    def test_std_train(self, scheme):
        model = scheme.train_std()
        assert model is not None

    def test_kfold_train(self, scheme):
        model = scheme.train_kfold()
        assert model is not None

    def test_bootstrap(self, scheme):
        model_list = scheme.train_bootstrap()
        # Check # of models
        assert len(model_list) == 3
        # Check that each model has different weights
        check_weight_diff(model_list)

    def test_ensemble(self, scheme):
        model_list = scheme.train_ensemble()
        # Check # of models
        assert len(model_list) == 3
        # Check that each model has different weights
        check_weight_diff(model_list)


class TestAEGANLearn:
    with open("tests/inputs/in_aegan_mlp.yaml", "r") as f:
        config = yaml.safe_load(f)

    @pytest.fixture(scope="class")
    def dataset(self):
        return init_dataset(self.config)

    @pytest.fixture(scope="class")
    def scheme(self, dataset):
        return init_scheme(self.config, dataset[0], dataset[1])

    @pytest.fixture(scope="class")
    def model(self, dataset):
        return init_model(self.config, dataset[0], dataset[1])

    def test_scheme_instance(self, scheme):
        assert isinstance(scheme, AEGANLearn)

    def test_single_step(self, scheme, model):
        model.to(scheme.device)

        initial_weights = [p.clone() for p in model.parameters()]
        model = scheme.setup_weight(model, scheme.weight_seed)
        output, _ = scheme.train(model, scheme.x_data, scheme.y_data)

        # Check if the model is on the expected device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert next(output.parameters()).device == device

        # Check that weight tensor has changed
        for initial, updated in zip(initial_weights, model.parameters()):
            assert not torch.equal(initial, updated)

        # Check gradients are non-zero
        for param in model.parameters():
            assert param.grad is None or param.grad.abs().sum().item() > 0

    def test_std_train(self, scheme):
        model = scheme.train_std()
        assert model is not None

    def test_kfold_train(self, scheme):
        model = scheme.train_kfold()
        assert model is not None

    def test_bootstrap(self, scheme):
        model_list = scheme.train_bootstrap()
        # Check # of models
        assert len(model_list) == 3
        # Check that each model has different weights
        check_weight_diff(model_list)

    def test_ensemble(self, scheme):
        model_list = scheme.train_ensemble()
        # Check # of models
        assert len(model_list) == 3
        # Check that each model has different weights
        check_weight_diff(model_list)


class TestGNNLearn:
    with open("tests/inputs/in_gnn.yaml", "r") as f:
        config = yaml.safe_load(f)

    @pytest.fixture(scope="class")
    def dataset(self):
        return init_dataset_gnn(self.config)

    @pytest.fixture(scope="class")
    def scheme(self, dataset):
        return init_scheme(self.config, dataset, None)

    @pytest.fixture(scope="class")
    def model(self, dataset):
        return init_model_gnn(self.config, dataset)

    def test_single_step(self, scheme, model):
        model.to(scheme.device)

        initial_weights = [p.clone() for p in model.parameters()]
        model = scheme.setup_weight(model, scheme.weight_seed)
        output, _ = scheme.train(model, scheme.x_data)

        # Check if the model is on the expected device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert next(output.parameters()).device == device

        # Check that weight tensor has changed
        for initial, updated in zip(initial_weights, model.parameters()):
            assert not torch.equal(initial, updated)

        # Check gradients are non-zero
        for param in model.parameters():
            assert param.grad is None or param.grad.abs().sum().item() > 0

    def test_std_train(self, scheme):
        model = scheme.train_std()
        assert model is not None

    def test_kfold_train(self, scheme):
        model = scheme.train_kfold()
        assert model is not None

    def test_bootstrap(self, scheme):
        model_list = scheme.train_bootstrap()
        # Check # of models
        assert len(model_list) == 3
        # Check that each model has different weights
        check_weight_diff(model_list)

    def test_ensemble(self, scheme):
        model_list = scheme.train_ensemble()
        # Check # of models
        assert len(model_list) == 3
        # Check that each model has different weights
        check_weight_diff(model_list)


def check_weight_diff(model_list):
    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            model_1 = model_list[i]
            model_2 = model_list[j]

            for param1, param2 in zip(model_1.parameters(), model_2.parameters()):  #
                # opt out activation layer
                if param1.shape == torch.Size([1]) and param2.shape == torch.Size([1]):
                    continue
                assert not torch.equal(param1, param2)
