import os
import shutil

import torch
import torch_geometric.nn as geom_nn

from torch import nn

from xanesnet.creator import create_descriptor, create_model
from xanesnet.utils.encode import data_learn, data_gnn_learn
from xanesnet.model import MLP, CNN, LSTM, GNN, AE_MLP, AE_CNN


def init_dataset(config, train_xyz):
    descriptor_list = []
    descriptors = config.get("descriptors")

    for d in descriptors:
        descriptor = create_descriptor(d["type"], **d["params"])
        descriptor_list.append(descriptor)

    xyz, xanes, index = data_learn(
        config["xyz_path"], config["xanes_path"], descriptor_list
    )

    if train_xyz:
        x_data = xyz
        y_data = xanes
    else:
        x_data = xanes
        y_data = xyz

    return [x_data, y_data]


def init_model(config, x_data, y_data):
    config["model"]["params"]["in_size"] = x_data.shape[1]
    config["model"]["params"]["out_size"] = y_data.shape[1]

    model = create_model(config.get("model")["type"], **config.get("model")["params"])

    return model


def init_dataset_gnn(config):
    graph_path = os.path.join(config["xyz_path"], "graph")

    # Remove existing graph data
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


def init_model_gnn(config, dataset):
    config.get("model")["params"]["in_size"] = dataset[0].x.shape[1]
    config.get("model")["params"]["out_size"] = dataset[0].y.shape[0]
    config.get("model")["params"]["mlp_feat_size"] = dataset[0].graph_attr.shape[0]
    model = create_model(config.get("model")["type"], **config.get("model")["params"])
    # model = learn_scheme.setup_weight(model, config.get("hyperparams")["weight_seed"])
    return model


class TestModelMLP:
    # input config
    config_mlp = {
        "xyz_path": "tests/data/xyz",
        "xanes_path": "tests/data/xanes",
        "descriptors": [
            {
                "type": "wacsf",
                "params": {"r_min": 1.0, "r_max": 6.0, "n_g2": 16, "n_g4": 32},
            },
        ],
        "model": {
            "type": "mlp",
            "params": {
                "hidden_size": 512,
                "dropout": 0.2,
                "num_hidden_layers": 3,
                "activation": "prelu",
                "shrink_rate": 0.5,
            },
        },
    }

    def test_arch_xyz(self):
        # Test on train_xyz mode
        x_data, y_data = init_dataset(self.config_mlp, train_xyz=True)
        model = init_model(self.config_mlp, x_data, y_data)
        # Check model type
        assert isinstance(model, MLP)
        # Check model input size
        assert model.dense_layers[0][0].in_features == 16 + 32 + 1
        # Check param num_hidden_layers
        assert len(model.dense_layers) == 3
        # Check param hidden_size
        assert model.dense_layers[0][0].out_features == 512
        # Check param dropout
        assert model.dense_layers[0][1].p == 0.2
        # Check param activation
        assert isinstance(model.dense_layers[0][2], nn.PReLU)
        # Check param shrink_rate
        assert model.dense_layers[1][0].in_features == 512
        assert model.dense_layers[1][0].out_features == 256
        # Check model output size
        assert model.dense_layers[2][0].out_features == 400
        # Check forward pass
        output = model(torch.from_numpy(x_data).float())
        assert output.shape == (8, 400)
        # Assert no NaNs or Infs in the output
        assert torch.isfinite(output).all()

    def test_arch_xanes(self):
        # test on train_xanes mode
        x_data, y_data = init_dataset(self.config_mlp, train_xyz=False)
        model = init_model(self.config_mlp, x_data, y_data)
        # Check model input size
        assert model.dense_layers[0][0].in_features == 400
        # Check model output size
        assert model.dense_layers[2][0].out_features == 49
        # Check forward pass
        output = model(torch.from_numpy(x_data).float())
        assert output.shape == (8, 49)
        # Assert no NaNs or Infs in the output
        assert torch.isfinite(output).all()


class TestModelCNN:
    # input config
    config_cnn = {
        "xyz_path": "tests/data/xyz",
        "xanes_path": "tests/data/xanes",
        "descriptors": [
            {
                "type": "wacsf",
                "params": {"r_min": 1.0, "r_max": 6.0, "n_g2": 16, "n_g4": 32},
            },
        ],
        "model": {
            "type": "cnn",
            "params": {
                "hidden_size": 512,
                "dropout": 0.2,
                "num_conv_layers": 2,
                "activation": "prelu",
                "out_channel": 32,
                "channel_mul": 2,
                "kernel_size": 3,
                "stride": 1,
            },
        },
    }

    def test_arch_xyz(self):
        # Test on train_xyz mode
        x_data, y_data = init_dataset(self.config_cnn, train_xyz=True)
        model = init_model(self.config_cnn, x_data, y_data)
        # Check model type
        assert isinstance(model, CNN)
        # Check param num_conv_layers
        assert len(model.conv_layers) == 2
        # Check param out_channel
        assert model.conv_layers[0][0].out_channels == 32
        # Check param channel_mul
        assert model.conv_layers[1][0].out_channels == 64
        # Check param kernel_size
        assert model.conv_layers[0][0].kernel_size == (3,)
        # Check param stride
        assert model.conv_layers[0][0].stride == (1,)
        # Check param dropout rate
        assert model.conv_layers[0][3].p == 0.2
        # Check param activation
        assert isinstance(model.conv_layers[0][2], nn.PReLU)
        # Check dense layer dimension
        assert model.dense_layers[0][0].in_features == 2880
        assert model.dense_layers[1][0].in_features == 512
        # Check model output size
        assert model.dense_layers[1][0].out_features == 400
        # Check forward pass
        output = model(torch.from_numpy(x_data).float())
        assert output.shape == (8, 400)
        # Assert no NaNs or Infs in the output
        assert torch.isfinite(output).all()

    def test_arch_xanes(self):
        # test on train_xanes mode
        x_data, y_data = init_dataset(self.config_cnn, train_xyz=False)
        model = init_model(self.config_cnn, x_data, y_data)
        # Check dense layer dimension
        assert model.dense_layers[0][0].in_features == 25344
        # Check model output size
        assert model.dense_layers[1][0].out_features == 49
        # Check forward pass
        output = model(torch.from_numpy(x_data).float())
        assert output.shape == (8, 49)
        # Assert no NaNs or Infs in the output
        assert torch.isfinite(output).all()


class TestModelLSTM:
    # input config
    config_lstm = {
        "xyz_path": "tests/data/xyz",
        "xanes_path": "tests/data/xanes",
        "descriptors": [
            {
                "type": "wacsf",
                "params": {"r_min": 1.0, "r_max": 6.0, "n_g2": 16, "n_g4": 32},
            },
        ],
        "model": {
            "type": "lstm",
            "params": {
                "hidden_size": 100,
                "hidden_out_size": 50,
                "num_layers": 1,
                "activation": "prelu",
                "dropout": 0.2,
            },
        },
    }

    def test_arch_xyz(self):
        # Test on train_xyz mode
        x_data, y_data = init_dataset(self.config_lstm, train_xyz=True)
        model = init_model(self.config_lstm, x_data, y_data)
        # Check model type
        assert isinstance(model, LSTM)
        # Check model input size
        assert model.lstm.input_size == 49
        # Check param hidden_size size, num_layers
        assert model.lstm.hidden_size == 100
        assert model.lstm.num_layers == 1
        # Check param hidden_out_size, activation, dropout
        assert model.dense_layers[0].out_features == 50
        assert isinstance(model.dense_layers[1], nn.PReLU)
        assert model.dense_layers[2].p == 0.2
        # Check model output size
        assert model.dense_layers[3].out_features == 400
        # Check forward pass
        output = model(torch.from_numpy(x_data).float())
        assert output.shape == (8, 400)
        # Assert no NaNs or Infs in the output
        assert torch.isfinite(output).all()

    def test_arch_xanes(self):
        # test on train_xanes mode
        x_data, y_data = init_dataset(self.config_lstm, train_xyz=False)
        model = init_model(self.config_lstm, x_data, y_data)
        # Check model input size
        assert model.lstm.input_size == 400
        # Check model output size
        assert model.dense_layers[3].out_features == 49
        # Check forward pass
        output = model(torch.from_numpy(x_data).float())
        assert output.shape == (8, 49)
        # Assert no NaNs or Infs in the output
        assert torch.isfinite(output).all()


class TestModelGNN:
    config_gnn = {
        "xyz_path": "tests/data/xyz",
        "xanes_path": "tests/data/xanes",
        "descriptors": [
            {
                "type": "wacsf",
                "params": {"r_min": 1.0, "r_max": 6.0, "n_g2": 16, "n_g4": 32},
            },
        ],
        "model": {
            "type": "gnn",
            "node_features": {},
            "edge_features": {"n": 16, "r_min": 0.0, "r_max": 4.0},
            "params": {
                "hidden_size": 512,
                "dropout": 0.2,
                "num_hidden_layers": 2,
                "activation": "prelu",
                "layer_name": "GATv2",
                "layer_params": {"heads": 2, "concat": True, "edge_dim": 16},
            },
        },
        "fourier_transform": False,
        "fourier_params": {"concat": True},
    }

    def test_arch_xyz(self):
        # Test on train_xyz mode
        dataset = init_dataset_gnn(self.config_gnn)
        model = init_model_gnn(self.config_gnn, dataset)
        # Check model type
        assert isinstance(model, GNN)
        # Check model input size
        assert model.gnn_layers[0].in_channels == 20
        # Check params num_hidden_layers, layer_name, hidden_size
        assert len(model.gnn_layers) == 5
        assert isinstance(model.gnn_layers[0], geom_nn.GATv2Conv)
        assert model.gnn_layers[0].out_channels == 512
        # Check param activation, dropout
        assert isinstance(model.gnn_layers[2], nn.PReLU)
        assert model.gnn_layers[3].p == 0.2
        # Check layer_params
        assert model.gnn_layers[0].heads == 2
        assert model.gnn_layers[0].edge_dim == 16
        # Check model output size
        assert model.mlp_layers[2][0].out_features == 400
        # Check forward pass
        graph = dataset[0]
        graph_attr = graph.graph_attr.reshape(1, 49)
        output = model(
            graph.x.float(),
            graph.edge_attr.float(),
            graph_attr.float(),
            graph.edge_index,
            torch.zeros(graph.num_nodes, dtype=torch.long),
        )
        # Check if model produces the corrected output shape
        assert output.shape == (1, 400)
        # Assert no NaNs or Infs in the output
        assert torch.isfinite(output).all()


class TestModelAEMLP:
    # input config
    config_aemlp = {
        "xyz_path": "tests/data/xyz",
        "xanes_path": "tests/data/xanes",
        "descriptors": [
            {
                "type": "wacsf",
                "params": {"r_min": 1.0, "r_max": 6.0, "n_g2": 16, "n_g4": 32},
            },
        ],
        "model": {
            "type": "ae_mlp",
            "params": {
                "hidden_size": 512,
                "dropout": 0.2,
                "num_hidden_layers": 3,
                "activation": "prelu",
                "shrink_rate": 0.5,
            },
        },
    }

    def test_arch_xyz(self):
        # Test on train_xyz mode
        x_data, y_data = init_dataset(self.config_aemlp, train_xyz=True)
        model = init_model(self.config_aemlp, x_data, y_data)
        # Check model type
        assert isinstance(model, AE_MLP)
        # Check encoder input size
        assert model.encoder_layers[0][0].in_features == 49
        # Check decoder output size
        assert model.decoder_layers[2][0].out_features == 49
        # Check param num_hidden_layers
        assert len(model.encoder_layers) == 3
        assert len(model.decoder_layers) == 3
        # Check param hidden_size
        assert model.encoder_layers[0][0].out_features == 512
        # Check param dropout
        assert model.dense_layers[0][2].p == 0.2
        # Check param activation
        assert isinstance(model.encoder_layers[0][1], nn.PReLU)
        # Check param shrink_rate
        assert model.encoder_layers[1][0].in_features == 512
        assert model.encoder_layers[1][0].out_features == 256
        # Check model output size
        assert model.dense_layers[1][0].out_features == 400
        # Check forward pass
        recon_input, output = model(torch.from_numpy(x_data).float())
        assert output.shape == (8, 400)
        assert recon_input.shape == (8, 49)
        # Assert no NaNs or Infs in the output
        assert torch.isfinite(output).all()
        assert torch.isfinite(recon_input).all()

    def test_arch_xanes(self):
        # test on train_xanes mode
        x_data, y_data = init_dataset(self.config_aemlp, train_xyz=False)
        model = init_model(self.config_aemlp, x_data, y_data)
        # Check encoder input size
        assert model.encoder_layers[0][0].in_features == 400
        # Check decoder output size
        assert model.decoder_layers[2][0].out_features == 400
        # Check model output size
        assert model.dense_layers[1][0].out_features == 49
        # Check forward pass
        recon_input, output = model(torch.from_numpy(x_data).float())
        assert output.shape == (8, 49)
        assert recon_input.shape == (8, 400)
        # Assert no NaNs or Infs in the output
        assert torch.isfinite(output).all()
        assert torch.isfinite(recon_input).all()


class TestModelAECNN:
    # input config
    config_aecnn = {
        "xyz_path": "tests/data/xyz",
        "xanes_path": "tests/data/xanes",
        "descriptors": [
            {
                "type": "wacsf",
                "params": {"r_min": 1.0, "r_max": 6.0, "n_g2": 16, "n_g4": 32},
            },
        ],
        "model": {
            "type": "ae_cnn",
            "params": {
                "hidden_size": 64,
                "dropout": 0.2,
                "num_conv_layers": 2,
                "activation": "prelu",
                "out_channel": 32,
                "channel_mul": 2,
                "kernel_size": 3,
                "stride": 1,
            },
        },
    }

    def test_arch_xyz(self):
        # Test on train_xyz mode
        x_data, y_data = init_dataset(self.config_aecnn, train_xyz=True)
        model = init_model(self.config_aecnn, x_data, y_data)
        # Check model type
        assert isinstance(model, AE_CNN)
        # Check encoder input size
        assert model.encoder_layers[0][0].in_channels == 1
        assert model.encoder_layers[0][0].out_channels == 32
        # Check decoder output size
        assert model.decoder_layers[1][0].out_channels == 1
        assert model.decoder_layers[1][0].in_channels == 32
        # Check model output size
        assert model.dense_layers[1][0].out_features == 400
        # Check forward pass
        recon_input, output = model(torch.from_numpy(x_data).float())
        assert output.shape == (8, 400)
        assert recon_input.shape == (8, 49)
        # Assert no NaNs or Infs in the output
        assert torch.isfinite(output).all()
        assert torch.isfinite(recon_input).all()

    def test_arch_xanes(self):
        # test on train_xanes mode
        x_data, y_data = init_dataset(self.config_aecnn, train_xyz=False)
        model = init_model(self.config_aecnn, x_data, y_data)
        # Check encoder input size
        assert model.encoder_layers[0][0].in_channels == 1
        assert model.encoder_layers[0][0].out_channels == 32
        # Check decoder output size
        assert model.decoder_layers[1][0].out_channels == 1
        assert model.decoder_layers[1][0].in_channels == 32
        # Check model output size
        assert model.dense_layers[1][0].out_features == 49
        # Check forward pass
        recon_input, output = model(torch.from_numpy(x_data).float())
        assert output.shape == (8, 49)
        assert recon_input.shape == (8, 400)
        # Assert no NaNs or Infs in the output
        assert torch.isfinite(output).all()
        assert torch.isfinite(recon_input).all()


# class TestModelAEGAN:
#     # input config
#     config_aegan = {
#         "xyz_path": "data/xyz",
#         "xanes_path": "data/xanes",
#         "descriptors": [
#             {
#                 "type": "wacsf",
#                 "params": {"r_min": 1.0, "r_max": 6.0, "n_g2": 16, "n_g4": 32},
#             },
#         ],
#         "model": {
#             "type": "aegan_mlp",
#             "params": {
#                 "hidden_size": 256,
#                 "dropout": 0.2,
#                 "n_hl_gen": 2,
#                 "n_hl_shared": 2,
#                 "n_hl_dis": 2,
#                 "activation": "prelu",
#                 "lr_gen": 0.01,
#                 "lr_dis": 0.0001,
#                 "optim_fn_gen": "Adam",
#                 "optim_fn_dis": "Adam",
#                 "loss_gen": {
#                     "loss_fn": "mse",
#                     "loss_args": 10,
#                     "loss_reg_type": "null",
#                     "loss_reg_param": 0.001,
#                 },
#                 "loss_dis": {
#                     "loss_fn": "bce",
#                     "loss_args": "null",
#                     "loss_reg_type": "null",
#                     "loss_reg_param": 0.001,
#                 },
#             },
#         },
#     }
#
#     def test_arch_xyz(self):
#         # Test on train_xyz mode
#         model = init_model(self.config_aegan, train_xyz=True)
#         print(model)
