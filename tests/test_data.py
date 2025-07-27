import os
import shutil

import pytest

from torch_geometric.utils import to_networkx
from networkx.algorithms import is_connected

from xanesnet.utils.encode import (
    data_gnn_learn,
    data_learn,
    data_predict,
    data_gnn_predict,
)
from xanesnet.creator import create_descriptor

xyz_path = "tests/data/xyz"
xanes_path = "tests/data/xanes"

# input config
config_descriptors = {
    "descriptors": [
        {
            "type": "wacsf",
            "params": {"r_min": 0.5, "r_max": 6.5, "n_g2": 10, "n_g4": 10},
        },
        {
            "type": "wacsf",
            "params": {"r_min": 0.5, "r_max": 6.5, "n_g2": 10, "n_g4": 10},
        },
    ],
}

config_gnn = {
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


@pytest.fixture(scope="module")
def descriptors():
    descriptor_list = []
    for d in config_descriptors["descriptors"]:
        descriptor = create_descriptor(d["type"], **d["params"])
        descriptor_list.append(descriptor)

    return descriptor_list


class TestDataLearn:
    @pytest.fixture(scope="class")
    def dataset(self, descriptors):
        xyz, xanes, index = data_learn(
            xyz_path,
            xanes_path,
            descriptors,
        )

        return xyz, xanes, index

    def test_xyz(self, dataset):
        xyz = dataset[0]
        # Check data shape
        assert xyz.shape == (8, 42)
        # Check data concatenation
        assert xyz[0][0] == xyz[0][21]

    def test_xanes(self, dataset):
        xanes = dataset[1]
        # Check data shape
        assert xanes.shape == (8, 400)

    def test_index(self, dataset):
        # Check data index
        index = dataset[2]
        assert index == ["0853", "0855", "0860", "0861", "0874", "0882", "0887", "0892"]


class TestDataPredict:
    def test_case_all(self, descriptors):
        xyz, xanes, e, index = data_predict(
            xyz_path,
            xanes_path,
            descriptors,
            "predict_all",
            False,
        )
        # Check data shape and index
        assert xyz.shape == (8, 42)
        assert xanes.shape == (8, 400)
        assert len(e) == 400
        assert index == ["0853", "0855", "0860", "0861", "0874", "0882", "0887", "0892"]

    def test_case_xyz(self, descriptors):
        xyz, xanes, e, index = data_predict(
            xyz_path,
            xanes_path,
            descriptors,
            "predict_xyz",
            False,
        )
        # Check data shape and index
        assert xyz is None
        assert xanes.shape == (8, 400)
        assert len(e) == 400
        assert index == ["0853", "0855", "0860", "0861", "0874", "0882", "0887", "0892"]

    def test_case_xanes(self, descriptors):
        xyz, xanes, e, index = data_predict(
            xyz_path,
            xanes_path,
            descriptors,
            "predict_xanes",
            False,
        )
        # Check data shape and index
        assert xyz.shape == (10, 42)
        assert xanes is None
        assert e is None
        assert index == [
            "0853",
            "0855",
            "0860",
            "0861",
            "0865",
            "0874",
            "0882",
            "0887",
            "0892",
            "0911",
        ]

    def test_case_eval(self, descriptors):
        xyz, xanes, e, index = data_predict(
            xyz_path,
            xanes_path,
            descriptors,
            "",
            True,
        )

        # Check data shape and index
        assert xyz.shape == (8, 42)
        assert xanes.shape == (8, 400)
        assert index == ["0853", "0855", "0860", "0861", "0874", "0882", "0887", "0892"]


class TestDataGNNLearn:
    @pytest.fixture(scope="class")
    def dataset(self, descriptors):
        graphs = data_gnn_learn(
            xyz_path,
            xanes_path,
            config_gnn["model"]["node_features"],
            config_gnn["model"]["edge_features"],
            descriptors,
            config_gnn["fourier_transform"],
            config_gnn["fourier_params"],
        )

        return graphs

    def test_graph_structure(self, dataset):
        # Graph 0853.xyz
        graph = dataset[0]
        # Check if the # of node matches the atom count
        assert graph.num_nodes == 11
        # Check if the # of edges matches the bond count
        assert graph.num_edges == 24
        # Check for the connectivity
        G = to_networkx(graph, to_undirected=True)
        assert is_connected(G)

    def test_node_feats(self, dataset):
        graph = dataset[0]
        # Check for the one-hot encoding
        assert graph.x.shape == (graph.num_nodes, 20)

    def test_edge_feats(self, dataset):
        graph = dataset[0]
        assert graph.edge_attr.shape == (graph.num_edges, 16)
        # Check all elements are greater than zero
        assert all(all(x > 0 for x in row) for row in graph.edge_attr)

    def test_graph_feats(self, dataset):
        graph = dataset[0]
        # Check if the graph feature length matches the total descriptor feature length
        assert len(graph.graph_attr) == 42

    def test_label(self, dataset):
        graph = dataset[0]
        # Check for label length
        assert len(graph.y) == 400
        # Check if the graph and label are consistent at lines 100, and 200
        assert pytest.approx(graph.y[97].item(), rel=1e-7) == 0.09866492
        assert pytest.approx(graph.y[197].item(), rel=1e-7) == 0.05046986

    def test_processed_file(self, dataset):
        # Check correctness of saved graphs
        assert len(dataset.processed_file_names) == 8
        assert dataset.processed_file_names[0] == "0_0853.pt"
        assert dataset.processed_file_names[7] == "7_0892.pt"

    def test_len(self, dataset):
        # Check len() fn
        assert len(dataset) == 8

    def test_get(self, dataset):
        # Check get(idx) fn
        assert dataset.get(0).x.shape == (11, 20)
        assert dataset.get(7).x.shape == (14, 20)

    def test_cleanup(self):
        # Remove graph dataset
        graph_path = os.path.join(xyz_path, "graph")
        if os.path.exists(graph_path):
            shutil.rmtree(graph_path)


class TestDataGNNPredict:
    def test_case_no_eval(self, descriptors):
        graphs, index, xanes, e = data_gnn_predict(
            xyz_path,
            xanes_path,
            config_gnn["model"]["node_features"],
            config_gnn["model"]["edge_features"],
            descriptors,
            False,
        )
        # Check data shape and index
        assert len(graphs) == 10
        assert len(index) == 10
        assert xanes is None
        assert e is None

    def test_case_eval(self, descriptors):
        graphs, index, xanes, e = data_gnn_predict(
            xyz_path,
            xanes_path,
            config_gnn["model"]["node_features"],
            config_gnn["model"]["edge_features"],
            descriptors,
            True,
        )
        # Check data shape and index
        assert len(graphs) == 8
        assert len(index) == 8
        assert xanes.shape == (8, 400)
        assert len(e) == 400

    def test_cleanup(self):
        # Remove graph dataset
        graph_path = os.path.join(xyz_path, "graph")
        if os.path.exists(graph_path):
            shutil.rmtree(graph_path)
