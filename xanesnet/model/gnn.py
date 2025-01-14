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

import numpy as np
import torch
from torch import nn
import torch_geometric.nn as geom_nn

from xanesnet.model.base_model import Model
from xanesnet.utils_model import ActivationSwitch

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GATv2": geom_nn.GATv2Conv,
    "GraphConv": geom_nn.GraphConv,
}


class GNN(Model):
    """
    A class for constructing a customisable GNN (Graph Neural Network) model.
    """

    def __init__(
        self,
        x_data: np.ndarray,
        layer_name: str,
        layer_params: dict,
        hidden_size: int,
        mlp_feat_size: int,
        dropout: float = 0.2,
        num_hidden_layers: int = 5,
        activation: str = "prelu",
    ):
        """
        Args:
            x_data (NumPy array): Input data for the network
            layer_name (string): Name of GNN layer (GAT, GATv2, GCN, GraphConv)
            layer_params (dict): parameters pass to GNN layers
            hidden_size (integer): Size of the hidden layer.
            mlp_feat_size (integer): Size of features added to the MLP layers.
                This should be the sum of feature sizes from all descriptors
            dropout (float): If none-zero, add dropout layer on the outputs
                of each hidden layer with dropout probability equal to dropout.
            num_hidden_layers (integer): Number of hidden layers
                in the network.
            activation (string): Name of activation function applied
                to the hidden layers.
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        self.nn_flag = 1
        input_size = x_data[0].x.shape[1]
        output_size = x_data[0].y.shape[0]
        heads = 1
        if layer_params is None:
            layer_params = {}
        elif "heads" in layer_params:
            heads = layer_params["heads"]

        # Instantiate ActivationSwitch for dynamic activation selection
        activation_switch = ActivationSwitch()
        act_fn = activation_switch.fn(activation)

        # Construct hidden layers
        layers = []
        for i in range(num_hidden_layers - 1):
            layers += [
                gnn_layer(
                    in_channels=input_size, out_channels=hidden_size, **layer_params
                ),
                nn.BatchNorm1d(hidden_size * heads),
                act_fn(),
                nn.Dropout(dropout),
            ]

            input_size = hidden_size * heads

        # Construct output layer
        layers += [
            gnn_layer(in_channels=input_size, out_channels=hidden_size, **layer_params)
        ]
        self.layers = nn.ModuleList(layers)

        # Construct final MLP layers
        layers = []

        num_hidden_layers = 3
        mlp_hidden_size = (hidden_size * heads) + mlp_feat_size

        for i in range(num_hidden_layers - 1):
            if i == 0:
                layer = nn.Sequential(
                    nn.Linear(mlp_hidden_size, mlp_hidden_size * 2),
                    nn.Dropout(dropout),
                    act_fn(),
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(
                        int(mlp_hidden_size * 2),
                        int(mlp_hidden_size * 2),
                    ),
                    nn.Dropout(dropout),
                    act_fn(),
                )

            layers.append(layer)

        output_layer = nn.Sequential(
            nn.Linear(mlp_hidden_size * 2, output_size),
            nn.Dropout(dropout),
            act_fn(),
        )
        layers.append(output_layer)

        self.head = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, edge_attr, graph_attr, edge_idx, batch_idx
    ) -> torch.Tensor:
        """
        Inputs:
            x - Input features per node
            edge_index - List of edge index pairs
            batch_idx - Index of batch element for each node
        """
        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                if isinstance(layer, geom_nn.GATv2Conv) or (
                    isinstance(layer, geom_nn.GATConv) and edge_attr is not None
                ):
                    x = layer(x, edge_idx, edge_attr)
                else:
                    x = layer(x, edge_idx)
            else:
                x = layer(x)

        #       # Specific node
        #       node_idx = 0
        #       node_feature = x[node_idx]
        #       out = self.head(node_feature)
        # Average pooling
        x = geom_nn.global_mean_pool(x, batch_idx)
        # Append graph feature before sending to mlp layers
        x = torch.cat((x, graph_attr), dim=1)
        out = self.head(x)

        return out
