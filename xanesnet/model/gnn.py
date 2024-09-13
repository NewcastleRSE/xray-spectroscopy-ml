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
        hidden_size: int,
        dropout: float,
        num_hidden_layers: int,
        activation: str,
        layer_name: str,
        x_data,
        y_data,
    ):
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        self.nn_flag = 1

        layers = []
        input_size = x_data[0].x.shape[1]
        output_size = x_data[0].y.shape[0]
        # Instantiate ActivationSwitch for dynamic activation selection
        activation_switch = ActivationSwitch()
        act_fn = activation_switch.fn(activation)

#       # Construct hidden layers
        num_heads = 4
        for i in range(num_hidden_layers - 1):
            layers += [
                gnn_layer(in_channels=input_size, out_channels=hidden_size, heads=num_heads, concat=True, edge_dim=16),
                nn.BatchNorm1d(hidden_size * num_heads),  
                act_fn(),
                nn.Dropout(dropout),
            ]
            input_size = hidden_size * num_heads  

        # Construct output layer
        layers += [gnn_layer(in_channels=input_size, out_channels=hidden_size, heads=num_heads, concat=True, edge_dim=16)]
        self.layers = nn.ModuleList(layers)


        # Construct final MLP layers
        layers = []

        num_hidden_layers = 3
        for i in range(num_hidden_layers - 1):
            if i == 0:
                layer = nn.Sequential(
                    nn.Linear(hidden_size * num_heads, hidden_size * num_heads * 2),
                    nn.Dropout(dropout),
                    act_fn(),
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(
                        int(hidden_size * num_heads * 2),
                        int(hidden_size * num_heads * 2),
                    ),
                    nn.Dropout(dropout),
                    act_fn(),
                )

            layers.append(layer)

        output_layer = nn.Sequential(
            nn.Linear(hidden_size * num_heads * 2, output_size),
            nn.Dropout(dropout),
            act_fn(),
        )
        layers.append(output_layer)

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_attr, edge_idx, batch_idx) -> torch.Tensor:
        """
        Inputs:
            x - Input features per node
            edge_index - List of edge index pairs
            batch_idx - Index of batch element for each node
        """
        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                if isinstance(layer, geom_nn.GATv2Conv) or (isinstance(layer, geom_nn.GATConv) and edge_attr is not None):
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
        out = self.head(x)

        return out
