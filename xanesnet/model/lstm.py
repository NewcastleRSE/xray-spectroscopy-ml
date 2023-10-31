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

from xanesnet.model_utils import ActivationSwitch


class LSTM(nn.Module):
    def __init__(
        self,
        hidden_size,
        hidden_out_size,
        num_layers,
        dropout,
        activation,
        x_data,
        y_data,
    ):
        super(LSTM, self).__init__()

        input_size = x_data.shape[1]
        output_size = y_data[0].size

        activation_switch = ActivationSwitch()
        act_fn = activation_switch.fn(activation)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_out_size),
            act_fn(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_out_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        out = self.fc(x)

        return out
