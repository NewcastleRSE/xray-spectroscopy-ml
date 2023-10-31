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
import numpy as np

from abc import ABC, abstractmethod
from scipy.stats import ttest_ind


class Eval(ABC):
    def __init__(
        self, model, train_loader, valid_loader, eval_loader, input_size, output_size
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.eval_loader = eval_loader
        self.input_size = input_size
        self.output_size = output_size

        self.model.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Get mean, sd for model input and output
        mean_input = torch.tensor([0] * self.input_size).to(self.device).float()
        mean_output = torch.tensor([0] * self.output_size).to(self.device).float()

        std_input = torch.tensor([0] * self.input_size).to(self.device).float()
        std_output = torch.tensor([0] * self.output_size).to(self.device).float()

        for x, y in self.train_loader:
            # Move input and output tensors to the same device as mean_input and mean_output
            x, y = x.to(self.device), y.to(self.device)
            mean_input += x.mean([0])
            mean_output += y.mean([0])

        mean_input = mean_input / len(self.train_loader)
        mean_output = mean_output / len(self.train_loader)

        std_input = torch.sqrt(std_input / len(self.train_loader))
        std_output = torch.sqrt(std_output / len(self.train_loader))

        self.mean_input = mean_input.to(self.device).float().view(1, self.input_size)
        self.mean_output = mean_output.to(self.device).float().view(1, self.output_size)

        self.std_input = std_input.to(self.device).float()
        self.std_output = std_output.to(self.device).float()

    @abstractmethod
    def eval(self):
        pass

    @staticmethod
    def functional_mse(x, y):
        loss_fn = torch.nn.MSELoss(reduction="none")
        loss = loss_fn(x, y)
        # Move CUDA tensor to CPU before converting to NumPy
        loss_np = loss.cpu().detach().numpy()
        return np.sum(loss_np, axis=1)

    @staticmethod
    def loss_ttest(true_loss, other_loss, alpha=0.05):
        tstat, pval = ttest_ind(true_loss, other_loss, alternative="less")
        if pval < alpha:
            return True
        else:
            return False
