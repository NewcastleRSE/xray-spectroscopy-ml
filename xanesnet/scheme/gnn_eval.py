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

from xanesnet.scheme.base_eval import Eval


class GNNEval():
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

        for batch in self.train_loader:
            # Move input and output tensors to the same device as mean_input and mean_output
            batch.to(self.device)
            nfeats = batch[0].graph_attr.shape[0]
            graph_attr = batch.graph_attr.reshape(len(batch), nfeats)
            y = batch.y.float().to(self.device)
            
            mean_input += graph_attr.mean([0])
            mean_output += y.mean([0])

        mean_input = mean_input / len(self.train_loader)
        mean_output = mean_output / len(self.train_loader)

        std_input = torch.sqrt(std_input / len(self.train_loader))
        std_output = torch.sqrt(std_output / len(self.train_loader))

        self.mean_input = mean_input.to(self.device).float().view(1, nfeats)
        self.mean_output = mean_output.to(self.device).float().view(1, self.output_size)

        self.std_input = std_input.to(self.device).float()
        self.std_output = std_output.to(self.device).float()

    @staticmethod
    def functional_mse(x, y):
        loss_fn = torch.nn.MSELoss(reduction="none")
        loss = loss_fn(x, y)
        # Move CUDA tensor to CPU before converting to NumPy
        loss_np = loss.cpu().detach().numpy()
        return np.sum(loss_np, axis=0)

    def eval(self):
        print(f"{'='*20} Running Model Evaluation Tests {'='*20}")

        test_results = {}
        l0 = self.get_true_loss()

        li1 = self.get_loss_input_shuffle()
        lo1 = self.get_loss_output_shuffle()

        li2 = self.get_loss_input_mean_train()
        lo2 = self.get_loss_output_mean_train()

        li3 = self.get_loss_input_mean_sd_train()
        lo3 = self.get_loss_output_mean_sd_train()

        li4 = self.get_loss_input_random_valid()
        lo4 = self.get_loss_output_random_valid()

        test_results["Shuffle Input"] = Eval.loss_ttest(l0, li1)
        test_results["Shuffle Output"] = Eval.loss_ttest(l0, lo1)

        test_results["Mean Train Input"] = Eval.loss_ttest(l0, li2)
        test_results["Mean Train Output"] = Eval.loss_ttest(l0, lo2)

        test_results["Mean Std. Train Input"] = Eval.loss_ttest(l0, li3)
        test_results["Mean Std. Train Output"] = Eval.loss_ttest(l0, lo3)

        test_results["Random Valid Input"] = Eval.loss_ttest(l0, li4)
        test_results["Random Valid Output"] = Eval.loss_ttest(l0, lo4)

        for k, v in test_results.items():
            print(f">>> {k:25}: {v}")

        test_results = {"ModelEvalResults-Prediction": test_results}

        return test_results

    def get_true_loss(self):
        true_loss = []
        device = self.device

        for batch in self.eval_loader:
            batch.to(device)
            nfeats = batch[0].graph_attr.shape[0]
            graph_attr = batch.graph_attr.reshape(len(batch), nfeats)

            target = self.model(
                batch.x.float(),
                batch.edge_attr.float(),
                graph_attr.float(),
                batch.edge_index,
                batch.batch,
            )
            target = torch.flatten(target)
            loss = self.functional_mse(target, batch.y.float())
            true_loss.extend([loss])

        return true_loss

    def get_loss_input_shuffle(self):
        other_loss = []
        device = self.device

        for batch in self.eval_loader:
            batch.to(device)
            nfeats = batch[0].graph_attr.shape[0]
            graph_attr = batch.graph_attr.reshape(len(batch), nfeats)

            # Shuffle graph attributes
            idx = torch.randperm(graph_attr.shape[0])
            graph_attr = graph_attr[idx]

            target = self.model(
                batch.x.float(),
                batch.edge_attr.float(),
                graph_attr.float(),
                batch.edge_index,
                batch.batch,
            )
            target = torch.flatten(target)
            loss = self.functional_mse(target, batch.y.float())
            other_loss.extend([loss])

        return other_loss

    def get_loss_output_shuffle(self):
        other_loss = []
        device = self.device

        for batch in self.eval_loader:
            batch.to(device)
            nfeats = batch[0].graph_attr.shape[0]
            graph_attr = batch.graph_attr.reshape(len(batch), nfeats)

            # Shuffle target values
            idx = torch.randperm(batch.y.shape[0])
            shuffled_y = batch.y[idx]

            target = self.model(
                batch.x.float(),
                batch.edge_attr.float(),
                graph_attr.float(),
                batch.edge_index,
                batch.batch,
            )
            target = torch.flatten(target)
            loss = self.functional_mse(target, shuffled_y.float())
            other_loss.extend([loss])

        return other_loss

    def get_loss_input_mean_train(self):
        other_loss = []
        device = self.device

        for batch in self.eval_loader:
            batch.to(device)
            nfeats = batch[0].graph_attr.shape[0]
            
            # Create a copy of the batch to modify
            mean_batch = batch.clone()
            
            # Replace graph attributes with mean values and reshape to match original format
            mean_graph_attr = self.mean_input.repeat(len(batch), 1)
            mean_graph_attr = mean_graph_attr.reshape(len(batch), nfeats)
            mean_batch.graph_attr = mean_graph_attr
            
            # Get predictions using mean graph attributes
            target = self.model(
                mean_batch.x.float(),
                mean_batch.edge_attr.float(),
                mean_batch.graph_attr.float(),
                mean_batch.edge_index,
                mean_batch.batch,
            )
            target = torch.flatten(target)
            
            # Compute loss with original labels
            loss = self.functional_mse(target, batch.y.float())
            other_loss.extend([loss])

        return other_loss

    def get_loss_output_mean_train(self):
        other_loss = []
        device = self.device

        for batch in self.eval_loader:
            batch.to(device)
            nfeats = batch[0].graph_attr.shape[0]
            graph_attr = batch.graph_attr.reshape(len(batch), nfeats)

            # Get predictions using original graph attributes
            target = self.model(
                batch.x.float(),
                batch.edge_attr.float(),
                graph_attr.float(),
                batch.edge_index,
                batch.batch,
            )
            target = torch.flatten(target)
            
            # Replace target with mean output, using target shape for repetition
            mean_target = self.mean_output.repeat(target.shape[0] // self.output_size, 1)
            mean_target = torch.flatten(mean_target)
            
            # Compute loss with mean target
            loss = self.functional_mse(mean_target, batch.y.float())
            other_loss.extend([loss])

        return other_loss
    
    def get_loss_input_mean_sd_train(self):
        other_loss = []
        device = self.device

        for batch in self.eval_loader:
            batch.to(device)
            nfeats = batch[0].graph_attr.shape[0]
            
            # Create a copy of the batch to modify
            mean_sd_batch = batch.clone()
            
            # Add noise to mean graph attributes and reshape to match original format
            mean_sd_graph_attr = self.mean_input.repeat(len(batch), 1) + torch.normal(
                torch.zeros([len(batch), nfeats], device=device),
                self.std_input
            )
            mean_sd_graph_attr = mean_sd_graph_attr.reshape(len(batch), nfeats)
            mean_sd_batch.graph_attr = mean_sd_graph_attr
            
            # Get predictions using noisy mean graph attributes
            target = self.model(
                mean_sd_batch.x.float(),
                mean_sd_batch.edge_attr.float(),
                mean_sd_batch.graph_attr.float(),
                mean_sd_batch.edge_index,
                mean_sd_batch.batch,
            )
            target = torch.flatten(target)
            
            # Compute loss with original labels
            loss = self.functional_mse(target, batch.y.float())
            other_loss.extend([loss])

        return other_loss

    def get_loss_output_mean_sd_train(self):
        other_loss = []
        device = self.device

        for batch in self.eval_loader:
            batch.to(device)
            nfeats = batch[0].graph_attr.shape[0]
            graph_attr = batch.graph_attr.reshape(len(batch), nfeats)

            target = self.model(
                batch.x.float(),
                batch.edge_attr.float(),
                graph_attr.float(),
                batch.edge_index,
                batch.batch,
            )
            target = torch.flatten(target)
            
            # Add noise to mean output and flatten to match target shape
            mean_sd_output = self.mean_output.repeat(target.shape[0] // self.output_size, 1) + torch.normal(
                torch.zeros([target.shape[0] // self.output_size, self.output_size], device=device),
                self.std_output,
            )
            mean_sd_output = torch.flatten(mean_sd_output)
            
            loss = self.functional_mse(mean_sd_output, batch.y.float())
            other_loss.extend([loss])

        return other_loss

    def get_loss_input_random_valid(self):
        other_loss = []
        device = self.device
        it = iter(self.valid_loader)

        for batch in self.eval_loader:
            batch.to(device)
            nfeats = batch[0].graph_attr.shape[0]
            
            # Get random batch from validation set
            alt_batch = next(it)
            alt_batch.to(device)
            alt_graph_attr = alt_batch.graph_attr.reshape(len(alt_batch), nfeats)
            
            if len(batch) < len(alt_batch):
                alt_graph_attr = alt_graph_attr[:len(batch)]

            target = self.model(
                batch.x.float(),
                batch.edge_attr.float(),
                alt_graph_attr.float(),
                batch.edge_index,
                batch.batch,
            )
            target = torch.flatten(target)
            loss = self.functional_mse(target, batch.y.float())
            other_loss.extend([loss])

        return other_loss

    def get_loss_output_random_valid(self):
        other_loss = []
        device = self.device
        it = iter(self.valid_loader)

        for batch in self.eval_loader:
            batch.to(device)
            nfeats = batch[0].graph_attr.shape[0]
            graph_attr = batch.graph_attr.reshape(len(batch), nfeats)

            target = self.model(
                batch.x.float(),
                batch.edge_attr.float(),
                graph_attr.float(),
                batch.edge_index,
                batch.batch,
            )
            target = torch.flatten(target)
            
            # Get random labels from validation set
            alt_batch = next(it)
            alt_batch.to(device)
            alt_y = alt_batch.y
            
            if target.shape[0] < alt_y.shape[0]:
                alt_y = alt_y[:target.shape[0]]
                
            loss = self.functional_mse(target, alt_y.float())
            other_loss.extend([loss])

        return other_loss
