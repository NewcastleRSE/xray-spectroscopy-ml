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

import mlflow
import mlflow.pytorch
import torch
from torch_geometric.data import DataLoader

from torchinfo import summary
from sklearn.model_selection import train_test_split

from xanesnet.scheme.base_learn import Learn
from xanesnet.utils_model import (
    OptimSwitch,
    LossSwitch,
)


class GNNLearn(Learn):
    def __init__(self, x_data, y_data, **kwargs):
        # Call the constructor of the parent class
        super().__init__(x_data, y_data, **kwargs)

        # loss parameter set
        self.lr = self.hyper_params["lr"]
        self.optim_fn = self.hyper_params["optim_fn"]
        self.loss_fn = self.hyper_params["loss"]["loss_fn"]
        self.loss_args = self.hyper_params["loss"]["loss_args"]

        layout = {
            "Multi": {
                "loss": ["Multiline", ["loss/train", "loss/validation"]],
            },
        }
        self.writer = self.setup_writer(layout)

    def setup_dataloader(self, x_data, y_data):
        # split dataset and setup train/valid/test dataloader
        indices = list(range(len(x_data)))

        if self.model_eval:
            # Data split: train/valid/test
            train_ratio = 0.75
            test_ratio = 0.15
            eval_ratio = 0.10

            train_idx, test_idx = train_test_split(
                indices, test_size=1 - train_ratio, random_state=42
            )

            test_idx, eval_idx = train_test_split(
                test_idx, test_size=eval_ratio / (eval_ratio + test_ratio)
            )
        else:
            train_idx, test_idx = train_test_split(
                indices, test_size=0.2, random_state=42
            )

        train_dataset = x_data[train_idx]

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        test_dataset = x_data[test_idx]
        valid_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        if self.model_eval:
            eval_dataset = x_data[eval_idx]
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            eval_loader = None

        return [train_loader, valid_loader, eval_loader]

    def train(self, model, x_data, y_data):
        device = self.device
        writer = self.writer

        # initialise dataloader
        train_loader, valid_loader, eval_loader = self.setup_dataloader(x_data, y_data)

        # initialise optimizer
        optim_fn = OptimSwitch().fn(self.optim_fn)
        optimizer = optim_fn(model.parameters(), self.lr)
        criterion = LossSwitch().fn(self.loss_fn, self.loss_args)

        # initialise schedular
        if self.lr_scheduler:
            scheduler = self.setup_scheduler(optimizer)

        with mlflow.start_run(experiment_id=self.exp_id, run_name=self.exp_time):
            mlflow.log_params(self.hyper_params)
            mlflow.log_param("n_epoch", self.n_epoch)
            for epoch in range(self.n_epoch):
                print(f">>> epoch = {epoch}")
                model.train()
                running_loss = 0
                for idx, batch in enumerate(train_loader):
                    # Send data to device
                    batch.to(device)
                    # Reset gradients
                    optimizer.zero_grad()
                    # Passing the node features and the edge info
                    pred = model(
                        batch.x.float(),
                        batch.edge_attr.float(),
                        batch.edge_index,
                        batch.batch,
                    )
                    pred = torch.flatten(pred)
                    # Calculating the loss and gradients
                    loss = criterion(pred, batch.y.float())
                    loss.backward()
                    optimizer.step()
                    # Update tracking
                    running_loss += loss.item()

                valid_loss = 0
                model.eval()

                for batch in valid_loader:
                    batch.to(device)
                    pred = model(
                        batch.x.float(),
                        batch.edge_attr.float(),
                        batch.edge_index,
                        batch.batch,
                    )
                    pred = torch.flatten(pred)
                    loss = criterion(pred, batch.y.float())

                    # Update tracking
                    valid_loss += loss.item()

                if self.lr_scheduler:
                    before_lr = optimizer.param_groups[0]["lr"]
                    scheduler.step()
                    after_lr = optimizer.param_groups[0]["lr"]
                    print(
                        "Epoch %d: Adam lr %.5f -> %.5f" % (epoch, before_lr, after_lr)
                    )

                print("Training loss:", running_loss / len(train_loader))
                print("Validation loss:", valid_loss / len(valid_loader))

                self.log_scalar(
                    writer,
                    "loss/train",
                    (running_loss / len(train_loader)),
                    epoch,
                )
                self.log_scalar(
                    writer,
                    "loss/validation",
                    (valid_loss / len(valid_loader)),
                    epoch,
                )

            self.write_log(model)

        self.writer.close()
        score = running_loss / len(train_loader)

        return model, score

    def train_std(self):
        x_data = self.x_data
        y_data = None

        model = self.setup_model(x_data, y_data)
        model = self.setup_weight(model, self.weight_seed)
        model, _ = self.train(model, x_data, y_data)

        summary(model)

        return model

    def train_kfold(self, x_data=None, y_data=None):
        pass

    def train_bootstrap(self):
        pass

    def train_ensemble(self):
        pass
