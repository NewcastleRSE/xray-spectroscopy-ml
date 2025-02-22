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

from xanesnet.creator import create_model
from xanesnet.scheme.base_learn import Learn
from xanesnet.utils_model import (
    OptimSwitch,
    LossSwitch,
)

import time
from sklearn.model_selection import RepeatedKFold
import numpy as np
import random
from xanesnet.param_optuna import ParamOptuna
from pathlib import Path
import yaml
from xanesnet.param_freeze import Freeze


class GNNLearn(Learn):
    def __init__(self, x_data, y_data, **kwargs):
        # Call the constructor of the parent class
        super().__init__(x_data, y_data, **kwargs)

        # loss parameter set
        self.lr = self.hyper_params["lr"]
        self.optim_fn = self.hyper_params["optim_fn"]
        self.loss_fn = self.hyper_params["loss"]["loss_fn"]
        self.loss_args = self.hyper_params["loss"]["loss_args"]

        # Initialise tensorboard writer with custom layout
        if self.tb_flag:
            layout = {
                "Losses": {
                    "Losses": ["Multiline", ["loss/train", "loss/validation"]],
                },
            }
            self.writer = self.setup_writer(layout)

        # Initialise mlflow experiment
        if self.mlflow_flag:
            self.setup_mlflow()

    def setup_dataloader(self, x_data, y_data=None):
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

        return train_loader, valid_loader, eval_loader

    def setup_model(self, x_data, y_data=None):
        if self.freeze:
            # Load existing model from the specified path
            model_path = self.freeze_params["model_path"]
            metadata_path = Path(f"{model_path}/metadata.yaml")
            print(f"Loading model from {model_path}")

            with open(metadata_path, "r") as file:
                metadata = yaml.safe_load(file)
            model_name = metadata["model_type"]

            # Get model with frozen layers
            fz = Freeze(model_path)
            model = fz.get_fn(model_name, self.freeze_params)

        else:
            # Setup model with specified parameters
            self.model_params["x_data"] = x_data
            self.model_params["mlp_feat_size"] = x_data[0].graph_attr.shape[0]

            model = create_model(self.model_name, **self.model_params)

        model.to(self.device)

        return model

    def train(self, model, x_data, y_data=None):
        device = self.device

        # Initialise dataloaders
        train_loader, valid_loader, eval_loader = self.setup_dataloader(x_data)

        # Initialise optimizer
        optim_fn = OptimSwitch().fn(self.optim_fn)
        optimizer = optim_fn(model.parameters(), self.lr)

        # Initialise loss function
        criterion = LossSwitch().fn(self.loss_fn, self.loss_args)

        # initialise schedular
        if self.lr_scheduler:
            scheduler = self.setup_scheduler(optimizer)

        for epoch in range(self.n_epoch):
            # Training
            running_loss = 0

            print(f">>> epoch = {epoch}")
            model.train()

            for batch in train_loader:
                batch.to(device)

                optimizer.zero_grad()

                # reshape concatenated graph_attr to [batch_size, feat_size]
                nfeats = batch[0].graph_attr.shape[0]
                graph_attr = batch.graph_attr.reshape(len(batch), nfeats)

                pred = model(
                    batch.x.float(),
                    batch.edge_attr.float(),
                    graph_attr.float(),
                    batch.edge_index,
                    batch.batch,
                )
                pred = torch.flatten(pred)

                loss = criterion(pred, batch.y.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Validation
            valid_loss = 0
            model.eval()

            for batch in valid_loader:
                batch.to(device)
                nfeats = batch[0].graph_attr.shape[0]
                graph_attr = batch.graph_attr.reshape(len(batch), nfeats)

                pred = model(
                    batch.x.float(),
                    batch.edge_attr.float(),
                    graph_attr.float(),
                    batch.edge_index,
                    batch.batch,
                )

                pred = torch.flatten(pred)
                loss = criterion(pred, batch.y.float())
                valid_loss += loss.item()

            if self.lr_scheduler:
                before_lr = optimizer.param_groups[0]["lr"]
                scheduler.step()
                after_lr = optimizer.param_groups[0]["lr"]
                print("Epoch %d: Adam lr %.5f -> %.5f" % (epoch, before_lr, after_lr))

            train_loss = running_loss / len(train_loader)
            valid_loss = valid_loss / len(valid_loader)

            # Print losses to screen
            print("Training Loss:", train_loss)
            print("Validation Loss:", valid_loss)

            # Log losses
            self.log_loss("loss/train", train_loss, epoch)
            self.log_loss("loss/validation", valid_loss, epoch)

        if self.mlflow_flag:
            self.log_mlflow(model)

        self.log_close()

        score = running_loss / len(train_loader)

        return model, score

    def train_std(self):
        x_data = self.x_data

        # Optuna optimisation if enabled
        if self.optuna:
            self.proc_optuna(x_data, None, self.weight_seed)

        model = self.setup_model(x_data)
        model = self.setup_weight(model, self.weight_seed)
        model, _ = self.train(model, x_data)

        summary(model)

        return model

    def train_kfold(self):
        # K-fold Cross Validation model evaluation
        device = self.device
        x_data = self.x_data

        prev_score = 1e6
        fit_time = []
        train_score = []
        test_score = []

        kfold_spooler = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.seed_kfold,
        )

        criterion = LossSwitch().fn(self.loss_fn, self.loss_args)

        # Generate indices for k-fold splits
        indices = list(range(len(x_data)))

        for fold, (train_index, test_index) in enumerate(kfold_spooler.split(indices)):
            start = time.time()

            # Training
            train_data = x_data[train_index]
            model = self.setup_model(train_data)
            model = self.setup_weight(model, self.weight_seed)
            model, score = self.train(model, train_data)

            train_score.append(score)
            fit_time.append(time.time() - start)

            # Testing
            model.eval()
            test_data = x_data[test_index]
            test_loader = DataLoader(
                test_data,
                batch_size=self.batch_size,
                shuffle=False,
            )

            score = 0
            for batch in test_loader:
                batch.to(device)
                nfeats = batch[0].graph_attr.shape[0]
                graph_attr = batch.graph_attr.reshape(len(batch), nfeats)
                pred = model(
                    batch.x.float(),
                    batch.edge_attr.float(),
                    graph_attr.float(),
                    batch.edge_index,
                    batch.batch,
                )
                pred = torch.flatten(pred)
                loss = criterion(pred, batch.y.float())
                score += loss.item()

            test_score.append(score / len(test_loader))

            if score < prev_score:
                best_model = model
            prev_score = score

        result = {
            "fit_time": fit_time,
            "train_score": train_score,
            "test_score": test_score,
        }

        self._print_kfold_result(result)
        summary(best_model)

        return best_model

    def train_bootstrap(self):
        model_list = []
        x_data = self.x_data

        for i in range(self.n_boot):
            boot_indices = []
            # Set a unique seed for each bootstrap iteration
            weight_seed = self.weight_seed_boot[i]
            random.seed(weight_seed)

            # Generate bootstrap sample of size n_size x original dataset size
            for _ in range(int(len(x_data) * self.n_size)):
                idx = random.randint(0, len(x_data) - 1)
                boot_indices.append(idx)

            # Index the dataset with integer indices
            boot_x = x_data[boot_indices]

            # Optuna optimisation if enabled
            if self.optuna:
                self.proc_optuna(x_data, None, weight_seed)

            # Train the model on the bootstrap sample
            model = self.setup_model(boot_x)
            model = self.setup_weight(model, weight_seed)
            model, _ = self.train(model, boot_x)

            model_list.append(model)

        return model_list

    def train_ensemble(self):
        model_list = []
        x_data = self.x_data

        for i in range(self.n_ens):
            weight_seed = self.weight_seed_ens[i]
            if self.optuna:
                self.proc_optuna(x_data, None, weight_seed)

            # Train the model
            model = self.setup_model(x_data)
            model = self.setup_weight(model, weight_seed)
            model, _ = self.train(model, x_data)

            model_list.append(model)

        return model_list

    def _print_kfold_result(self, scores: dict):
        # prints a summary table of the scores from k-fold cross validation;
        # summarises the elapsed time and train/test metric scores for each k-fold
        # with overall k-fold cross validation statistics (mean and std. dev.)
        # using the `scores` dictionary returned from `cross_validate`

        print("")
        print(">> summarising scores from k-fold cross validation...")
        print("")

        print("*" * 16 * 3)
        fmt = "{:<10s}{:>6s}{:>16s}{:>16s}"
        print(fmt.format("k-fold", "time", "train", "test"))
        print("*" * 16 * 3)

        fmt = "{:<10.0f}{:>5.1f}s{:>16.8f}{:>16.8f}"
        for kf, (t, train, test) in enumerate(
            zip(scores["fit_time"], scores["train_score"], scores["test_score"])
        ):
            print(fmt.format(kf, t, np.absolute(train), np.absolute(test)))

        print("*" * 16 * 3)
        fmt = "{:<10s}{:>5.1f}s{:>16.8f}{:>16.8f}"
        means_ = (
            np.mean(np.absolute(scores[score]))
            for score in ("fit_time", "train_score", "test_score")
        )
        print(fmt.format("mean", *means_))
        stdevs_ = (
            np.std(np.absolute(scores[score]))
            for score in ("fit_time", "train_score", "test_score")
        )
        print(fmt.format("std. dev.", *stdevs_))
        print("*" * 16 * 3)
        print("")
