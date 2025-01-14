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

import random
import mlflow
import time
import mlflow.pytorch
import numpy as np
import torch

from numpy.random import RandomState
from torchinfo import summary
from sklearn.model_selection import RepeatedKFold

from xanesnet.scheme.base_learn import Learn
from xanesnet.creator import create_eval_scheme
from xanesnet.utils_model import (
    loss_reg_fn,
    OptimSwitch,
    LossSwitch,
)


class NNLearn(Learn):
    def __init__(self, x_data, y_data, **kwargs):
        super().__init__(x_data, y_data, **kwargs)

        # loss parameter set
        self.lr = self.hyper_params["lr"]
        self.optim_fn = self.hyper_params["optim_fn"]
        self.loss_fn = self.hyper_params["loss"]["loss_fn"]
        self.loss_args = self.hyper_params["loss"]["loss_args"]
        self.loss_reg_type = self.hyper_params["loss"]["loss_reg_type"]
        self.lambda_reg = self.hyper_params["loss"]["loss_reg_param"]

        layout = {
            "Multi": {
                "loss": ["Multiline", ["loss/train", "loss/validation"]],
            },
        }
        self.writer = self.setup_writer(layout)

    def train(self, model, x_data, y_data):
        device = self.device
        writer = self.writer

        # Apply standardscaler to training dataset
        if self.scaler:
            print(">> Applying standardscaler to training dataset...")
            x_data = self.setup_scaler(x_data)

        # ########################TEST#############################
        # import matplotlib.pyplot as plt
        # print("test code at nn_learn.py")
        # plt.title('FFT + Standardscaler Xanes')
        # plt.plot(x_data[0])
        # plt.show()
        # ########################TEST#############################

        # initialise dataloaders
        train_loader, valid_loader, eval_loader = self.setup_dataloader(x_data, y_data)
        # initialise optimizer
        optim_fn = OptimSwitch().fn(self.optim_fn)
        optimizer = optim_fn(model.parameters(), self.lr)
        # initialise loss criterion
        criterion = LossSwitch().fn(self.loss_fn, self.loss_args)
        # initialise schedular
        if self.lr_scheduler:
            scheduler = self.setup_scheduler(optimizer)

        with mlflow.start_run(experiment_id=self.exp_id, run_name=self.exp_time):
            mlflow.log_params(self.hyper_params)
            mlflow.log_param("n_epoch", self.n_epoch)
            for epoch in range(self.n_epoch):
                print(f">>> epoch = {epoch}")
                # Train the model
                model.train()
                running_loss = 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs, labels = inputs.float(), labels.float()
                    optimizer.zero_grad()
                    # Pass input data through model to get predictions
                    pred = model(inputs)
                    # calculate loss
                    loss = criterion(pred, labels)
                    # Apply loss regularisation if enabled
                    if self.loss_reg_type is not None:
                        l_reg = loss_reg_fn(model, self.loss_reg_type, device)
                        loss += self.lambda_reg * l_reg
                    # Backpropagation
                    loss.mean().backward()
                    # Update model parameters
                    optimizer.step()
                    # Update tracking
                    running_loss += loss.item()

                # Model validation
                valid_loss = 0
                model.eval()
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs, labels = inputs.float(), labels.float()

                    pred = model(inputs)
                    loss = criterion(pred, labels)
                    valid_loss += loss.item()

                # Adjust learning rate if enabled
                if self.lr_scheduler:
                    before_lr = optimizer.param_groups[0]["lr"]
                    scheduler.step()
                    after_lr = optimizer.param_groups[0]["lr"]
                    print(
                        "Epoch %d: Adam lr %.5f -> %.5f" % (epoch, before_lr, after_lr)
                    )

                train_loss = running_loss / len(train_loader)
                valid_loss = valid_loss / len(valid_loader)
                # Print to screen
                print("Training loss:", train_loss)
                print("Validation loss:", valid_loss)
                # Save to log
                self.log_scalar(writer, "loss/train", train_loss, epoch)
                self.log_scalar(writer, "loss/validation", valid_loss, epoch)

            self.write_log(model)

            # Model evaluation using invariance tests if enabled
            if self.model_eval:
                eval_test = create_eval_scheme(
                    self.model_name,
                    model,
                    train_loader,
                    valid_loader,
                    eval_loader,
                    x_data.shape[1],
                    y_data[0].size,
                )
                eval_results = eval_test.eval()
                # Log results
                for k, v in eval_results.items():
                    mlflow.log_dict(v, f"{k}.yaml")

        self.writer.close()
        score = running_loss / len(train_loader)

        return model, score

    def train_std(self):
        x_data = self.x_data
        y_data = self.y_data

        # Optuna optimisation if enabled
        if self.optuna:
            self.proc_optuna(x_data, y_data, self.weight_seed)

        # Train the model
        model = self.setup_model(x_data, y_data)
        model = self.setup_weight(model, self.weight_seed)
        model, _ = self.train(model, x_data, y_data)

        # Display the model summary
        summary(model, (1, x_data.shape[1]))

        return model

    def train_kfold(self):
        # Perform K-fold Repeated Cross Validation
        device = self.device
        x_data = self.x_data
        y_data = self.y_data

        prev_score = 1e6
        fit_time = []
        train_score = []
        test_score = []

        # Initialise K-fold splitter
        kfold_spooler = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.seed_kfold,
        )
        # Initialise loss criterion
        criterion = LossSwitch().fn(self.loss_fn, self.loss_args)

        for fold, (train_index, test_index) in enumerate(kfold_spooler.split(x_data)):
            start = time.time()
            # Train the model
            model = self.setup_model(x_data[train_index], y_data[train_index])
            model = self.setup_weight(model, self.weight_seed)
            model, score = self.train(model, x_data[train_index], y_data[train_index])

            train_score.append(score)
            fit_time.append(time.time() - start)
            # Model evaluation on the test dataset
            model.eval()
            x_test = torch.from_numpy(x_data[test_index]).float().to(device)
            pred = model(x_test)
            score = criterion(torch.tensor(y_data[test_index]).to(device), pred).item()
            test_score.append(score)

            if score < prev_score:
                best_model = model
            prev_score = score

        result = {
            "fit_time": fit_time,
            "train_score": train_score,
            "test_score": test_score,
        }
        # Print results to screen
        self._print_kfold_result(result)
        summary(best_model, (1, x_data.shape[1]))

        return best_model

    def train_bootstrap(self):
        # Train multiple models using bootstrap resampling
        model_list = []
        x_data = self.x_data
        y_data = self.y_data

        for i in range(self.n_boot):
            boot_x = []
            boot_y = []
            # Set a unique seed for each bootstrap iteration
            weight_seed = self.weight_seed_boot[i]
            random.seed(weight_seed)

            # Generate bootstrap sample of size n_size x original dataset size
            for _ in range(int(x_data.shape[0] * self.n_size)):
                idx = random.randint(0, x_data.shape[0] - 1)
                boot_x.append(x_data[idx])
                boot_y.append(y_data[idx])

            boot_x = np.asarray(boot_x)
            boot_y = np.asarray(boot_y)

            # Optuna optimisation if enabled
            if self.optuna:
                self.proc_optuna(x_data, y_data, weight_seed)

            # Train the model on the bootstrap sample
            model = self.setup_model(boot_x, boot_y)
            model = self.setup_weight(model, weight_seed)
            model, _ = self.train(model, boot_x, boot_y)

            model_list.append(model)

        return model_list

    def train_ensemble(self):
        # Train multiple models using ensemble learning
        model_list = []
        x_data = self.x_data
        y_data = self.y_data

        for i in range(self.n_ens):
            if self.optuna:
                self.proc_optuna(x_data, y_data, self.weight_seed_ens[i])
            # Train the model
            model = self.setup_model(x_data, y_data)
            model = self.setup_weight(model, self.weight_seed_ens[i])
            model, _ = self.train(model, x_data, y_data)

            model_list.append(model)

        return model_list

    def _print_kfold_result(self, scores: dict):
        # prints a summary table of the scores from k-fold cross validation;
        # summarises the elapsed time and train/test metric scores for each k-fold
        # with overall k-fold cross validation statistics (mean and std. dev.)
        # using the `scores` dictionary returned from `cross_validate`

        print("")
        print(">> Summarising scores from k-fold cross validation...")
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
