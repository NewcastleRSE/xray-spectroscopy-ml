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
import torch
import mlflow.pytorch
import numpy as np

from sklearn.model_selection import RepeatedKFold
from torchinfo import summary

from xanesnet.optuna import ParamOptuna
from xanesnet.scheme.base_learn import Learn
from xanesnet.creator import create_eval_scheme
from xanesnet.utils_model import OptimSwitch, LossSwitch


class AELearn(Learn):
    def __init__(self, x_data, y_data, **kwargs):
        # Call the constructor of the parent class
        super().__init__(x_data, y_data, **kwargs)

        # hyperparameter set
        self.lr = self.hyper_params["lr"]
        self.optim_fn = self.hyper_params["optim_fn"]
        self.loss_fn = self.hyper_params["loss"]["loss_fn"]
        self.loss_args = self.hyper_params["loss"]["loss_args"]

        layout = {
            "Multi": {
                "recon_loss": [
                    "Multiline",
                    ["recon_loss/train", "recon_loss/validation"],
                ],
                "pred_loss": ["Multiline", ["pred_loss/train", "pred_loss/validation"]],
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

        # initialise dataloader
        train_loader, valid_loader, eval_loader = self.setup_dataloader(x_data, y_data)

        # Initialise optimizer using specified optimization function and learning rate
        optim_fn = OptimSwitch().fn(self.optim_fn)
        optimizer = optim_fn(model.parameters(), lr=self.lr)
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
                loss_r = 0
                loss_p = 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs, labels = inputs.float(), labels.float()

                    optimizer.zero_grad()

                    recon_input, outputs = model(inputs)

                    loss_recon = criterion(recon_input, inputs)
                    loss_pred = criterion(outputs, labels)

                    loss = loss_recon + loss_pred
                    loss.backward()

                    optimizer.step()
                    running_loss += loss.mean().item()
                    loss_r += loss_recon.item()
                    loss_p += loss_pred.item()

                valid_loss = 0
                valid_loss_r = 0
                valid_loss_p = 0

                model.eval()

                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs, labels = inputs.float(), labels.float()

                    recon_input, outputs = model(inputs)

                    loss_recon = criterion(recon_input, inputs)
                    loss_pred = criterion(outputs, labels)

                    loss = loss_recon + loss_pred

                    valid_loss = loss.item()
                    valid_loss_r += loss_recon.item()
                    valid_loss_p += loss_pred.item()

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
                    "total_loss/train",
                    (running_loss / len(train_loader)),
                    epoch,
                )
                self.log_scalar(
                    writer,
                    "total_loss/validation",
                    (valid_loss / len(valid_loader)),
                    epoch,
                )

                self.log_scalar(
                    writer, "recon_loss/train", (loss_r / len(train_loader)), epoch
                )
                self.log_scalar(
                    writer,
                    "recon_loss/validation",
                    (valid_loss_r / len(train_loader)),
                    epoch,
                )

                self.log_scalar(
                    writer, "pred_loss/train", (loss_p / len(train_loader)), epoch
                )
                self.log_scalar(
                    writer,
                    "pred_loss/validation",
                    (valid_loss_p / len(valid_loader)),
                    epoch,
                )

            self.write_log(model)

            # Perform model evaluation using invariance tests
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
        weight_seed = self.weight_seed

        if self.optuna:
            self.proc_optuna(x_data, y_data, weight_seed)
        model = self.setup_model(x_data, y_data)
        model = self.setup_weight(model, weight_seed)
        model, _ = self.train(model, x_data, y_data)

        summary(model, (1, x_data.shape[1]))

        return model

    def train_kfold(self, x_data=None, y_data=None):
        # K-fold Cross Validation model evaluation
        device = self.device

        if x_data is None:
            x_data = self.x_data
        if y_data is None:
            y_data = self.y_data

        prev_score = 1e6
        fit_time = []
        train_score = []
        test_recon_score = []
        test_pred_score = []

        kfold_spooler = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.seed_kfold,
        )

        criterion = LossSwitch().fn(self.loss_fn, self.loss_args)

        for fold, (train_index, test_index) in enumerate(kfold_spooler.split(x_data)):
            start = time.time()

            model = self.setup_model(x_data[train_index], y_data[train_index])
            model = self.setup_weight(model, self.weight_seed)
            model, score = self.train(model, x_data[train_index], y_data[train_index])

            train_score.append(score)
            fit_time.append(time.time() - start)
            # Testing
            model.eval()
            x_test = torch.from_numpy(x_data[test_index]).float().to(device)
            y_test = torch.from_numpy(y_data[test_index]).float().to(device)
            recon_x, pred_y = model(x_test)
            recon_score = criterion(x_test, recon_x).item()
            pred_score = criterion(y_test, pred_y).item()
            test_recon_score.append(recon_score)
            test_pred_score.append(pred_score)
            mean_score = np.mean([recon_score, pred_score])

            if mean_score < prev_score:
                best_model = model
            prev_score = mean_score

        result = {
            "fit_time": fit_time,
            "train_score": train_score,
            "test_recon_score": test_recon_score,
            "test_pred_score": test_pred_score,
        }

        self._print_kfold_result(result)

        return best_model

    def train_bootstrap(self):
        model_list = []
        x_data = self.x_data
        y_data = self.y_data

        for i in range(self.n_boot):
            weight_seed = self.weight_seed_boot[i]
            random.seed(weight_seed)

            boot_x = []
            boot_y = []

            for _ in range(int(x_data.shape[0] * self.n_size)):
                idx = random.randint(0, x_data.shape[0] - 1)
                boot_x.append(x_data[idx])
                boot_y.append(y_data[idx])

            boot_x = np.asarray(boot_x)
            boot_y = np.asarray(boot_y)

            if self.optuna:
                self.proc_optuna(x_data, y_data, weight_seed)

            model = self.setup_model(boot_x, boot_y)
            model = self.setup_weight(model, weight_seed)
            model, _ = self.train(model, boot_x, boot_y)

            model_list.append(model)

        return model_list

    def train_ensemble(self):
        model_list = []
        x_data = self.x_data
        y_data = self.y_data

        for i in range(self.n_ens):
            weight_seed = self.weight_seed_ens[i]

            if self.optuna:
                self.proc_optuna(x_data, y_data, weight_seed)
            model = self.setup_model(x_data, y_data)
            model = self.setup_weight(model, weight_seed)
            model, _ = self.train(model, x_data, y_data)

            model_list.append(model)

        return model_list

    def _print_kfold_result(self, scores: dict):
        # prints a summary table of the scores from k-fold cross validation;
        # summarises the elapsed time and train/test metric scores for each k-fold
        # with overall k-fold cross validation statistics (mean and std. dev.)
        # using the `scores` dictionary returned from `cross_validate`
        print("*" * 16 * 4)
        fmt = "{:<10s}{:>6s}{:>16s}{:>16s}{:>16s}"
        print(fmt.format("k-fold", "time", "train", "test recon", "test pred"))
        print("*" * 16 * 4)

        fmt = "{:<10.0f}{:>5.1f}s{:>16.8f}{:>16.8f}{:>16.8f}"
        for kf, (t, train, test_recon, test_pred) in enumerate(
            zip(
                scores["fit_time"],
                scores["train_score"],
                scores["test_recon_score"],
                scores["test_pred_score"],
            )
        ):
            print(
                fmt.format(
                    kf,
                    t,
                    np.absolute(train),
                    np.absolute(test_recon),
                    np.absolute(test_pred),
                )
            )

        print("*" * 16 * 4)

        fmt = "{:<10s}{:>5.1f}s{:>16.8f}{:>16.8f}{:>16.8f}"
        means_ = (
            np.mean(np.absolute(scores[score]))
            for score in (
                "fit_time",
                "train_score",
                "test_recon_score",
                "test_pred_score",
            )
        )
        print(fmt.format("mean", *means_))
        stdevs_ = (
            np.std(np.absolute(scores[score]))
            for score in (
                "fit_time",
                "train_score",
                "test_recon_score",
                "test_pred_score",
            )
        )
        print(fmt.format("std. dev.", *stdevs_))
        print("*" * 16 * 4)
        print("")
