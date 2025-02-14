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

from sklearn.model_selection import RepeatedKFold
from torchinfo import summary

from xanesnet.scheme.base_learn import Learn
from xanesnet.utils_model import LossSwitch, loss_reg_fn
from xanesnet.creator import create_eval_scheme


class AEGANLearn(Learn):
    def __init__(self, x_data, y_data, **kwargs):
        # Call the constructor of the parent class
        super().__init__(x_data, y_data, **kwargs)

        # Regularisation of gen loss function
        self.loss_fn = self.model_params["loss_gen"]["loss_fn"]
        self.loss_args = self.model_params["loss_gen"]["loss_args"]
        self.loss_gen_reg_type = self.model_params["loss_gen"]["loss_reg_type"]
        self.lambda_gen_reg = self.model_params["loss_gen"]["loss_reg_param"]

        # Regularisation of dis loss function
        self.loss_dis_reg_type = self.model_params["loss_dis"]["loss_reg_type"]
        self.lambda_dis_reg = self.model_params["loss_dis"]["loss_reg_param"]

        # Initialise tensorboard writer with custom layout
        if self.tb_flag:
            layout = {
                "Generator & Discriminator Losses": {
                    "Generator Loss": ("Multiline", ["gen_loss/train"]),
                    "Discriminator Loss": ("Multiline", ["dis_loss/train"]),
                },
                "Reconstruction Losses": {
                    "Reconstruction X": (
                        "Multiline",
                        ["recon_X_loss/train", "recon_X_loss/valid"],
                    ),
                    "Reconstruction Y": (
                        "Multiline",
                        ["recon_Y_loss/train", "recon_Y_loss/valid"],
                    ),
                },
                "Prediction Losses": {
                    "Prediction X": (
                        "Multiline",
                        ["pred_X_loss/train", "pred_X_loss/valid"],
                    ),
                    "Prediction Y": (
                        "Multiline",
                        ["pred_Y_loss/train", "pred_Y_loss/valid"],
                    ),
                },
            }
            self.writer = self.setup_writer(layout)

        # Initialise mlflow experiment
        if self.mlflow_flag:
            self.setup_mlflow()

    def train(self, model, x_data, y_data):
        device = self.device

        train_total_loss = [None] * self.n_epoch
        train_loss_x_recon = [None] * self.n_epoch
        train_loss_y_recon = [None] * self.n_epoch
        train_loss_x_pred = [None] * self.n_epoch
        train_loss_y_pred = [None] * self.n_epoch

        # Initialise dataloaders
        train_loader, valid_loader, eval_loader = self.setup_dataloader(x_data, y_data)
        if self.model_eval and len(eval_loader) < 2:
            raise ValueError("eval_loader expects more than 1 data")

        # Initialise optimizer
        gen_opt, dis_opt = model.get_optimizer()

        # Initialise loss function
        criterion = LossSwitch().fn(self.loss_fn, self.loss_args)

        # Initialise lr_schedular
        if self.lr_scheduler:
            scheduler_gen, scheduler_dis = self.setup_scheduler(gen_opt, dis_opt)

        # Apply standardscaler to training dataset
        if self.scaler:
            x_data = self.setup_scaler(x_data)
            y_data = self.setup_scaler(y_data)

        for epoch in range(self.n_epoch):
            # Training
            running_loss_recon_x = 0
            running_loss_recon_y = 0
            running_loss_pred_x = 0
            running_loss_pred_y = 0
            running_gen_loss = 0
            running_dis_loss = 0

            print(f">>> epoch = {epoch}")
            model.train()

            for inputs_x, inputs_y in train_loader:
                inputs_x, inputs_y = inputs_x.to(device), inputs_y.to(device)
                inputs_x, inputs_y = inputs_x.float(), inputs_y.float()

                model.gen_update(inputs_x, inputs_y)
                model.dis_update(inputs_x, inputs_y, device)

                (
                    recon_x,
                    recon_y,
                    pred_x,
                    pred_y,
                ) = model.reconstruct_all_predict_all(inputs_x, inputs_y)

                # Track running losses
                running_loss_recon_x += criterion(recon_x, inputs_x)
                running_loss_recon_y += criterion(recon_y, inputs_y)
                running_loss_pred_x += criterion(pred_x, inputs_x)
                running_loss_pred_y += criterion(pred_y, inputs_y)

                loss_gen_total = (
                    running_loss_recon_x
                    + running_loss_recon_y
                    + running_loss_pred_x
                    + running_loss_pred_y
                )
                loss_dis_total = model.loss_dis_total

                # Regularisation of gen loss
                if self.loss_gen_reg_type is not None:
                    gen_a_l_reg = loss_reg_fn(
                        model.gen_a, self.loss_gen_reg_type, device
                    )
                    gen_b_l_reg = loss_reg_fn(
                        model.gen_b, self.loss_gen_reg_type, device
                    )
                    loss_gen_total += self.lambda_gen_reg * (gen_a_l_reg + gen_b_l_reg)

                # Regularisation of dis loss
                if self.loss_dis_reg_type is not None:
                    dis_a_l_reg = loss_reg_fn(
                        model.dis_a, self.loss_dis_reg_type, device
                    )
                    dis_b_l_reg = loss_reg_fn(
                        model.dis_b, self.loss_dis_reg_type, device
                    )
                    loss_dis_total += self.lambda_dis_reg * (dis_a_l_reg + dis_b_l_reg)

                running_gen_loss += loss_gen_total.item()
                running_dis_loss += loss_dis_total.item()

            if self.lr_scheduler:
                before_lr_gen = gen_opt.param_groups[0]["lr"]
                scheduler_gen.step()
                after_lr_gen = gen_opt.param_groups[0]["lr"]
                print(
                    "Epoch %d: Adam lr %.5f -> %.5f"
                    % (epoch, before_lr_gen, after_lr_gen)
                )

                before_lr_dis = dis_opt.param_groups[0]["lr"]
                scheduler_dis.step()
                after_lr_dis = dis_opt.param_groups[0]["lr"]
                print(
                    "Epoch %d: Adam lr %.5f -> %.5f"
                    % (epoch, before_lr_dis, after_lr_dis)
                )

            running_gen_loss = running_gen_loss / len(train_loader)
            running_dis_loss = running_dis_loss / len(train_loader)

            running_loss_recon_x = running_loss_recon_x.item() / len(train_loader)
            running_loss_recon_y = running_loss_recon_y.item() / len(train_loader)
            running_loss_pred_x = running_loss_pred_x.item() / len(train_loader)
            running_loss_pred_y = running_loss_pred_y.item() / len(train_loader)

            # Log training losses
            self.log_loss("gen_loss/train", running_gen_loss, epoch)
            self.log_loss("dis_loss/train", running_dis_loss, epoch)
            self.log_loss("recon_X_loss/train", running_loss_recon_x, epoch)
            self.log_loss("recon_Y_loss/train", running_loss_recon_y, epoch)
            self.log_loss("pred_X_loss/train", running_loss_pred_x, epoch)
            self.log_loss("pred_Y_loss/train", running_loss_pred_y, epoch)

            train_loss_x_recon[epoch] = running_loss_recon_x
            train_loss_y_recon[epoch] = running_loss_recon_y
            train_loss_x_pred[epoch] = running_loss_pred_x
            train_loss_y_pred[epoch] = running_loss_pred_y

            train_total_loss[epoch] = running_gen_loss

            # Validation
            valid_loss_recon_x = 0
            valid_loss_recon_y = 0
            valid_loss_pred_x = 0
            valid_loss_pred_y = 0
            valid_loss_total = 0

            model.eval()

            for inputs_x, inputs_y in valid_loader:
                inputs_x, inputs_y = inputs_x.to(device), inputs_y.to(device)
                inputs_x, inputs_y = inputs_x.float(), inputs_y.float()

                (
                    recon_x,
                    recon_y,
                    pred_x,
                    pred_y,
                ) = model.reconstruct_all_predict_all(inputs_x, inputs_y)

                valid_loss_recon_x += criterion(recon_x, inputs_x)
                valid_loss_recon_y += criterion(recon_y, inputs_y)
                valid_loss_pred_x += criterion(pred_x, inputs_x)
                valid_loss_pred_y += criterion(pred_y, inputs_y)

                valid_loss = (
                    valid_loss_recon_x
                    + valid_loss_recon_y
                    + valid_loss_pred_x
                    + valid_loss_pred_y
                )

                valid_loss_total += valid_loss.item()

            valid_loss_recon_x = valid_loss_recon_x.item() / len(valid_loader)
            valid_loss_recon_y = valid_loss_recon_y.item() / len(valid_loader)
            valid_loss_pred_x = valid_loss_pred_x.item() / len(valid_loader)
            valid_loss_pred_y = valid_loss_pred_y.item() / len(valid_loader)

            # Log validation losses
            self.log_loss("recon_X_loss/valid", valid_loss_recon_x, epoch)
            self.log_loss("recon_Y_loss/valid", valid_loss_recon_y, epoch)
            self.log_loss("pred_X_loss/valid", valid_loss_pred_x, epoch)
            self.log_loss("pred_Y_loss/valid", valid_loss_pred_y, epoch)

            # Print losses to screen
            print(f">>> Training Loss (recon X)   = {running_loss_recon_x:.4f}")
            print(f">>> Training Loss (recon Y)   = {running_loss_recon_y:.4f}")
            print(f">>> Training Loss (pred X)    = {running_loss_pred_x:.4f}")
            print(f">>> Training Loss (pred Y)    = {running_loss_pred_y:.4f}")

            print(f">>> Validation Loss (recon X) = {valid_loss_recon_x:.4f}")
            print(f">>> Validation Loss (recon Y) = {valid_loss_recon_y:.4f}")
            print(f">>> Validation Loss (pred X)  = {valid_loss_pred_x:.4f}")
            print(f">>> Validation Loss (pred Y)  = {valid_loss_pred_y:.4f}")

            losses = {
                "train_loss": train_total_loss,
                "loss_x_recon": train_loss_x_recon,
                "loss_y_recon": train_loss_y_recon,
                "loss_x_pred": train_loss_x_pred,
                "loss_y_pred": train_loss_y_pred,
            }

        # Evaluation using invariance tests
        if self.model_eval:
            eval_test = create_eval_scheme(
                self.model_name,
                model,
                train_loader,
                valid_loader,
                eval_loader,
                x_data.shape[1],
                y_data.shape[1],
            )

            eval_results = eval_test.eval()

            # Log evaluation results
            if self.mlflow_flag:
                for k, v in eval_results.items():
                    mlflow.log_dict(v, f"{k}.yaml")
                print(f"{'='*19} MLFlow: Evaluation Results Logged {'='*18}")

        self.log_close()

        score = losses

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

        summary(model)

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
        test_recon_xyz_score = []
        test_recon_xanes_score = []
        test_pred_xyz_score = []
        test_pred_xanes_score = []

        kfold_spooler = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.seed_kfold,
        )

        criterion = LossSwitch().fn(self.loss_fn, self.loss_args)

        for fold, (train_index, test_index) in enumerate(kfold_spooler.split(x_data)):
            start = time.time()
            # Training
            model = self.setup_model(x_data[train_index], y_data[train_index])
            model = self.setup_weight(model, self.weight_seed)
            model, score = self.train(model, x_data[train_index], y_data[train_index])

            train_score.append(score["train_loss"][-1])
            fit_time.append(time.time() - start)
            # Testing
            model.eval()
            xyz_test = torch.from_numpy(x_data[test_index]).float().to(device)
            xanes_test = torch.from_numpy(y_data[test_index]).float().to(device)
            (
                recon_xyz,
                recon_xanes,
                pred_xyz,
                pred_xanes,
            ) = model.reconstruct_all_predict_all(xyz_test, xanes_test)

            recon_xyz_score = criterion(xyz_test, recon_xyz).item()
            recon_xanes_score = criterion(xanes_test, recon_xanes).item()
            pred_xyz_score = criterion(xyz_test, pred_xyz).item()
            pred_xanes_score = criterion(xanes_test, pred_xanes).item()

            test_recon_xyz_score.append(recon_xyz_score)
            test_recon_xanes_score.append(recon_xanes_score)
            test_pred_xyz_score.append(pred_xyz_score)
            test_pred_xanes_score.append(pred_xanes_score)

            mean_score = np.mean(
                [
                    recon_xyz_score,
                    recon_xanes_score,
                    pred_xyz_score,
                    pred_xanes_score,
                ]
            )
            if mean_score < prev_score:
                best_model = model
            prev_score = mean_score

        result = {
            "fit_time": fit_time,
            "train_score": train_score,
            "test_recon_xyz_score": test_recon_xyz_score,
            "test_recon_xanes_score": test_recon_xanes_score,
            "test_pred_xyz_score": test_pred_xyz_score,
            "test_pred_xanes_score": test_pred_xanes_score,
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
        print("*" * (16 + 18 * 5))
        fmt = "{:<10s}{:>6s}{:>18s}{:>18s}{:>18s}{:>18s}{:>18s}"
        print(
            fmt.format(
                "k-fold",
                "time",
                "train",
                "test recon xyz",
                "test recon xanes",
                "test pred xyz",
                "test pred xanes",
            )
        )
        print("*" * (16 + 18 * 5))

        fmt = "{:<10.0f}{:>5.1f}s{:>18.8f}{:>18.8f}{:>18.8f}{:>18.8f}{:>18.8f}"
        for kf, (t, train, recon_xyz, recon_xanes, pred_xyz, pred_xanes) in enumerate(
            zip(
                scores["fit_time"],
                scores["train_score"],
                scores["test_recon_xyz_score"],
                scores["test_recon_xanes_score"],
                scores["test_pred_xyz_score"],
                scores["test_pred_xanes_score"],
            )
        ):
            print(
                fmt.format(
                    kf,
                    t,
                    np.absolute(train),
                    np.absolute(recon_xyz),
                    np.absolute(recon_xanes),
                    np.absolute(pred_xyz),
                    np.absolute(pred_xanes),
                )
            )

        print("*" * (16 + 18 * 5))

        fmt = "{:<10s}{:>5.1f}s{:>18.8f}{:>18.8f}{:>18.8f}{:>18.8f}{:>18.8f}"
        means_ = (
            np.mean(np.absolute(scores[score]))
            for score in (
                "fit_time",
                "train_score",
                "test_recon_xyz_score",
                "test_recon_xanes_score",
                "test_pred_xyz_score",
                "test_pred_xanes_score",
            )
        )
        print(fmt.format("mean", *means_))
        stdevs_ = (
            np.std(np.absolute(scores[score]))
            for score in (
                "fit_time",
                "train_score",
                "test_recon_xyz_score",
                "test_recon_xanes_score",
                "test_pred_xyz_score",
                "test_pred_xanes_score",
            )
        )
        print(fmt.format("std. dev.", *stdevs_))
        print("*" * (16 + 18 * 5))
        print("")
