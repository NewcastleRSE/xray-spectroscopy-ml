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
import copy
from pathlib import Path

import mlflow
import optuna
import torch
import time
import os
import pickle
import tempfile
import random

from abc import ABC, abstractmethod
from datetime import datetime

import yaml
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from xanesnet.creator import create_model
from xanesnet.freeze import Freeze
from xanesnet.utils_model import LRScheduler, WeightInitSwitch, weight_bias_init
from xanesnet.optuna import ParamOptuna


class Learn(ABC):
    """Base class for model training procedures"""

    def __init__(
        self,
        x_data,
        y_data,
        model_params,
        hyper_params,
        kfold,
        kfold_params,
        bootstrap_params,
        ensemble_params,
        scheduler,
        scheduler_params,
        optuna,
        optuna_params,
        freeze,
        freeze_params,
    ):
        self.x_data = x_data
        self.y_data = y_data
        self.hyper_params = hyper_params
        self.kfold = kfold
        self.lr_scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.model_params = model_params["params"]
        self.optuna = optuna
        self.optuna_params = optuna_params
        self.freeze = freeze
        self.freeze_params = freeze_params

        # kfold parameter set
        self.n_splits = kfold_params["n_splits"]
        self.n_repeats = kfold_params["n_repeats"]
        self.seed_kfold = kfold_params["seed"]

        # hyperparameter set
        self.batch_size = hyper_params["batch_size"]
        self.n_epoch = hyper_params["epochs"]
        self.kernel = hyper_params["kernel_init"]
        self.bias = hyper_params["bias_init"]
        self.model_eval = hyper_params["model_eval"]
        self.weight_seed = hyper_params["weight_seed"]
        self.seed = hyper_params["seed"]

        # bootstrap parameter set
        self.n_boot = bootstrap_params["n_boot"]
        self.weight_seed_boot = bootstrap_params["weight_seed"]
        self.n_size = bootstrap_params["n_size"]

        # ensemble parameter set
        self.n_ens = ensemble_params["n_ens"]
        self.weight_seed_ens = ensemble_params["weight_seed"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # mlflow experiment info
        self.model_name = model_params["type"]
        exp_name = f"{self.model_name}"
        self.exp_time = f"run_{datetime.today()}"
        try:
            self.exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
        except:
            self.exp_id = mlflow.create_experiment(exp_name)

        self.recon_flag = 0

    @abstractmethod
    def train(self, model, x_data, y_data):
        pass

    @abstractmethod
    def train_std(self):
        pass

    @abstractmethod
    def train_kfold(self, x_data=None, y_data=None):
        pass

    @abstractmethod
    def train_bootstrap(self):
        pass

    @abstractmethod
    def train_ensemble(self):
        pass

    def train_optuna(self, trial, x_data, y_data, seed):
        po = ParamOptuna(trial, self.model_params, self.hyper_params)

        for name, flag in self.optuna_params.items():
            if name.startswith("tune_") and flag:
                po.get_fn(name)

        model = self.setup_model(x_data, y_data)
        model = self.setup_weight(model, seed)
        _, score = self.train(model, x_data, y_data)

        if len(score) > 1:
            score = sum([v[-1] for k, v in score.items()])

        return score

    def setup_writer(self, layout):
        # setup tensorboard stuff
        writer = SummaryWriter(f"/tmp/tensorboard/{int(time.time())}")
        writer.add_custom_scalars(layout)

        return writer

    def setup_model(self, x_data, y_data):
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
            self.model_params["y_data"] = y_data

            model = create_model(self.model_name, **self.model_params)

        model.to(self.device)

        return model

    def setup_scheduler(self, optimizer):
        scheduler_type = self.scheduler_params["type"]
        params = {
            key: value for key, value in self.scheduler_params.items() if key != "type"
        }
        scheduler = LRScheduler(optimizer, scheduler_type=scheduler_type, params=params)

        return scheduler

    def setup_dataloader(self, x_data, y_data):
        # split dataset and setup train/valid/test dataloader
        x_data = torch.from_numpy(x_data)
        y_data = torch.from_numpy(y_data)

        if self.model_eval:
            # Data split: train/valid/test
            train_ratio = 0.75
            test_ratio = 0.15
            eval_ratio = 0.10

            x_train, x_test, y_train, y_test = train_test_split(
                x_data, y_data, test_size=1 - train_ratio, random_state=42
            )

            x_test, x_eval, y_test, y_eval = train_test_split(
                x_test, y_test, test_size=eval_ratio / (eval_ratio + test_ratio)
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x_data, y_data, test_size=0.2, random_state=42
            )

        train_set = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
        )

        valid_set = torch.utils.data.TensorDataset(x_test, y_test)
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=self.batch_size,
            shuffle=False,
        )

        if self.model_eval:
            eval_set = torch.utils.data.TensorDataset(x_eval, y_eval)
            eval_loader = torch.utils.data.DataLoader(
                eval_set,
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            eval_loader = None

        return [train_loader, valid_loader, eval_loader]

    def setup_weight(self, model, weight_seed):
        # Initialise model weight & bias
        kernel_init = WeightInitSwitch().fn(self.kernel)
        bias_init = WeightInitSwitch().fn(self.bias)
        # set seed
        torch.cuda.manual_seed(
            weight_seed
        ) if torch.cuda.is_available() else torch.manual_seed(weight_seed)

        model.apply(
            lambda m: weight_bias_init(
                m=m, kernel_init_fn=kernel_init, bias_init_fn=bias_init
            )
        )
        return model

    def log_scalar(self, writer, name, value, epoch):
        """Log a scalar value to both MLflow and TensorBoard"""
        writer.add_scalar(name, value, epoch)
        mlflow.log_metric(name, value)

    def write_log(self, model):
        # # Create a SummaryWriter to write TensorBoard events locally
        output_dir = tempfile.mkdtemp()

        # Upload the TensorBoard event logs as a run artifact
        print("Uploading TensorBoard events as a run artifact...")
        mlflow.log_artifacts(output_dir, artifact_path="events")
        print(
            "\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s"
            % os.path.join(mlflow.get_artifact_uri(), "events")
        )

        # Log the model as an artifact of the MLflow run.
        print("\nLogging the trained model as a run artifact...")
        mlflow.pytorch.log_model(
            model, artifact_path="pytorch-model", pickle_module=pickle
        )
        print(
            "\nThe model is logged at:\n%s"
            % os.path.join(mlflow.get_artifact_uri(), "pytorch-model")
        )

        mlflow.pytorch.load_model(mlflow.get_artifact_uri("pytorch-model"))

    def proc_optuna(self, x_data, y_data, seed):
        n_trials = self.optuna_params["n_trials"]

        func = lambda trial: self.train_optuna(trial, x_data, y_data, seed)

        study = optuna.create_study(direction="minimize")
        study.optimize(func, n_trials=n_trials, timeout=None)
        self.print_optuna(study)

    def print_optuna(self, study):
        # Print optuna study statistics
        print(f"{'='*20} Optuna {'='*20}")
        print("Study statistics: ")
        print(f"  Number of finished trials: {len(study.trials)}")

        print("Best trial:")
        trial = study.best_trial

        print(f"  Value: {trial.value}")

        print("  Params: ")
        for k, v in trial.params.items():
            print(f"    {k}: {v}")
        print(f"{'='*20} Optuna {'='*20}")
