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

import optuna


class ParamOptuna:
    def __init__(self, trial, model_params, hyper_params):
        self.trial = trial
        self.model_params = model_params
        self.hyper_params = hyper_params

    def get_fn(self, name):
        return getattr(self, name)()

    def tune_optim_fn(self):
        options = ["Adam", "SGD", "RMSprop"]
        self.hyper_params["optim_fn"] = self.trial.suggest_categorical(
            "optim_fn", options
        )

    def tune_batch_size(self):
        options = [8, 16, 32, 64]

        self.hyper_params["batch_size"] = self.trial.suggest_categorical(
            "batch_size", options
        )

    def tune_activation(self):
        options = [
            "relu",
            "prelu",
            "tanh",
            "sigmoid",
            "elu",
            "leakyrelu",
            "selu",
        ]
        self.hyper_params["activation"] = self.trial.suggest_categorical(
            "activation", options
        )

    def tune_dropout(self):
        dropout_min = 0.2
        dropout_max = 0.5
        self.hyper_params["dropout"] = self.trial.suggest_uniform(
            "dropout", dropout_min, dropout_max
        )

    def tune_loss_fn(self):
        options = ["mse", "emd", "cosine", "l1", "wcc"]
        wcc_min = 5
        wcc_max = 15

        fn = self.trial.suggest_categorical("loss_fn", options)
        if fn == "wcc":
            self.hyper_params["loss"][
                "loss_args"
            ] = self.trial.suggest_discrete_uniform("loss_args", wcc_min, wcc_max, q=1)

        self.hyper_params["loss"]["loss_fn"] = fn

    def tune_lr(self):
        lr_min = 1e-7
        lr_max = 1e-3
        self.hyper_params["lr"] = self.trial.suggest_float(
            "lr", lr_min, lr_max, log=True
        )

    def tune_aegan_mlp(self):
        lr_min = 1e-7
        lr_max = 1e-3
        n_hl_min = 2
        n_hl_max = 5

        self.model_params["lr_gen"] = self.trial.suggest_float(
            "lr_gen", lr_min, lr_max, log=True
        )
        self.model_params["lr_dis"] = self.trial.suggest_float(
            "lr_dis", lr_min, lr_max, log=True
        )

        self.model_params["n_hl_gen"] = self.trial.suggest_int(
            "n_hl_gen", n_hl_min, n_hl_max
        )
        self.model_params["n_hl_shared"] = self.trial.suggest_int(
            "n_hl_shared", n_hl_min, n_hl_max
        )
        self.model_params["n_hl_dis"] = self.trial.suggest_int(
            "n_hl_dis", n_hl_min, n_hl_max
        )

    def tune_mlp(self):
        n_hl_min = 2
        n_hl_max = 5
        hidden_size = [64, 128, 256, 512]
        shrink_min = 0.2
        shrink_max = 0.5

        n_hl = self.trial.suggest_int("num_hidden_layers", n_hl_min, n_hl_max)
        hl_size = self.trial.suggest_categorical("hidden_size", hidden_size)
        hl_shrink = self.trial.suggest_uniform("shrink_rate", shrink_min, shrink_max)

        last_hl_size = int(hl_size * hl_shrink ** (n_hl - 1))

        if last_hl_size < 1:
            raise optuna.exceptions.TrialPruned(
                "The size of the last hidden layer was less than 1. Terminated Optuna trial for this parameterisation."
            )
        else:
            self.model_params["num_hidden_layers"] = n_hl
            self.model_params["hidden_size"] = hl_size
            self.model_params["shrink_rate"] = hl_shrink

    def tune_cnn(self):
        n_cl_min = 1
        n_cl_max = 5
        hidden_size = [64, 128, 256, 512]

        self.model_params["num_conv_layers"] = self.trial.suggest_int(
            "num_conv_layers", n_cl_min, n_cl_max
        )
        self.model_params["hidden_size"] = self.trial.suggest_categorical(
            "hidden_size", hidden_size
        )

    def tune_lstm(self):
        n_hl_min = 2
        n_hl_max = 5
        hidden_size = [64, 128, 256, 512]

        self.model_params["num_hidden_layers"] = self.trial.suggest_int(
            "num_hidden_layers", n_hl_min, n_hl_max
        )
        self.model_params["hidden_size"] = self.trial.suggest_categorical(
            "hidden_size", hidden_size
        )
