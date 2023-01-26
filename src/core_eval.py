###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import numpy as np
from pathlib import Path
import pickle
import tqdm as tqdm

import torch


from inout import load_xyz
from inout import load_xanes

from utils import unique_path
from utils import linecount
from utils import list_filestems
from structure.rdc import RDC
from structure.wacsf import WACSF


from scipy.stats import gaussian_kde
from scipy.stats import ttest_ind

from torch.utils.tensorboard import SummaryWriter
import time


# Tensorboard setup
# layout = {
#     "Multi": {
#         "loss": ["Multiline", ["loss/train", "loss/validation"]],
#     },
# }
# writer = SummaryWriter(f"/tmp/tensorboard/{int(time.time())}")
# writer.add_custom_scalars(layout)


def main(
    mode: str,
    model_dir: str,
    model_mode: str,
    x_train_path: str,
    y_train_path: str,
    x_test_path: str,
    y_test_path: str,
    test_params: dict = {},
    save_tensorboard: bool = True,
    seed: int = None,
):

    # ---------- load model ----------#
    model_dir = Path(model_dir)

    with open(model_dir / "descriptor.pickle", "rb") as f:
        descriptor = pickle.load(f)

    print(">> Loading model from disk...")
    model = torch.load(model_dir / "model.pt", map_location=torch.device("cpu"))
    model.eval()
    print(">> ...loaded!\n")

    # ---------- Load training data ----------#

    x_train_path = Path(x_train_path)
    y_train_path = Path(y_train_path)

    train_ids = list(
        set(list_filestems(x_train_path)) & set(list_filestems(y_train_path))
    )
    train_ids.sort()

    n_train_samples = len(train_ids)
    n_x_train_features = descriptor.get_len()
    n_y_train_features = linecount(y_train_path / f"{train_ids[0]}.txt") - 2

    x_train = np.full((n_train_samples, n_x_train_features), np.nan)
    y_train = np.full((n_train_samples, n_y_train_features), np.nan)

    print(">> loading training data into array(s)...")
    for i, id_ in enumerate(tqdm.tqdm(train_ids)):
        with open(x_train_path / f"{id_}.xyz", "r") as f:
            atoms = load_xyz(f)
        x_train[i, :] = descriptor.transform(atoms)
        with open(y_train_path / f"{id_}.txt", "r") as f:
            xanes = load_xanes(f)
            e, y_train[i, :] = xanes.spectrum
    print(">> ...loaded!\n")

    # ---------- Load test data ----------#

    x_test_path = Path(x_test_path)
    y_test_path = Path(y_test_path)

    test_ids = list(set(list_filestems(x_test_path)) & set(list_filestems(y_test_path)))
    test_ids.sort()

    n_test_samples = len(test_ids)
    n_x_test_features = descriptor.get_len()
    n_y_test_features = linecount(y_test_path / f"{test_ids[0]}.txt") - 2

    x_test = np.full((n_test_samples, n_x_test_features), np.nan)
    y_test = np.full((n_test_samples, n_y_test_features), np.nan)

    print(">> loading testing data into array(s)...")
    for i, id_ in enumerate(tqdm.tqdm(test_ids)):
        with open(x_test_path / f"{id_}.xyz", "r") as f:
            atoms = load_xyz(f)
        x_test[i, :] = descriptor.transform(atoms)
        with open(y_test_path / f"{id_}.txt", "r") as f:
            xanes = load_xanes(f)
            e, y_test[i, :] = xanes.spectrum
    print(">> ...loaded!\n")

    # ----------  convert data       ----------#

    xyz_train = torch.from_numpy(x_train).float()
    xanes_train = torch.from_numpy(y_train).float()

    xyz_test = torch.from_numpy(x_test).float()
    xanes_test = torch.from_numpy(y_test).float()

    # ---------- tensorboard ----------#

    # if save_tensorboard:
    # 	writer = SummaryWriter(log_dir = f"{model_dir}/tensorboard/tmp")

    ##################################################################

    modeltest = ModelTestInit(
        model, model_mode, mode, xyz_train, xanes_train, xyz_test, xanes_test
    )

    print("********** Running tests **********")
    print("\n")
    print(
        "[?] Output tests: Are true losses better than losses from artifical model output?"
    )
    print(
        "[?] Input tests:  Are true losses better than losses from model predictions from artifical input?"
    )
    print("[+] True if model passes check")
    print("[-] False if model fails check")
    print("\n")

    for (
        test_name,
        run_test,
    ) in test_params.items():
        if run_test:
            check = modeltest.run(test_name)
            print(f">> Test {test_name:20}: {check}")
            # if save_tensorboard:
            # writer.add_text(test_name, str(check))

    # if save_tensorboard:
    # 	writer.close()

    ##################################################################


class ModelTestInit:
    def __init__(self, model, model_mode, mode, x_train, y_train, x_test, y_test):
        self.model = model
        self.model_mode = model_mode
        self.mode = mode
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_train_samples = x_train.size(0)
        self.n_test_samples = x_test.size(0)
        self.n_x_features = x_test.size(1)
        self.n_y_features = y_test.size(1)

        # Init true model losses
        self.true_label, self.true_pred = self.predict(self.x_test, self.y_test)
        self.loss_fn = torch.nn.MSELoss(reduction="none")
        self.true_loss = (
            torch.sum(self.loss_fn(self.true_label, self.true_pred), dim=1)
            .detach()
            .numpy()
        )

    def run(self, test_name):

        # Create artifical source/targets from test set
        alt_xyz, alt_xanes = self.get_test_data(test_name)

        if "input" in test_name:

            alt_label, alt_pred = self.predict(alt_xyz, alt_xanes)

        if "output" in test_name:
            if "xyz" in self.mode:
                alt_label = self.true_label
                alt_pred = alt_xyz
            if "xanes" in self.mode:
                alt_label = self.true_label
                alt_pred = alt_xanes

        fake_loss = torch.sum(self.loss_fn(alt_label, alt_pred), dim=1).detach().numpy()

        check = perform_ttest(self.true_loss, fake_loss)

        return check

    def get_test_data(self, test_name):
        return getattr(
            self, f"test_function_{test_name.lower()}", lambda: (None, None)
        )()

    def test_function_shuffle_input(self):
        # input
        test_input = self.x_test[np.random.permutation(self.n_test_samples)]
        # output
        test_output = self.y_test
        return test_input, test_output

    def test_function_shuffle_output(self):
        # input
        test_input = self.x_test
        # output
        test_output = self.y_test[np.random.permutation(self.n_test_samples)]
        return test_input, test_output

    def test_function_mean_train_input(self):
        # input
        mu_x = np.mean(self.x_train.detach().numpy(), axis=0)
        test_input = torch.from_numpy(np.repeat([mu_x], self.n_test_samples, 0)).float()
        # output
        test_output = self.y_test
        return test_input, test_output

    def test_function_mean_train_output(self):
        # input
        test_input = self.x_test
        # output
        mu_y = np.mean(self.y_train.detach().numpy(), axis=0)
        test_output = torch.from_numpy(
            np.repeat([mu_y], self.n_test_samples, 0)
        ).float()
        return test_input, test_output

    def test_function_random_train_input(self):
        # input
        test_input = self.x_train[
            np.random.choice(
                np.arange(self.n_train_samples), self.n_test_samples, replace=False
            ),
            :,
        ].float()
        # output
        test_output = self.y_test
        return test_input, test_output

    def test_function_random_train_output(self):
        # input
        test_input = self.x_test
        # output
        test_output = self.y_train[
            np.random.choice(
                np.arange(self.n_train_samples), self.n_test_samples, replace=False
            ),
            :,
        ].float()
        return test_input, test_output

    def test_function_gauss_train_input(self):
        # input
        mu_x = np.mean(self.x_train.detach().numpy(), axis=0)
        sd_x = np.std(self.x_train.detach().numpy(), axis=0)
        test_input = np.transpose(
            np.array(
                [
                    [None for i in range(self.n_test_samples)]
                    for j in range(self.n_x_features)
                ]
            )
        )
        for i in range(self.n_test_samples):
            for j in range(self.n_x_features):
                test_input[i, j] = np.float64(mu_x[j] + np.random.normal(0, sd_x[j], 1))

        test_input = torch.from_numpy(test_input.astype(np.float64)).float()
        # output
        test_output = self.y_test
        return test_input, test_output

    def test_function_gauss_train_output(self):
        # input
        test_input = self.x_test
        # output
        mu_y = np.mean(self.y_train.detach().numpy(), axis=0)
        sd_y = np.std(self.y_train.detach().numpy(), axis=0)
        test_output = np.transpose(
            np.array(
                [
                    [None for i in range(self.n_test_samples)]
                    for j in range(self.n_y_features)
                ]
            )
        )
        for i in range(self.n_test_samples):
            for j in range(self.n_y_features):
                test_output[i, j] = np.float64(
                    mu_y[j] + np.random.normal(0, sd_y[j], 1)
                )
        test_output = torch.from_numpy(test_output.astype(np.float64)).float()
        return test_input, test_output

    def predict(self, xyz, xanes):

        if self.model_mode.lower() in ["mlp", "cnn"]:
            # ORGINAL MODEL
            if self.mode.lower() == "eval_pred_xanes":
                out = self.model(xyz)
            if self.mode == "eval_pred_xyz":
                out = self.model(xanes)

        if self.model_mode.lower() in ["ae_mlp", "ae_cnn"]:
            # AUTOENCODER MODEL
            if self.mode.lower() in ["eval_pred_xyz", "eval_recon_xanes"]:
                recon, pred = self.model(xanes)
            elif self.mode.lower() in ["eval_pred_xanes", "eval_recon_xyz"]:
                recon, pred = self.model(xyz)
            if "recon" in self.mode.lower():
                out = recon
            else:
                out = pred

        if self.model_mode.lower() in ["aegan_mlp", "aegan_cnn"]:
            # AUTOENCODER GAN MODEL
            if self.mode.lower() == "eval_pred_xanes":
                out = self.model.predict_spectrum(xyz)
            elif self.mode.lower() == "eval_pred_xyz":
                out = self.model.predict_structure(xanes)
            elif self.mode.lower() == "eval_recon_xyz":
                out = self.model.reconstruct_structure(xyz)
            elif self.mode.lower() == "eval_recon_xanes":
                out = self.model.reconstruct_spectrum(xanes)

        if "xyz" in self.mode.lower():
            compare = xyz
        elif "xanes" in self.mode.lower():
            compare = xanes

        return compare, out


def perform_ttest(true_loss, other_loss, alpha=0.05):
    """
    Performs a one-tailed two-sample T-Test at (default) 5% level.
    Tests whether the true distribution of errors is less than the alternative using scipy.stats.ttest_ind
    Returns True if true errors are less than alternative
    Returns False if true errors are not less than alternative


    Args:
            true_loss, other_loss : array_like
                    The arrays must have the same shape, except in the dimension
                    corresponding to `axis` (the first, by default).
            alpha (float, optional, default = 0.05) : p-value significance level

    """
    tstat, pval = ttest_ind(true_loss, other_loss, alternative="less")

    if pval < alpha:
        # Model is better than alternative
        # print(f"Model is better than alternative at {int(100*alpha):.0f}% level (pval = {pval:.3e})\n")
        return True
    else:
        # Model not better than alternative
        # print(f"Model is NOT better than alternative at {int(100*alpha):.0f}% level (pval = {pval:.3e})\n")
        return False
