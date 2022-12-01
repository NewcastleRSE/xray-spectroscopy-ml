

# if args.mode == 'test':
# need
# - model_dir
# - training input, training output
# - test input, test output

# - generate the artifical output and input
# - do model inference on test data
# - ttests for model output
# - ttests for model input
# - write results to tensorboard




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
    model_dir: str,
    x_train_path: str,
    y_train_path: str,
    x_test_path: str,
    y_test_path: str,
    model_type: str,
    test_params: dict = {},
    save_tensorboard: bool = True,
    seed: int = None
    ):


    #---------- load model ----------#
    model_dir = Path(model_dir)

    with open(model_dir / "descriptor.pickle", "rb") as f:
        descriptor = pickle.load(f)

    print('>> Loading model from disk...')
    model = torch.load(model_dir / "model.pt", map_location=torch.device("cpu"))
    model.eval()
    print(">> ...loaded!\n")
    
    #---------- Load training data ----------#

    x_train_path = Path(x_train_path)
    y_train_path = Path(y_train_path)

    train_ids = list(set(list_filestems(x_train_path)) & set(list_filestems(y_train_path)))
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

    #---------- Load test data ----------#

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


    #----------  convert data       ----------#

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()

    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()


    ##################################################################
    
    test_inout = TestInputOutput(model, x_train, y_train, x_test, y_test)

    for k, v in test_params.items():

        test_input, test_output = test_inout.get_input_output(k, v)

        print(f">> Test name: {k}, run? {v}, input? {test_input is not None} output? {test_output is not None}")

    ##################################################################




class TestInputOutput:
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_train_samples = x_train.size(0)
        self.n_test_samples = x_test.size(0)
        self.n_x_features = x_test.size(1)
        self.n_y_features = y_test.size(1)


    def get_input_output(self, test_name, run_test):
        if run_test is True:
            return getattr(self, f"test_function_{test_name.lower()}", lambda: (None, None))()
        else:
            return None, None


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
        mu_x = np.mean(self.x_train.detach().numpy(),axis = 0)
        test_input = torch.from_numpy(np.repeat([mu_x],self.n_test_samples,0)).float()
        # output
        test_output = self.y_test
        return test_input, test_output

    def test_function_mean_train_output(self):
        # input
        test_input = self.x_test
        # output
        mu_y = np.mean(self.y_train.detach().numpy(),axis = 0)
        test_output = torch.from_numpy(np.repeat([mu_y],self.n_test_samples,0)).float()
        return test_input, test_output

    def test_function_random_train_input(self):
        # input
        test_input = self.x_train[np.random.choice(np.arange(self.n_train_samples),self.n_test_samples,replace = False),:].float()
        # output
        test_output = self.y_test
        return test_input, test_output

    def test_function_random_train_output(self):
        # input
        test_input = self.x_test
        # output
        test_output = self.y_train[np.random.choice(np.arange(self.n_train_samples),self.n_test_samples,replace = False),:].float()
        return test_input, test_output

    def test_function_gauss_train_input(self):
        # input
        mu_x = np.mean(self.x_train.detach().numpy(),axis = 0)
        sd_x = np.std(self.x_train.detach().numpy(),axis = 0)
        test_input = np.transpose(np.array([[None for i in range(self.n_test_samples)] for j in range(self.n_x_features)]))
        for i in range(self.n_test_samples):
            for j in range(self.n_x_features):
                test_input[i,j] = np.float64(mu_x[j] + np.random.normal(0,sd_x[j],1))

        test_input = torch.from_numpy(test_input.astype(np.float64)).float()
        # output
        test_output = self.y_test
        return test_input, test_output

    def test_function_gauss_train_output(self):
        # input
        test_input = self.x_test
        # output
        mu_y = np.mean(self.y_train.detach().numpy(),axis = 0)
        sd_y = np.std(self.y_train.detach().numpy(),axis = 0)
        test_output = np.transpose(np.array([[None for i in range(self.n_test_samples)] for j in range(self.n_y_features)]))
        for i in range(self.n_test_samples):
            for j in range(self.n_y_features):
                test_output[i,j] = np.float64(mu_y[j] + np.random.normal(0,sd_y[j],1))
        test_output = torch.from_numpy(test_output.astype(np.float64)).float()
        return test_input, test_output


###############################################################################
###############################################################################
###############################################################################


###############################################################################
################################### FUNCTIONS #################################
###############################################################################



def get_density(data):
    density = gaussian_kde(data)
    xs = np.linspace(0,max(data),200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    return xs,density(xs)

def ttest_true_loss_less_than_other_loss(true_loss, other_loss,alpha = 0.05, plot_density = False):
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
        plot_density (bool, optional, default = False) : Plot normalised density of the two distributions 

    """
    tstat, pval = ttest_ind(true_loss, other_loss,alternative = 'less')

    if plot_density:
        xs_true, ds_true = get_density(true_loss)
        xs_other, ds_other = get_density(other_loss)
        ds_true_norm = ds_true/sum(ds_true)
        ds_other_norm = ds_other/sum(ds_other)
        plt.plot(xs_true, ds_true_norm, label = 'True loss')
        plt.plot(xs_other, ds_other_norm, label = 'Other loss')
        plt.legend(loc = 'upper right')
        plt.show()
        
    if pval < alpha:
        # Model is better than alternative
        print(f'Model is better than alternative at {int(100*alpha):.0f}% level (pval = {pval:.3e})\n')
        return True
    else:
        # Model not better than alternative
        print(f"Model is NOT better than alternative at {int(100*alpha):.0f}% level (pval = {pval:.3e})\n")
        return False
