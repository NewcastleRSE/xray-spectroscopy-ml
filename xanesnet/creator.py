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

import numpy as np

from xanesnet.model.base_model import Model
from torch.utils.data import DataLoader

"""
Factory methods to create instance of model, descriptor or scheme based on
the specified name and parameters. To register a new class, add the label and class
name to the corresponding dictionary. The class name also need to be registered in 
the __init__.py file in the /scheme, /descriptor, or /model directory
"""


def create_model(name: str, **kwargs):
    from xanesnet.model import MLP, CNN, LSTM, AE_CNN, AE_MLP, AEGAN_MLP, GNN

    models = {
        "mlp": MLP,
        "cnn": CNN,
        "lstm": LSTM,
        "gnn": GNN,
        "ae_mlp": AE_MLP,
        "ae_cnn": AE_CNN,
        "aegan_mlp": AEGAN_MLP,
    }

    if name in models:
        return models[name](**kwargs)
    else:
        raise ValueError(f"Unsupported module name: {name}")


def create_descriptor(name: str, **kwargs):
    from xanesnet.descriptor import (
        RDC,
        WACSF,
        SOAP,
        MBTR,
        LMBTR,
        MSR,
        ARMSR,
        PDOS,
        XTB,
        DIRECT,
        MACE,
    )

    descriptors = {
        "rdc": RDC,
        "wacsf": WACSF,
        "soap": SOAP,
        "mbtr": MBTR,
        "lmbtr": LMBTR,
        "msr": MSR,
        "armsr": ARMSR,
        "pdos": PDOS,
        "xtb": XTB,
        "direct": DIRECT,
        "mace": MACE,
    }

    if name in descriptors:
        return descriptors[name](**kwargs)
    else:
        raise ValueError(f"Unsupported descriptor name: {name}")


def create_learn_scheme(x_data: np.ndarray, y_data: np.ndarray, **kwargs):
    from xanesnet.scheme import NNLearn, AELearn, AEGANLearn, GNNLearn

    scheme = {
        "mlp": NNLearn,
        "cnn": NNLearn,
        "lstm": NNLearn,
        "ae_mlp": AELearn,
        "ae_cnn": AELearn,
        "aegan_mlp": AEGANLearn,
        "gnn": GNNLearn,
    }

    model_params = kwargs.get("model")
    name = model_params["type"]

    if name in scheme:
        return scheme[name](x_data, y_data, **kwargs)
    else:
        raise ValueError(f"Unsupported learn scheme name: {name}")


def create_eval_scheme(
    name: str,
    model: Model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    eval_loader: DataLoader,
    input_size: int,
    output_size: int,
):
    from xanesnet.scheme import NNEval, AEEval, AEGANEval

    scheme = {
        "mlp": NNEval,
        "cnn": NNEval,
        "lstm": NNEval,
        "ae_mlp": AEEval,
        "ae_cnn": AEEval,
        "aegan_mlp": AEGANEval,
    }

    if name in scheme:
        return scheme[name](
            model,
            train_loader,
            valid_loader,
            eval_loader,
            input_size,
            output_size,
        )
    else:
        raise ValueError(f"Unsupported eval scheme name: {name}")


def create_predict_scheme(
    name: str,
    xyz_data: np.ndarray,
    xanes_data: np.ndarray,
    pred_mode: str,
    pred_eval: bool,
    scaler: bool,
    fourier: bool,
    fourier_param: dict,
):
    from xanesnet.scheme import NNPredict, AEPredict, AEGANPredict, GNNPredict

    scheme = {
        "mlp": NNPredict,
        "cnn": NNPredict,
        "lstm": NNPredict,
        "ae_mlp": AEPredict,
        "ae_cnn": AEPredict,
        "aegan_mlp": AEGANPredict,
        "gnn": GNNPredict,
    }

    if name in scheme:
        return scheme[name](
            xyz_data,
            xanes_data,
            pred_mode,
            pred_eval,
            scaler,
            fourier,
            fourier_param,
        )
    else:
        raise ValueError(f"Unsupported prediction scheme name: {name}")
