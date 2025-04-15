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
Factory functions to create instance of model, descriptor or scheme
"""


def create_model(name: str, **kwargs):
    """
    Returns an instance of a registered model class based on the given name.

    To register a new model, first import the model class (e.g, "MLP")
    and then an entry to the `models` dictionary with:
      - the model type as the key (e.g., "mlp")
      - the corresponding class as the value (e.g., MLP)
    """
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
        raise ValueError(f"Unsupported model name: {name}")


def create_descriptor(name: str, **kwargs):
    """
    Returns an instance of a registered descriptor class based on the given name.

    To register a new descriptor, first import the descriptor class (e.g, "WACSF")
    and then an entry to the `descriptorS` dictionary with:
      - the descriptor name as the key (e.g., "wacsf")
      - the corresponding class as the value (e.g., WACSF)
    """
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


def create_learn_scheme(
    name: str, x: np.ndarray = None, y: np.ndarray = None, **kwargs
):
    """
    Returns an instance of a learn scheme class based on the model type.

    To register a new scheme, first import the scheme class (e.g, "NNLearn")
    and then an entry to the `scheme` dictionary with:
      - the model type as the key (e.g., "mlp")
      - the corresponding class as the value (e.g., NNLearn)
    """
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

    if name in scheme:
        return scheme[name](x, y, **kwargs)
    else:
        raise ValueError(f"Unsupported learn scheme for the model: {name}")


def create_eval_scheme(
    name: str,
    model: Model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    eval_loader: DataLoader,
    input_size: int,
    output_size: int,
):
    """
    Returns an instance of an evaluation scheme class based on the model type.

    To register a new scheme, first import the scheme class (e.g, "NNEval")
    and then an entry to the `scheme` dictionary with:
      - the model type as the key (e.g., "mlp")
      - the corresponding class as the value (e.g., NNEval)
    """
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
        raise ValueError(f"Unsupported eval scheme for the model: {name}")


def create_predict_scheme(
    name: str, xyz: np.ndarray = None, xanes: np.ndarray = None, **kwargs
):
    """
    Returns an instance of a prediction scheme class based on the model type.

    To register a new scheme, first import the scheme class (e.g, "NNPredict")
    and then an entry to the `scheme` dictionary with:
      - the model type as the key (e.g., "mlp")
      - the corresponding class as the value (e.g., NNPredict)
    """
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
        return scheme[name](xyz, xanes, **kwargs)
    else:
        raise ValueError(f"Unsupported prediction scheme for the model: {name}")
