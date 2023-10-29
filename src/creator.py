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

"""
Factory methods to create instance of model, descriptor or scheme based on
the specified name and parameters. To register a new class, add the label and class
name to the corresponding dictionary. The class name also need to be registered in 
the __init__.py file in the /scheme, /descriptor, or /model directory
"""


def create_model(name, **kwargs):
    from model import MLP, CNN, LSTM, AE_CNN, AE_MLP, AEGAN_MLP

    models = {
        "mlp": MLP,
        "cnn": CNN,
        "lstm": LSTM,
        "ae_mlp": AE_MLP,
        "ae_cnn": AE_CNN,
        "aegan_mlp": AEGAN_MLP,
    }

    if name in models:
        return models[name](**kwargs)
    else:
        raise ValueError(f"Unsupported module name: {name}")


def create_descriptor(name, **kwargs):
    from descriptor import RDC, WACSF, SOAP, MBTR, LMBTR, MSR, ARMSR

    descriptors = {
        "rdc": RDC,
        "wacsf": WACSF,
        "soap": SOAP,
        "mbtr": MBTR,
        "lmbtr": LMBTR,
        "msr": MSR,
        "armsr": ARMSR,
    }

    if name in descriptors:
        return descriptors[name](**kwargs)
    else:
        raise ValueError(f"Unsupported descriptor name: {name}")


def create_learn_scheme(
    x_data,
    y_data,
    model_params,
    hyperparams,
    kfold,
    kfold_params,
    bootstrap_params,
    ensemble_params,
    scheduler,
    schedular_param,
):
    from scheme import NNLearn, AELearn, AEGANLearn

    scheme = {
        "mlp": NNLearn,
        "cnn": NNLearn,
        "lstm": NNLearn,
        "ae_mlp": AELearn,
        "ae_cnn": AELearn,
        "aegan_mlp": AEGANLearn,
    }

    name = model_params["type"]

    if name in scheme:
        return scheme[name](
            x_data,
            y_data,
            model_params,
            hyperparams,
            kfold,
            kfold_params,
            bootstrap_params,
            ensemble_params,
            scheduler,
            schedular_param,
        )
    else:
        raise ValueError(f"Unsupported learn scheme name: {name}")


def create_eval_scheme(
    name, model, train_loader, valid_loader, eval_loader, input_size, output_size
):
    from scheme import NNEval, AEEval, AEGANEval

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
    name, xyz_data, xanes_data, pred_mode, index, pred_eval, fourier
):
    from scheme import NNPredict, AEPredict, AEGANPredict

    scheme = {
        "mlp": NNPredict,
        "cnn": NNPredict,
        "lstm": NNPredict,
        "ae_mlp": AEPredict,
        "ae_cnn": AEPredict,
        "aegan_mlp": AEGANPredict,
    }

    if name in scheme:
        return scheme[name](xyz_data, xanes_data, pred_mode, pred_eval, index, fourier)
    else:
        raise ValueError(f"Unsupported prediction scheme name: {name}")
