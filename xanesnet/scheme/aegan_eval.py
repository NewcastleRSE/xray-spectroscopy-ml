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

import mlflow
import torch

from xanesnet.scheme.base_eval import Eval


class AEGANEval(Eval):
    def eval(self):
        rxl0, ryl0, pxl0, pyl0 = self.get_true_loss()

        rxli1, ryli1, pxli1, pyli1 = self.get_loss_input_shuffle()
        rxlo1, rylo1, pxlo1, pylo1 = self.get_loss_output_shuffle()

        rxli2, ryli2, pxli2, pyli2 = self.get_loss_input_mean_train()
        rxlo2, rylo2, pxlo2, pylo2 = self.get_loss_output_mean_train()

        rxli3, ryli3, pxli3, pyli3 = self.get_loss_input_mean_sd_train()
        rxlo3, rylo3, pxlo3, pylo3 = self.get_loss_output_mean_sd_train()

        rxli4, ryli4, pxli4, pyli4 = self.get_loss_input_random_valid()
        rxlo4, rylo4, pxlo4, pylo4 = self.get_loss_output_random_valid()

        p_x_results = {}
        p_y_results = {}
        r_x_results = {}
        r_y_results = {}

        # Prediction XYZ
        p_x_results["Shuffle Input"] = Eval.loss_ttest(pxl0, pxli1)
        p_x_results["Shuffle Output"] = Eval.loss_ttest(pxl0, pxlo1)

        p_x_results["Mean Train Input"] = Eval.loss_ttest(pxl0, pxli2)
        p_x_results["Mean Train Output"] = Eval.loss_ttest(pxl0, pxlo2)

        p_x_results["Mean Std. Train Input"] = Eval.loss_ttest(pxl0, pxli3)
        p_x_results["Mean Std. Train Output"] = Eval.loss_ttest(pxl0, pxlo3)

        p_x_results["Random Valid Input"] = Eval.loss_ttest(pxl0, pxli4)
        p_x_results["Random Valid Output"] = Eval.loss_ttest(pxl0, pxlo4)

        # Prediction Xanes
        p_y_results["Shuffle Input"] = Eval.loss_ttest(pyl0, pyli1)
        p_y_results["Shuffle Output"] = Eval.loss_ttest(pyl0, pylo1)

        p_y_results["Mean Train Input"] = Eval.loss_ttest(pyl0, pyli2)
        p_y_results["Mean Train Output"] = Eval.loss_ttest(pyl0, pylo2)

        p_y_results["Mean Std. Train Input"] = Eval.loss_ttest(pyl0, pyli3)
        p_y_results["Mean Std. Train Output"] = Eval.loss_ttest(pyl0, pylo3)

        p_y_results["Random Valid Input"] = Eval.loss_ttest(pyl0, pyli4)
        p_y_results["Random Valid Output"] = Eval.loss_ttest(pyl0, pylo4)

        # Reconstruction XYZ
        r_x_results["Shuffle Input"] = Eval.loss_ttest(rxl0, rxli1)
        r_x_results["Shuffle Output"] = Eval.loss_ttest(rxl0, rxlo1)

        r_x_results["Mean Train Input"] = Eval.loss_ttest(rxl0, rxli2)
        r_x_results["Mean Train Output"] = Eval.loss_ttest(rxl0, rxlo2)

        r_x_results["Mean Std. Train Input"] = Eval.loss_ttest(rxl0, rxli3)
        r_x_results["Mean Std. Train Output"] = Eval.loss_ttest(rxl0, rxlo3)

        r_x_results["Random Valid Input"] = Eval.loss_ttest(rxl0, rxli4)
        r_x_results["Random Valid Output"] = Eval.loss_ttest(rxl0, rxlo4)

        # Reconstruction Xanes
        r_y_results["Shuffle Input"] = Eval.loss_ttest(ryl0, ryli1)
        r_y_results["Shuffle Output"] = Eval.loss_ttest(ryl0, rylo1)

        r_y_results["Mean Train Input"] = Eval.loss_ttest(ryl0, ryli2)
        r_y_results["Mean Train Output"] = Eval.loss_ttest(ryl0, rylo2)

        r_y_results["Mean Std. Train Input"] = Eval.loss_ttest(ryl0, ryli3)
        r_y_results["Mean Std. Train Output"] = Eval.loss_ttest(ryl0, rylo3)

        r_y_results["Random Valid Input"] = Eval.loss_ttest(ryl0, ryli4)
        r_y_results["Random Valid Output"] = Eval.loss_ttest(ryl0, rylo4)

        print("    Prediction XYZ:")
        for k, v in p_x_results.items():
            print(f">>> {k:25}: {v}")
        print("\n")

        print("    Prediction Xanes:")
        for k, v in p_y_results.items():
            print(f">>> {k:25}: {v}")
        print("\n")

        print("    Reconstruction XYZ:")
        for k, v in r_x_results.items():
            print(f">>> {k:25}: {v}")
        print("\n")

        print("    Reconstruction Xanes:")
        for k, v in r_y_results.items():
            print(f">>> {k:25}: {v}")
        print("\n")

        test_results = {
            "ModelEvalResults-Reconstruction-XYZ": r_x_results,
            "ModelEvalResults-Reconstruction-Xanes": r_y_results,
            "ModelEvalResults-Prediction-XYZ:": p_x_results,
            "ModelEvalResults-Prediction-Xanes": p_y_results,
        }

        print(f"{'='*19} MLFlow: Evaluation Results Logged {'='*18}")
        return test_results

    def get_true_loss(self):
        true_recon_x_loss = []
        true_recon_y_loss = []
        true_pred_x_loss = []
        true_pred_y_loss = []

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            (
                recon_x,
                recon_y,
                pred_x,
                pred_y,
            ) = self.model.reconstruct_all_predict_all(inputs, labels)

            recon_x_loss = Eval.functional_mse(recon_x, inputs)
            recon_y_loss = Eval.functional_mse(recon_y, labels)
            pred_x_loss = Eval.functional_mse(pred_x, inputs)
            pred_y_loss = Eval.functional_mse(pred_y, labels)

            true_recon_x_loss.extend(recon_x_loss)
            true_recon_y_loss.extend(recon_y_loss)
            true_pred_x_loss.extend(pred_x_loss)
            true_pred_y_loss.extend(pred_y_loss)

        return (
            true_recon_x_loss,
            true_recon_y_loss,
            true_pred_x_loss,
            true_pred_y_loss,
        )

    def get_loss_input_shuffle(self):
        other_recon_x_loss = []
        other_recon_y_loss = []
        other_pred_x_loss = []
        other_pred_y_loss = []

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            idx = torch.randperm(inputs.shape[0])
            inputs_shuffle = inputs[idx]

            jdx = torch.randperm(labels.shape[0])
            labels_shuffle = labels[jdx]

            (
                recon_x,
                recon_y,
                pred_x,
                pred_y,
            ) = self.model.reconstruct_all_predict_all(inputs_shuffle, labels_shuffle)

            recon_x_loss = Eval.functional_mse(recon_x, inputs)
            recon_y_loss = Eval.functional_mse(recon_y, labels)
            pred_x_loss = Eval.functional_mse(pred_x, inputs)
            pred_y_loss = Eval.functional_mse(pred_y, labels)

            other_recon_x_loss.extend(recon_x_loss)
            other_recon_y_loss.extend(recon_y_loss)
            other_pred_x_loss.extend(pred_x_loss)
            other_pred_y_loss.extend(pred_y_loss)

        return (
            other_recon_x_loss,
            other_recon_y_loss,
            other_pred_x_loss,
            other_pred_y_loss,
        )

    def get_loss_output_shuffle(self):
        other_recon_x_loss = []
        other_recon_y_loss = []
        other_pred_x_loss = []
        other_pred_y_loss = []

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            idx = torch.randperm(inputs.shape[0])
            inputs_shuffle = inputs[idx]

            jdx = torch.randperm(labels.shape[0])
            labels_shuffle = labels[jdx]

            (
                recon_x,
                recon_y,
                pred_x,
                pred_y,
            ) = self.model.reconstruct_all_predict_all(inputs, labels)

            recon_x_loss = Eval.functional_mse(recon_x, inputs_shuffle)
            recon_y_loss = Eval.functional_mse(recon_y, labels_shuffle)
            pred_x_loss = Eval.functional_mse(pred_x, inputs_shuffle)
            pred_y_loss = Eval.functional_mse(pred_y, labels_shuffle)

            other_recon_x_loss.extend(recon_x_loss)
            other_recon_y_loss.extend(recon_y_loss)
            other_pred_x_loss.extend(pred_x_loss)
            other_pred_y_loss.extend(pred_y_loss)

        return (
            other_recon_x_loss,
            other_recon_y_loss,
            other_pred_x_loss,
            other_pred_y_loss,
        )

    def get_loss_input_mean_train(self):
        other_recon_x_loss = []
        other_recon_y_loss = []
        other_pred_x_loss = []
        other_pred_y_loss = []

        (
            recon_x_val,
            recon_y_val,
            pred_x_val,
            pred_y_val,
        ) = self.model.reconstruct_all_predict_all(self.mean_input, self.mean_output)

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            recon_x = recon_x_val.repeat(labels.shape[0], 1)
            recon_y = recon_y_val.repeat(labels.shape[0], 1)
            pred_x = pred_x_val.repeat(labels.shape[0], 1)
            pred_y = pred_y_val.repeat(labels.shape[0], 1)

            recon_x_loss = Eval.functional_mse(recon_x, inputs)
            recon_y_loss = Eval.functional_mse(recon_y, labels)
            pred_x_loss = Eval.functional_mse(pred_x, inputs)
            pred_y_loss = Eval.functional_mse(pred_y, labels)

            other_recon_x_loss.extend(recon_x_loss)
            other_recon_y_loss.extend(recon_y_loss)
            other_pred_x_loss.extend(pred_x_loss)
            other_pred_y_loss.extend(pred_y_loss)

        return (
            other_recon_x_loss,
            other_recon_y_loss,
            other_pred_x_loss,
            other_pred_y_loss,
        )

    def get_loss_output_mean_train(self):
        other_recon_x_loss = []
        other_recon_y_loss = []
        other_pred_x_loss = []
        other_pred_y_loss = []

        recon_x_val = pred_x_val = self.mean_input
        recon_y_val = pred_y_val = self.mean_output

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            recon_x = recon_x_val.repeat(labels.shape[0], 1)
            recon_y = recon_y_val.repeat(labels.shape[0], 1)
            pred_x = pred_x_val.repeat(labels.shape[0], 1)
            pred_y = pred_y_val.repeat(labels.shape[0], 1)

            recon_x_loss = Eval.functional_mse(recon_x, inputs)
            recon_y_loss = Eval.functional_mse(recon_y, labels)
            pred_x_loss = Eval.functional_mse(pred_x, inputs)
            pred_y_loss = Eval.functional_mse(pred_y, labels)

            other_recon_x_loss.extend(recon_x_loss)
            other_recon_y_loss.extend(recon_y_loss)
            other_pred_x_loss.extend(pred_x_loss)
            other_pred_y_loss.extend(pred_y_loss)

        return (
            other_recon_x_loss,
            other_recon_y_loss,
            other_pred_x_loss,
            other_pred_y_loss,
        )

    def get_loss_input_mean_sd_train(self):
        device = self.device

        other_recon_x_loss = []
        other_recon_y_loss = []
        other_pred_x_loss = []
        other_pred_y_loss = []

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            mean_sd_input = self.mean_input.repeat(labels.shape[0], 1) + torch.normal(
                torch.zeros([labels.shape[0], self.input_size], device=device),
                self.std_input,
            )
            mean_sd_output = self.mean_output.repeat(labels.shape[0], 1) + torch.normal(
                torch.zeros([labels.shape[0], self.output_size], device=device),
                self.std_output,
            )

            (
                recon_x,
                recon_y,
                pred_x,
                pred_y,
            ) = self.model.reconstruct_all_predict_all(mean_sd_input, mean_sd_output)

            recon_x_loss = Eval.functional_mse(recon_x, inputs)
            recon_y_loss = Eval.functional_mse(recon_y, labels)
            pred_x_loss = Eval.functional_mse(pred_x, inputs)
            pred_y_loss = Eval.functional_mse(pred_y, labels)

            other_recon_x_loss.extend(recon_x_loss)
            other_recon_y_loss.extend(recon_y_loss)
            other_pred_x_loss.extend(pred_x_loss)
            other_pred_y_loss.extend(pred_y_loss)

        return (
            other_recon_x_loss,
            other_recon_y_loss,
            other_pred_x_loss,
            other_pred_y_loss,
        )

    def get_loss_output_mean_sd_train(self):
        device = self.device
        other_recon_x_loss = []
        other_recon_y_loss = []
        other_pred_x_loss = []
        other_pred_y_loss = []

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            recon_x = self.mean_input.repeat(labels.shape[0], 1) + torch.normal(
                torch.zeros([labels.shape[0], self.input_size], device=device),
                self.std_input,
            )
            recon_y = self.mean_output.repeat(labels.shape[0], 1) + torch.normal(
                torch.zeros([labels.shape[0], self.output_size], device=device),
                self.std_output,
            )
            pred_x = self.mean_input.repeat(labels.shape[0], 1) + torch.normal(
                torch.zeros([labels.shape[0], self.input_size], device=device),
                self.std_input,
            )
            pred_y = self.mean_output.repeat(labels.shape[0], 1) + torch.normal(
                torch.zeros([labels.shape[0], self.output_size], device=device),
                self.std_output,
            )

            recon_x_loss = Eval.functional_mse(recon_x, inputs)
            recon_y_loss = Eval.functional_mse(recon_y, labels)
            pred_x_loss = Eval.functional_mse(pred_x, inputs)
            pred_y_loss = Eval.functional_mse(pred_y, labels)

            other_recon_x_loss.extend(recon_x_loss)
            other_recon_y_loss.extend(recon_y_loss)
            other_pred_x_loss.extend(pred_x_loss)
            other_pred_y_loss.extend(pred_y_loss)

        return (
            other_recon_x_loss,
            other_recon_y_loss,
            other_pred_x_loss,
            other_pred_y_loss,
        )

    def get_loss_input_random_valid(self):
        other_recon_x_loss = []
        other_recon_y_loss = []
        other_pred_x_loss = []
        other_pred_y_loss = []

        it = iter(self.valid_loader)

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            alt_inputs, alt_labels = next(it)

            alt_inputs = alt_inputs.to(self.device).float()
            alt_labels = alt_labels.to(self.device).float()

            if labels.shape[0] < alt_inputs.shape[0]:
                alt_inputs = alt_inputs[: labels.shape[0], :]
                alt_labels = alt_labels[: labels.shape[0], :]

            (
                recon_x,
                recon_y,
                pred_x,
                pred_y,
            ) = self.model.reconstruct_all_predict_all(alt_inputs, alt_labels)

            recon_x_loss = Eval.functional_mse(recon_x, inputs)
            recon_y_loss = Eval.functional_mse(recon_y, labels)
            pred_x_loss = Eval.functional_mse(pred_x, inputs)
            pred_y_loss = Eval.functional_mse(pred_y, labels)

            other_recon_x_loss.extend(recon_x_loss)
            other_recon_y_loss.extend(recon_y_loss)
            other_pred_x_loss.extend(pred_x_loss)
            other_pred_y_loss.extend(pred_y_loss)

        return (
            other_recon_x_loss,
            other_recon_y_loss,
            other_pred_x_loss,
            other_pred_y_loss,
        )

    def get_loss_output_random_valid(self):
        other_recon_x_loss = []
        other_recon_y_loss = []
        other_pred_x_loss = []
        other_pred_y_loss = []

        it = iter(self.valid_loader)

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            alt_inputs, alt_labels = next(it)

            alt_inputs = alt_inputs.to(self.device).float()
            alt_labels = alt_labels.to(self.device).float()

            if labels.shape[0] < alt_inputs.shape[0]:
                alt_inputs = alt_inputs[: labels.shape[0], :]
                alt_labels = alt_labels[: labels.shape[0], :]

            recon_x = alt_inputs
            recon_y = alt_labels

            idx = torch.randperm(labels.shape[0])
            jdx = torch.randperm(labels.shape[0])
            pred_x = alt_inputs[idx]
            pred_y = alt_labels[jdx]

            recon_x_loss = Eval.functional_mse(recon_x, inputs)
            recon_y_loss = Eval.functional_mse(recon_y, labels)
            pred_x_loss = Eval.functional_mse(pred_x, inputs)
            pred_y_loss = Eval.functional_mse(pred_y, labels)

            other_recon_x_loss.extend(recon_x_loss)
            other_recon_y_loss.extend(recon_y_loss)
            other_pred_x_loss.extend(pred_x_loss)
            other_pred_y_loss.extend(pred_y_loss)

        return (
            other_recon_x_loss,
            other_recon_y_loss,
            other_pred_x_loss,
            other_pred_y_loss,
        )
