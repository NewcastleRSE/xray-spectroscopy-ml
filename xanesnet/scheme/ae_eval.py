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


class AEEval(Eval):
    def eval(self):
        rl0, pl0 = self.get_true_loss()

        rli1, pli1 = self.get_loss_input_shuffle()
        rlo1, plo1 = self.get_loss_output_shuffle()

        rli2, pli2 = self.get_loss_input_mean_train()
        rlo2, plo2 = self.get_loss_output_mean_train()

        rli3, pli3 = self.get_loss_input_mean_sd_train()
        rlo3, plo3 = self.get_loss_output_mean_sd_train()

        rli4, pli4 = self.get_loss_input_random_valid()
        rlo4, plo4 = self.get_loss_output_random_valid()

        p_results = {}
        r_results = {}

        p_results["Shuffle Input"] = Eval.loss_ttest(pl0, pli1)
        p_results["Shuffle Output"] = Eval.loss_ttest(pl0, plo1)

        r_results["Shuffle Input"] = Eval.loss_ttest(rl0, rli1)
        r_results["Shuffle Output"] = Eval.loss_ttest(rl0, rlo1)

        p_results["Mean Train Input"] = Eval.loss_ttest(pl0, pli2)
        p_results["Mean Train Output"] = Eval.loss_ttest(pl0, plo2)

        r_results["Mean Train Input"] = Eval.loss_ttest(rl0, rli2)
        r_results["Mean Train Output"] = Eval.loss_ttest(rl0, rlo2)

        p_results["Mean Std. Train Input"] = Eval.loss_ttest(pl0, pli3)
        p_results["Mean Std. Train Output"] = Eval.loss_ttest(pl0, plo3)

        r_results["Mean Std. Train Input"] = Eval.loss_ttest(rl0, rli3)
        r_results["Mean Std. Train Output"] = Eval.loss_ttest(rl0, rlo3)

        p_results["Random Valid Input"] = Eval.loss_ttest(pl0, pli4)
        p_results["Random Valid Output"] = Eval.loss_ttest(pl0, plo4)

        r_results["Random Valid Input"] = Eval.loss_ttest(rl0, rli4)
        r_results["Random Valid Output"] = Eval.loss_ttest(rl0, rlo4)

        print("    Prediction:")
        for k, v in p_results.items():
            print(f">>> {k:25}: {v}")
        print("\n")

        print("    Reconstruction:")
        for k, v in r_results.items():
            print(f">>> {k:25}: {v}")

        test_results = {
            "ModelEvalResults-Reconstruction": r_results,
            "ModelEvalResults-Prediction:": p_results,
        }

        return test_results

    def get_true_loss(self):
        true_recon_loss = []
        true_pred_loss = []

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            recon_target, pred_target = self.model(inputs)

            recon_loss = Eval.functional_mse(recon_target, inputs)
            pred_loss = Eval.functional_mse(pred_target, labels)

            true_recon_loss.extend(recon_loss)
            true_pred_loss.extend(pred_loss)

        return true_recon_loss, true_pred_loss

    def get_loss_input_shuffle(self):
        other_recon_loss = []
        other_pred_loss = []

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            idx = torch.randperm(inputs.shape[0])
            inputs_shuffle = inputs[idx]

            recon_target, pred_target = self.model(inputs_shuffle)

            recon_loss = Eval.functional_mse(recon_target, inputs)
            pred_loss = Eval.functional_mse(pred_target, labels)

            other_recon_loss.extend(recon_loss)
            other_pred_loss.extend(pred_loss)

        return other_recon_loss, other_pred_loss

    def get_loss_output_shuffle(self):
        other_recon_loss = []
        other_pred_loss = []

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            idx = torch.randperm(labels.shape[0])
            labels_shuffle = labels[idx]

            recon_target, pred_target = self.model(inputs)

            jdx = torch.randperm(recon_target.shape[0])
            recon_target_shuffle = recon_target[jdx]

            recon_loss = Eval.functional_mse(recon_target_shuffle, inputs)
            pred_loss = Eval.functional_mse(pred_target, labels_shuffle)

            other_recon_loss.extend(recon_loss)
            other_pred_loss.extend(pred_loss)

        return other_recon_loss, other_pred_loss

    def get_loss_input_mean_train(self):
        other_recon_loss = []
        other_pred_loss = []

        recon_target_val, pred_target_val = self.model(self.mean_input)

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            recon_target = recon_target_val.repeat(labels.shape[0], 1)
            pred_target = pred_target_val.repeat(labels.shape[0], 1)

            recon_loss = Eval.functional_mse(recon_target, inputs)
            pred_loss = Eval.functional_mse(pred_target, labels)

            other_recon_loss.extend(recon_loss)
            other_pred_loss.extend(pred_loss)

        return other_recon_loss, other_pred_loss

    def get_loss_output_mean_train(self):
        other_recon_loss = []
        other_pred_loss = []

        recon_target_val = self.mean_input
        pred_target_val = self.mean_output

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels = inputs.float(), labels.float()

            recon_target = recon_target_val.repeat(labels.shape[0], 1)
            pred_target = pred_target_val.repeat(labels.shape[0], 1)

            recon_loss = Eval.functional_mse(recon_target, inputs)
            pred_loss = Eval.functional_mse(pred_target, labels)

            other_recon_loss.extend(recon_loss)
            other_pred_loss.extend(pred_loss)

        return other_recon_loss, other_pred_loss

    def get_loss_input_mean_sd_train(self):
        other_recon_loss = []
        other_pred_loss = []
        device = self.device

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            mean_sd_input = self.mean_input.repeat(labels.shape[0], 1) + torch.normal(
                torch.zeros([labels.shape[0], self.input_size], device=device),
                self.std_input,
            )

            recon_target, pred_target = self.model(mean_sd_input)

            recon_loss = Eval.functional_mse(recon_target, inputs)
            pred_loss = Eval.functional_mse(pred_target, labels)

            other_recon_loss.extend(recon_loss)
            other_pred_loss.extend(pred_loss)

        return other_recon_loss, other_pred_loss

    def get_loss_output_mean_sd_train(self):
        other_recon_loss = []
        other_pred_loss = []
        device = self.device

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            recon_target = self.mean_input.repeat(labels.shape[0], 1) + torch.normal(
                torch.zeros([labels.shape[0], self.input_size], device=device),
                self.std_input,
            )
            pred_target = self.mean_output.repeat(labels.shape[0], 1) + torch.normal(
                torch.zeros([labels.shape[0], self.output_size], device=device),
                self.std_output,
            )

            recon_loss = Eval.functional_mse(recon_target, inputs)
            pred_loss = Eval.functional_mse(pred_target, labels)

            other_recon_loss.extend(recon_loss)
            other_pred_loss.extend(pred_loss)

        return other_recon_loss, other_pred_loss

    def get_loss_input_random_valid(self):
        other_recon_loss = []
        other_pred_loss = []
        device = self.device

        it = iter(self.valid_loader)

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            alt_inputs, alt_labels = next(it)

            alt_inputs = alt_inputs.to(device).float()
            alt_labels = alt_labels.to(device).float()

            if labels.shape[0] < alt_inputs.shape[0]:
                alt_inputs = alt_inputs[: labels.shape[0], :]
                alt_labels = alt_labels[: labels.shape[0], :]

            recon_target, pred_target = self.model(alt_inputs)

            recon_loss = Eval.functional_mse(recon_target, inputs)
            pred_loss = Eval.functional_mse(pred_target, labels)

            other_recon_loss.extend(recon_loss)
            other_pred_loss.extend(pred_loss)

        return other_recon_loss, other_pred_loss

    def get_loss_output_random_valid(self):
        other_recon_loss = []
        other_pred_loss = []
        device = self.device

        it = iter(self.valid_loader)

        for inputs, labels in self.eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            alt_inputs, alt_labels = next(it)

            alt_inputs = alt_inputs.to(device).float()
            alt_labels = alt_labels.to(device).float()

            if labels.shape[0] < alt_inputs.shape[0]:
                alt_inputs = alt_inputs[: labels.shape[0], :]
                alt_labels = alt_labels[: labels.shape[0], :]

            recon_target, pred_target = alt_inputs, alt_labels

            recon_loss = Eval.functional_mse(recon_target, inputs)
            pred_loss = Eval.functional_mse(pred_target, labels)

            other_recon_loss.extend(recon_loss)
            other_pred_loss.extend(pred_loss)

        return other_recon_loss, other_pred_loss
