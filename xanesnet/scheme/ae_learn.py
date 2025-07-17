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
import logging
import torch
import numpy as np

from collections import defaultdict
from sklearn.model_selection import RepeatedKFold

from xanesnet.scheme.base_learn import Learn
from xanesnet.utils.switch import LossSwitch, LossRegSwitch


class AELearn(Learn):
    def train(self, model, X, y):
        """
        Main training loop
        """
        train_loader, valid_loader, eval_loader = self.setup_dataloaders(X, y)

        optimizer, criterion, regularizer, scheduler = self.setup_components(model)
        model.to(self.device)

        valid_losses = {}
        logging.info(f"--- Starting Training for {self.epochs} epochs ---")
        for epoch in range(self.epochs):
            # Run the training phase
            train_losses = self._run_one_epoch(
                "train", train_loader, model, criterion, regularizer, optimizer
            )

            # Run the validation phase
            valid_losses = self._run_one_epoch(
                "valid", valid_loader, model, criterion, regularizer, optimizer=None
            )

            # Adjust learning rate if scheduler is used
            if self.lr_scheduler:
                scheduler.step()

            # Print and log losses using the returned dictionaries
            logging.info(
                f">>> Epoch {epoch+1:03d} | "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Valid Loss: {valid_losses['total']:.4f}"
            )

            self.log_loss("total_loss/train", train_losses["total"], epoch)
            self.log_loss("recon_loss/train", train_losses["recon"], epoch)
            self.log_loss("pred_loss/train", train_losses["predict"], epoch)

            self.log_loss("total_loss/valid", valid_losses["total"], epoch)
            self.log_loss("recon_loss/valid", valid_losses["recon"], epoch)
            self.log_loss("pred_loss/valid", valid_losses["predict"], epoch)

        logging.info("--- Training Finished ---")

        # Log model and final evaluation
        if self.mlflow_flag:
            logging.info("\nLogging the trained model as a run artifact...")
            self.log_mlflow(model)

        self.log_close()

        # The final score is the total loss from the last training epoch
        score = valid_losses["total"]

        return model, score

    def train_std(self):
        """
        Performs standard training run
        """
        model, _ = self.train(self.model, self.X, self.y)

        return self.model

    def train_kfold(self, x_data=None, y_data=None):
        """
        Performs K-fold cross-validation
        """
        X, y = self.X, self.y
        best_model = None
        best_score = float("inf")
        score_list = {"train_score": [], "test_score": []}

        kfold_splitter = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.seed_kfold,
        )

        # Initialise loss criterion and regularizer
        criterion = LossSwitch().get(self.loss)
        regularizer = LossRegSwitch()

        for i, (train_index, test_index) in enumerate(kfold_splitter.split(X, y)):
            # Deep copy model
            model = copy.deepcopy(self.model)

            #  Train model on the training split
            X_train, y_train = X[train_index], y[train_index]
            model, train_score = self.train(model, X_train, y_train)

            # Evaluate model on the test split
            X_test, y_test = X[test_index], y[test_index]
            test_loader = self._create_dataloader(X_test, y_test, shuffle=False)

            test_losses = self._run_one_epoch(
                "valid", test_loader, model, criterion, regularizer
            )

            test_score = test_losses["total"]
            score_list["train_score"].append(train_score)
            score_list["test_score"].append(test_score)

            if test_score < best_score:
                logging.info(
                    f"--- [Fold {i+1}] New best model found with test score: {test_score:.6f} ---"
                )
                best_score = test_score
                best_model = model

        # Log final averaged results
        logging.info("--- K-Fold Cross-Validation Finished ---")
        logging.info(
            f"Average Train Score: {np.mean(score_list['train_score']):.6f} +/- {np.std(score_list['train_score']):.6f}"
        )
        logging.info(
            f"Average Test Score : {np.mean(score_list['test_score']):.6f} +/- {np.std(score_list['test_score']):.6f}"
        )

        return best_model

    def _run_one_epoch(
        self, phase, loader, model, criterion, regularizer, optimizer=None
    ):
        """Runs a single epoch of training or validation."""
        is_train = phase == "train"
        model.train() if is_train else model.eval()

        epoch_losses = defaultdict(float)
        device = self.device

        with torch.set_grad_enabled(is_train):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device).float(), labels.to(device).float()

                # Zero the parameter gradients only during training
                if is_train:
                    optimizer.zero_grad()

                recon, predict = model(inputs)
                # Calculate individual loss
                recon_loss = criterion(recon, inputs).mean()
                predict_loss = criterion(predict, labels).mean()
                # Calculate total loss
                loss = recon_loss + predict_loss
                # Add regularization loss
                loss_reg = regularizer.loss(model, self.loss_reg, device)
                loss += self.loss_lambda * loss_reg

                if is_train:
                    loss.backward()
                    optimizer.step()

                epoch_losses["total"] += loss.item()
                epoch_losses["recon"] += recon_loss.item()
                epoch_losses["predict"] += predict_loss.item()

        running_losses = {k: v / len(loader) for k, v in epoch_losses.items()}

        return running_losses

    def tensorboard_layout(self):
        layout = {
            "Losses": {
                "Total Losses": (
                    "Multiline",
                    ["total_loss/train", "total_loss/valid"],
                ),
                "Reconstruction Losses": (
                    "Multiline",
                    ["recon_loss/train", "recon_loss/valid"],
                ),
                "Prediction Losses": (
                    "Multiline",
                    ["pred_loss/train", "pred_loss/valid"],
                ),
            },
        }
        return layout
