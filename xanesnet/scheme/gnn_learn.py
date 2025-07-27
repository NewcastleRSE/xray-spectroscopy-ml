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
import torch_geometric
import numpy as np

from sklearn.model_selection import RepeatedKFold, train_test_split

from xanesnet.scheme.base_learn import Learn
from xanesnet.utils.switch import LossSwitch, LossRegSwitch


class GNNLearn(Learn):
    def train(self, model, X, y=None):
        """
        Main training loop
        """
        train_loader, valid_loader, eval_loader = self.setup_dataloaders(X)

        optimizer, criterion, regularizer, scheduler = self.setup_components(model)
        model.to(self.device)

        valid_loss = 0.0
        logging.info(f"--- Starting Training for {self.epochs} epochs ---")
        for epoch in range(self.epochs):
            # Run training phase
            train_loss = self._run_one_epoch(
                "train", train_loader, model, criterion, regularizer, optimizer
            )

            # Run validation phase
            valid_loss = self._run_one_epoch(
                "valid", valid_loader, model, criterion, regularizer, optimizer=None
            )

            # Adjust learning rate if scheduler is used
            if self.lr_scheduler:
                scheduler.step()

            # Logging for the current epoch
            logging.info(
                f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}"
            )
            self.log_loss("loss/train", train_loss, epoch)
            self.log_loss("loss/validation", valid_loss, epoch)

        logging.info("--- Training Finished ---")

        # Log model and final evaluation
        if self.mlflow_flag:
            logging.info("\nLogging the trained model as a run artifact...")
            self.log_mlflow(model)

        self.log_close()

        # The final score is the validation loss from the last epoch
        score = valid_loss

        return model, score

    def train_std(self):
        """
        Performs standard training run
        """
        model, _ = self.train(self.model, self.X, self.y)

        return model

    def train_kfold(self):
        """
        Performs K-fold cross-validation
        """
        X = self.X
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

        # Generate indices for k-fold splits
        indices = list(range(len(X)))

        for i, (train_index, test_index) in enumerate(kfold_splitter.split(indices)):
            # Deep copy model
            model = copy.deepcopy(self.model)

            train_data = X[train_index]
            model, train_score = self.train(model, train_data)

            # Evaluate model on the test split
            test_data = X[test_index]
            test_loader = torch_geometric.data.DataLoader(
                test_data,
                batch_size=self.batch_size,
                shuffle=False,
            )

            test_score = self._run_one_epoch(
                "valid", test_loader, model, criterion, regularizer
            )

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

    def train_bootstrap(self):
        """
        Trains multiple models on bootstrap resamples of the provided dataset.
        """

        model_list = []
        n_samples = int(len(self.X))

        # Size of each bootstrap sample
        sample_size = int(n_samples * self.n_size)

        for i in range(self.n_boot):
            rng = np.random.default_rng(self.weight_seed_boot[i])

            # Generate all random indices at once
            bootstrap_indices = rng.choice(n_samples, size=sample_size, replace=True)

            # Create the bootstrap sample in a single, fast indexing operation
            X_boot = self.X[bootstrap_indices]

            # Deep copy model and re-initialise model weight using bootstrap seeds
            model = copy.deepcopy(self.model)
            self.weights_params["seed"] = self.weight_seed_boot[i]
            model = self._init_model_weights(model, **self.weights_params)

            # Train the model on the bootstrap sample
            model, _ = self.train(model, X_boot)

            model_list.append(model)

        return model_list

    def _run_one_epoch(
        self, phase, loader, model, criterion, regularizer, optimizer=None
    ):
        """Runs a single epoch of training or validation."""
        is_train = phase == "train"
        model.train() if is_train else model.eval()

        running_loss = 0.0
        device = self.device

        with torch.set_grad_enabled(is_train):
            for batch in loader:
                batch.to(device)

                # Zero the parameter gradients only during training
                if is_train:
                    optimizer.zero_grad()

                predict = model(batch)
                predict = torch.flatten(predict)
                loss = criterion(predict, batch.y.float())

                # Add regularization loss
                loss_reg = regularizer.loss(model, self.loss_reg, device)
                loss += self.loss_lambda * loss_reg

                if is_train:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

        return running_loss / len(loader)

    def setup_dataloaders(self, X, y=None):
        """
        Splits data and creates DataLoaders.
        """
        indices = list(range(len(X)))

        if self.model_eval:
            # Data split: train/valid/test
            train_ratio, valid_ratio, eval_ratio = 0.75, 0.15, 0.10

            # First split: train vs (test + eval)
            train_idx, valid_idx = train_test_split(
                indices, test_size=1 - train_ratio, random_state=self.seed
            )

            # Second split: test vs eval
            valid_idx, eval_idx = train_test_split(
                valid_idx, test_size=eval_ratio / (eval_ratio + valid_ratio)
            )
        else:
            train_idx, valid_idx = train_test_split(
                indices, test_size=0.2, random_state=self.seed
            )

        train_loader = torch_geometric.data.DataLoader(
            X[train_idx], batch_size=self.batch_size, shuffle=True
        )
        valid_loader = torch_geometric.data.DataLoader(
            X[valid_idx], batch_size=self.batch_size, shuffle=False
        )

        if self.model_eval:
            eval_loader = torch_geometric.data.DataLoader(
                X[eval_idx], batch_size=self.batch_size, shuffle=False
            )
        else:
            eval_loader = None

        return train_loader, valid_loader, eval_loader

    def tensorboard_layout(self):
        layout = {
            "Losses": {
                "Losses": ["Multiline", ["loss/train", "loss/validation"]],
            },
        }
        return layout
