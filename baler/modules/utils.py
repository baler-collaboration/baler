# Copyright 2022 Baler Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from torch.nn import functional
from tqdm import tqdm

factor = 0.5
min_lr = 1e-6


class Loss:
    """Class which contains all loss functions used for training the model.


    Args:
        model_children (list): List of model parameters
        true_data (torch.Tensor): Input data as torch.Tensor
        reconstructed_data (torch.Tensor): Input data ran through model as torch.Tensor
        reg_param (float): Proportionality constant for the L1 loss

    """

    def __init__(self, model_children, true_data, reconstructed_data, reg_param):
        self.true_data = true_data  # Input data
        self.reconstructed_data = (
            reconstructed_data  # Input data evaluated through the model during training
        )
        self.model_children = (
            model_children  # pytorch structure containing model weights
        )

    def mse_avg(true_data, reconstructed_data):
        """The Mean Squared Error (MSE) function. The most commonly used loss function in machine learning, defined as
            `MSE = 1/m 1/n \sum^m_{j=1} \sum_{i=1}^n (true_data_i - reconstructed_data_i)^2_j`

        Args:
            true_data (torch.Tensor): Input data as torch.Tensor
            reconstructed_data (torch.Tensor): Input data ran through model as torch.Tensor

        Returns:
            torch.Tensor: The loss after one iteration
        """
        mse = nn.MSELoss()
        loss = mse(reconstructed_data, true_data)
        return loss

    def mse_sum(true_data, reconstructed_data):
        """A variant of the MSE function, but now defined as:
            `MSE = 1/n \sum^m_{j=1} \sum_{i=1}^n (true_data_i - reconstructed_data_i)^2_j`

        Args:
            true_data (torch.Tensor): Input data as torch.Tensor
            reconstructed_data (torch.Tensor): Input data ran through model as torch.Tensor

        Returns:
            torch.Tensor: The loss after one iteration
        """
        mse = nn.MSELoss(reduction="sum")
        number_of_columns = true_data.shape[1]

        loss = mse(reconstructed_data, true_data) / number_of_columns
        return loss

    def emd(true_data, reconstructed_data):
        """Another loss function called the Earths Movers Distance (EMD). This loss measures the distance between the input data and the reconstructed data.
            The functionality is implemented from `scipy.stats.wasserstein_distance`.

        Args:
            true_data (torch.Tensor): Input data as torch.Tensor
            reconstructed_data (torch.Tensor): Input data ran through model as torch.Tensor

        Returns:
            torch.Tensor: The loss after one iteration
        """
        wasserstein_distance_list = [
            wasserstein_distance(
                true_data.detach().numpy()[i, :],
                reconstructed_data.detach().numpy()[i, :],
            )
            for i in range(len(true_data))
        ]
        wasserstein_distance_tensor = torch.Tensor(wasserstein_distance_list)
        wasserstein_distance_tensor.requires_grad_()
        loss = torch.sum(wasserstein_distance_tensor)
        return loss

    def l1(model_children, true_data, reg_param):
        """The "Lasso" Regularizer term, also known as the L1 term, defined as:
        `L1 = reg_param * \sum_i \abs{w_i}`

        The implementation of this term is heavily inspired from https://github.com/syorami/Autoencoders-Variants by `tmac1997`
        and full credit goes to him for this implementation.

        Args:
            model_children (list): List containing model parameters, most importantly the model weights.
            true_data (torch.Tensor): Input data as torch.Tensor
            reconstructed_data (torch.Tensor): Input data ran through model as torch.Tensor

        Returns:
            torch.Tensor: The loss after one iteration
        """
        l1_loss = 0.0
        values = true_data
        for i in range(len(model_children)):
            values = functional.relu(model_children[i](values))
            l1_loss += torch.mean(torch.abs(values))

        loss = reg_param * l1_loss
        return loss

    def __call__(self):
        return


# Accuracy function still WIP. Not working properly.
# Probably has to do with total_correct counter.


def accuracy(model, dataloader):
    """
    Computes the accuracy of a PyTorch model on a given dataset.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): DataLoader object containing the dataset to evaluate on.

    Returns:
        accuracy_frac (float): The fraction of correctly classified instances in the dataset.
    """
    print("Accuracy")
    model.eval()

    total_correct = 0
    total_instances = 0

    with torch.no_grad():
        for data in tqdm(dataloader):
            x, _ = data
            classifications = torch.argmax(x)

            correct_pred = torch.sum(classifications == x).item()

            total_correct += correct_pred
            total_instances += len(x)

    accuracy_frac = round(total_correct / total_instances, 3)
    print(accuracy_frac)
    return accuracy_frac


class EarlyStopping:
    """
    Class to perform early stopping during model training.

    Attributes:
        patience (int): The number of epochs to wait before stopping the training process if the
            validation loss doesn't improve.
        min_delta (float): The minimum difference between the new loss and the previous best loss
            for the new loss to be considered an improvement.
        counter (int): Counts the number of times the validation loss hasn't improved.
        best_loss (float): The best validation loss observed so far.
        early_stop (bool): Flag that indicates whether early stopping criteria have been met.
    """

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience  # Nr of times we allow val. loss to not improve before early stopping
        self.min_delta = min_delta  # min(new loss - best loss) for new loss to be considered improvement
        self.counter = 0  # counts nr of times val_loss dosent improve
        self.best_loss = None
        self.early_stop = False

    def __call__(self, train_loss):
        if self.best_loss is None:
            self.best_loss = train_loss

        elif self.best_loss - train_loss > self.min_delta:
            self.best_loss = train_loss
            self.counter = 0  # Resets if val_loss improves

        elif self.best_loss - train_loss < self.min_delta:
            self.counter += 1

            print(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("Early Stopping")
                self.early_stop = True


class LRScheduler:
    """
    A learning rate scheduler that adjusts the learning rate of an optimizer based on the training loss.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be adjusted.
        patience (int): The number of epochs with no improvement in training loss after which the learning rate
            will be reduced.
        min_lr (float, optional): The minimum learning rate that can be reached (default: 1e-6).
        factor (float, optional): The factor by which the learning rate will be reduced (default: 0.1).

    Attributes:
        lr_scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): The PyTorch learning rate scheduler that
            actually performs the adjustments.

    Example usage:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        lr_scheduler = LRScheduler(optimizer, patience=3, min_lr=1e-5, factor=0.5)

        for epoch in range(num_epochs):
            train_loss = train(model, train_data_loader)
            lr_scheduler(train_loss)
            # ...
    """

    def __init__(self, optimizer, patience, min_lr=min_lr, factor=factor):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        # Maybe add if statements for selectment of lr schedulers
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
        )

    def __call__(self, train_loss):
        self.lr_scheduler.step(train_loss)
