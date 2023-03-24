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


def mse_avg(true_data, reconstructed_data, reg_param):
    mse = nn.MSELoss()
    if not reg_param:
        loss = mse(reconstructed_data, true_data)
    else:
        loss = reg_param * mse(reconstructed_data, true_data)
    return loss


def mse_sum(true_data, reconstructed_data, reg_param):
    mse = nn.MSELoss(reduction="sum")
    number_of_columns = true_data.shape[1]

    if not reg_param:
        loss = mse(reconstructed_data, true_data)
    else:
        loss = reg_param * mse(reconstructed_data, true_data)

    loss = loss / number_of_columns

    return loss

def emd(true_data, reconstructed_data, reg_param):
    wasserstein_distance_list = [
        wasserstein_distance(
            true_data.detach().numpy()[i, :], reconstructed_data.detach().numpy()[i, :]
        )
        for i in range(len(true_data))
    ]
    emd_loss = sum(wasserstein_distance_list)
    if not reg_param:
        loss = emd_loss
    else:
        loss = reg_param * emd_loss
    return loss


def l1(model_children, true_data, reg_param):
    l1_loss = 0.0
    values = true_data
    for i in range(len(model_children)):
        values = model_children[i](values)
        l1_loss += torch.mean(torch.abs(values))

    if not reg_param:
        loss = l1_loss
    else:
        loss = reg_param * l1_loss
    return loss

def mse_loss_emd_l1(model_children, true_data, reconstructed_data, reg_param, validate):
    """
    Computes a sparse loss function consisting of three terms: the Earth Mover's Distance (EMD) loss between the
    true and reconstructed data, the mean squared error (MSE) loss between the reconstructed and true data, and a
    L1 regularization term on the output of a list of model children.

    Args: model_children (list): List of PyTorch modules representing the model architecture to be regularized.
    true_data (torch.Tensor): The ground truth data, with shape (batch_size, num_features). reconstructed_data (
    torch.Tensor): The reconstructed data, with shape (batch_size, num_features). reg_param (float): The weight of
    the L1 regularization term in the loss function. validate (bool): If True, returns only the EMD loss. If False,
    computes the full loss with the L1 regularization term.

    Returns:
        If validate is False, returns a tuple with three elements:
        - loss (torch.Tensor): The full sparse loss function, with shape ().
        - emd_loss (float): The EMD loss between the true and reconstructed data.
        - l1_loss (float): The L1 regularization term on the output of the model children.

        If validate is True, returns only the EMD loss as a float.
    """
    mse = nn.MSELoss()
    mse_loss = mse(reconstructed_data, true_data)
    wasserstein_distance_list = [
        wasserstein_distance(
            true_data.detach().numpy()[i, :], reconstructed_data.detach().numpy()[i, :]
        )
        for i in range(len(true_data))
    ]
    emd_loss = sum(wasserstein_distance_list)

    l1_loss = 0
    values = true_data
    if not validate:
        for i in range(len(model_children)):
            values = model_children[i](values)
            l1_loss += torch.mean(torch.abs(values))

        loss = emd_loss + mse_loss + reg_param * l1_loss
        return loss, emd_loss, l1_loss
    else:
        return emd_loss


def mse_loss_l1(model_children, true_data, reconstructed_data, reg_param, validate):
    # This function is a modified version of the original function by George Dialektakis found at
    # https://github.com/Autoencoders-compression-anomaly/Deep-Autoencoders-Data-Compression-GSoC-2021
    # Released under the Apache License 2.0 found at https://www.apache.org/licenses/LICENSE-2.0.txt
    # Copyright 2021 George Dialektakis

    """
    Computes a sparse loss function consisting of two terms: the mean squared error (MSE) loss between the
    reconstructed and true data, and a L1 regularization term on the output of a list of model children.

    Args: model_children (list): List of PyTorch modules representing the model architecture to be regularized.
    true_data (torch.Tensor): The ground truth data, with shape (batch_size, num_features). reconstructed_data (
    torch.Tensor): The reconstructed data, with shape (batch_size, num_features). reg_param (float): The weight of
    the L1 regularization term in the loss function. validate (bool): If True, returns only the MSE loss. If False,
    computes the full loss with the L1 regularization term.

    Returns:
        If validate is False, returns a tuple with three elements:
        - loss (torch.Tensor): The full sparse loss function, with shape ().
        - mse_loss (float): The MSE loss between the true and reconstructed data.
        - l1_loss (float): The L1 regularization term on the output of the model children.

        If validate is True, returns only the MSE loss as a float.
    """
    mse = nn.MSELoss()
    mse_loss = mse(reconstructed_data, true_data)

    l1_loss = 0
    values = true_data
    if not validate:
        for i in range(len(model_children)):
            values = functional.relu(model_children[i](values))
            l1_loss += torch.mean(torch.abs(values))

        loss = mse_loss + reg_param * l1_loss
        return loss, mse_loss, l1_loss
    else:
        return mse_loss, 0, 0


def mse_sum_loss_l1(model_children, true_data, reconstructed_data, reg_param, validate):
    """
    Computes the sum of mean squared error (MSE) loss and L1 regularization loss.

    Args:
        model_children (list): List of PyTorch modules representing the encoder network.
        true_data (tensor): Ground truth tensor of shape (batch_size, input_size).
        reconstructed_data (tensor): Reconstructed tensor of shape (batch_size, input_size).
        reg_param (float): Regularization parameter for L1 loss.
        validate (bool): Whether to return only MSE loss or both MSE and L1 losses.

    Returns:
        If validate is False:
            loss (tensor): Total loss consisting of MSE loss and L1 regularization loss.
            mse_sum_loss (tensor): Mean squared error loss.
            l1_loss (tensor): L1 regularization loss.
        If validate is True:
            mse_sum_loss (tensor): Mean squared error loss.
            0 (int): Placeholder for MSE loss since it is not calculated during validation.
            0 (int): Placeholder for L1 loss since it is not calculated during validation.
    """
    mse_sum = nn.MSELoss(reduction="sum")
    mse_loss = mse_sum(reconstructed_data, true_data)
    number_of_columns = true_data.shape[1]

    mse_sum_loss = mse_loss / number_of_columns

    l1_loss = 0
    values = true_data
    if not validate:
        for i in range(len(model_children)):
            values = functional.relu(model_children[i](values))
            l1_loss += torch.mean(torch.abs(values))

        loss = mse_sum_loss + reg_param * l1_loss
        return loss, mse_sum_loss, l1_loss
    else:
        return mse_sum_loss, 0, 0


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
