import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import modules.utils as utils
import modules.helper as helper
import os
import sys

baler_dir = os.path.join(os.path.dirname(__file__), "..", "baler")
modules_dir = os.path.join(baler_dir, "modules")
sys.path.append(modules_dir)
import training
from training import fit
from training import validate
from training import seed_worker


def test_fit():
    # Initialize some test data
    train_data = torch.randn(100, 10)
    test_data = torch.randn(20, 10)

    # Create a simple test model
    model = torch.nn.Linear(10, 1)

    # Set up optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Set up a dataloader
    train_dl = DataLoader(train_data, batch_size=10, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=10, shuffle=True)

    # Train the model
    epoch_loss, mse_loss, l1_loss, trained_model = fit(
        model, train_dl, model.children(), 0.01, optimizer, 0.5, True, 10
    )

    # Test that the trained model is indeed a PyTorch model object
    assert isinstance(trained_model, torch.nn.Module)

    # Test that the losses are non-negative
    assert epoch_loss >= 0
    assert mse_loss >= 0
    assert l1_loss >= 0


def test_validate():
    # Initialize some test data
    train_data = torch.randn(100, 10)
    test_data = torch.randn(20, 10)

    # Create a simple test model
    model = torch.nn.Linear(10, 1)

    # Set up optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Set up a dataloader
    train_dl = DataLoader(train_data, batch_size=10, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=10, shuffle=True)

    # Train the model without L1 regularization
    _, _, _, trained_model = fit(
        model, train_dl, model.children(), 0.01, optimizer, 0.5, False, 10
    )

    # Test the validation function without L1 regularization
    validation_loss = validate(trained_model, test_dl, model.children(), 0.01)

    # Test that the validation loss is non-negative
    assert validation_loss >= 0

    # Train the model with L1 regularization
    _, _, l1_loss, trained_model_l1 = fit(
        model, train_dl, model.children(), 0.01, optimizer, 0.5, True, 10
    )

    # Test the validation function with L1 regularization
    validation_loss_l1 = validate(trained_model_l1, test_dl, model.children(), 0.01)

    # Test that the validation loss is non-negative
    assert validation_loss_l1 >= 0

    # Test that L1 loss is lower than validation loss with L1 regularization
    assert l1_loss < validation_loss_l1


def test_seed_worker():
    # Set seed for main process
    torch.manual_seed(0)

    # Call seed_worker() function for worker 1 and check if seeds are set properly
    seed_worker(1)
    worker_seed = torch.initial_seed() % 2 ** 32
    np_seed = np.random.get_state()[1][0]
    assert worker_seed == 0
    assert np_seed == 0
