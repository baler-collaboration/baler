import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import modules.helper as helper
import modules.utils as utils


def fit(model, train_dl, train_ds, model_children, regular_param, optimizer, RHO, l1):
    print("### Beginning Training")

    model.train()

    running_loss = 0.0
    counter = 0
    n_data = int(len(train_ds) / train_dl.batch_size)
    for inputs in tqdm(train_dl):
        counter += 1
        inputs = inputs.to(model.device)

        optimizer.zero_grad()
        reconstructions = model(inputs)
        loss, mse_loss, l1_loss = utils.sparse_loss_function_EMD_L1(
            model_children=model_children,
            true_data=inputs,
            reconstructed_data=reconstructions,
            reg_param=regular_param,
            validate=False,
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / counter
    print(f"# Finished. Training Loss: {loss:.6f}")
    return epoch_loss, mse_loss, l1_loss, model


def validate(model, test_dl, test_ds, model_children, reg_param):
    print("### Beginning Validating")

    model.eval()
    counter = 0
    running_loss = 0.0
    n_data = int(len(test_ds) / test_dl.batch_size)
    with torch.no_grad():
        for inputs in tqdm(test_dl):
            counter += 1
            inputs = inputs.to(model.device)

            reconstructions = model(inputs)
            loss = utils.sparse_loss_function_EMD_L1(
                model_children=model_children,
                true_data=inputs,
                reconstructed_data=reconstructions,
                reg_param=reg_param,
                validate=True,
            )
            running_loss += loss.item()

    epoch_loss = running_loss / counter
    print(f"# Finished. Validation Loss: {loss:.6f}")
    return epoch_loss


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(model, variables, train_data, test_data, parent_path, config):
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(0)

    test_size = config.test_size
    learning_rate = config.lr
    bs = config.batch_size
    reg_param = config.reg_param
    rho = config.RHO
    l1 = config.l1
    epochs = config.epochs
    latent_space_size = config.latent_space_size

    model_children = list(model.children())

    # Initialize model with appropriate device
    device = helper.get_device()
    model = model.to(device)

    # Converting data to tensors
    train_ds = torch.tensor(train_data.values, dtype=torch.float64, device=device)
    valid_ds = torch.tensor(train_data.values, dtype=torch.float64, device=device)

    # Pushing input data into the torch-DataLoader object and combines into one DataLoaders object (a basic wrapper
    # around several DataLoader objects).
    train_dl = DataLoader(
        train_ds, batch_size=bs, shuffle=False, worker_init_fn=seed_worker, generator=g
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=bs, worker_init_fn=seed_worker, generator=g
    )  # Used to be batch_size = bs * 2

    # Select Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Activate early stopping
    if config.early_stopping:
        early_stopping = utils.EarlyStopping(
            patience=config.patience, min_delta=config.min_delta
        )  # Changes to patience & min_delta can be made in configs

    # Activate LR Scheduler
    if config.lr_scheduler:
        lr_scheduler = utils.LRScheduler(optimizer=optimizer, patience=config.patience)

    # train and validate the autoencoder neural network
    train_loss = []
    val_loss = []
    start = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        train_epoch_loss, mse_loss_fit, regularizer_loss_fit, trained_model = fit(
            model=model,
            train_dl=train_dl,
            train_ds=train_ds,
            model_children=model_children,
            optimizer=optimizer,
            RHO=rho,
            regular_param=reg_param,
            l1=l1,
        )

        train_loss.append(train_epoch_loss)

        if test_size:
            val_epoch_loss = validate(
                model=trained_model,
                test_dl=valid_dl,
                test_ds=valid_ds,
                model_children=model_children,
                reg_param=reg_param,
            )
            val_loss.append(val_epoch_loss)
        else:
            val_epoch_loss = train_epoch_loss
            val_loss.append(val_epoch_loss)

        if config.lr_scheduler:
            lr_scheduler(val_epoch_loss)
        if config.early_stopping:
            early_stopping(val_epoch_loss)
            if early_stopping.early_stop:
                break

    end = time.time()

    print(f"{(end - start) / 60:.3} minutes")
    pd.DataFrame({"Train Loss": train_loss, "Val Loss": val_loss}).to_csv(
        parent_path + "loss_data.csv"
    )

    data_as_tensor = torch.tensor(test_data.values, dtype=torch.float64)
    data_as_tensor = data_as_tensor.to(model.device)
    pred_as_tensor = trained_model(data_as_tensor)

    return data_as_tensor, pred_as_tensor, trained_model
