import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import modules.helper as helper
import modules.utils as utils


def fit(model, train_dl, train_ds, model_children, regular_param, optimizer, RHO, l1):
    print("### Beginning Training")

    model.train()

    running_loss = 0.0
    n_data = int(len(train_ds) / train_dl.batch_size)
    for inputs in tqdm(
        train_dl, total=n_data, desc="# Training", file=sys.stdout
    ):
        optimizer.zero_grad()
        reconstructions = model(inputs)
        loss, mse_loss, l1_loss = utils.sparse_loss_function_L1(
            model_children=model_children,
            true_data=inputs,
            reconstructed_data=reconstructions,
            reg_param=regular_param,
            validate=False,
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dl)
    print(f"# Finished. Training Loss: {loss:.6f}")
    return epoch_loss, mse_loss, l1_loss


def validate(model, test_dl, test_ds, model_children, reg_param):
    print("### Beginning Validating")

    model.eval()

    running_loss = 0.0
    n_data = int(len(test_ds) / test_dl.batch_size)
    with torch.no_grad():
        for inputs in tqdm(
            test_dl, total=n_data, desc="# Validating", file=sys.stdout
        ):
            reconstructions = model(inputs)
            loss = utils.sparse_loss_function_L1(
                model_children=model_children,
                true_data=inputs,
                reconstructed_data=reconstructions,
                reg_param=reg_param,
                validate=True,
            )
            running_loss += loss.item()

    epoch_loss = running_loss / len(test_dl)
    print(f"# Finished. Validation Loss: {loss:.6f}")
    return epoch_loss


def train(model, variables, train_data, test_data, parent_path, config):
    learning_rate = config["lr"]
    bs = config["batch_size"]
    reg_param = config["reg_param"]
    RHO = config["RHO"]
    l1 = config["l1"]
    epochs = config["epochs"]
    latent_space_size = config["latent_space_size"]

    model_children = list(model.children())

    # Initialize model with appropriate device
    device = helper.get_device()
    model = model.to(device)

    # Converting data to tensors
    train_ds = torch.tensor(train_data.values, dtype=torch.float64, device=device)
    valid_ds = torch.tensor(train_data.values, dtype=torch.float64, device=device)

    # Pushing input data into the torch-DataLoader object and combines into one DataLoaders object (a basic wrapper
    # around several DataLoader objects).
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs, drop_last=True)

    ## Select Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ## Activate early stopping
    if config["early_stopping"] == True:
        early_stopping = utils.EarlyStopping(
            patience=config["patience"], min_delta=config["min_delta"]
        )  # Changes to patience & min_delta can be made in configs

    ## Activate LR Scheduler
    if config["lr_scheduler"] == True:
        lr_scheduler = utils.LRScheduler(
            optimizer=optimizer, patience=config["patience"]
        )

    # train and validate the autoencoder neural network
    train_loss = []
    val_loss = []
    start = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        train_epoch_loss, mse_loss_fit, regularizer_loss_fit = fit(
            model=model,
            train_dl=train_dl,
            train_ds=train_ds,
            model_children=model_children,
            optimizer=optimizer,
            RHO=RHO,
            regular_param=reg_param,
            l1=l1,
        )

        train_loss.append(train_epoch_loss)

        val_epoch_loss = validate(
            model=model,
            test_dl=valid_dl,
            test_ds=valid_ds,
            model_children=model_children,
            reg_param=reg_param,
        )
        val_loss.append(val_epoch_loss)
        if config["lr_scheduler"] is True:
            lr_scheduler(val_epoch_loss)
        if config["early_stopping"] is True:
            early_stopping(val_epoch_loss)
            if early_stopping.early_stop:
                break

    end = time.time()

    print(f"{(end - start) / 60:.3} minutes")
    pd.DataFrame({"Train Loss": train_loss, "Val Loss": val_loss}).to_csv(
        parent_path + "loss_data.csv"
    )

    data_as_tensor = torch.tensor(test_data.values, dtype=torch.float64)
    data_as_tensor = data_as_tensor.to(device)
    pred_as_tensor = model(data_as_tensor)

    return data_as_tensor, pred_as_tensor
