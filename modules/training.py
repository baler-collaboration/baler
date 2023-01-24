import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
import pandas as pd

import modules.loss_functions as loss_functions

def fit(model, train_dl, train_ds, model_children, regular_param, optimizer, RHO, l1,lr):
    print('Training')
    model.train()
    running_loss = 0.0
    counter = 0
    mse_loss_fit = 0.0
    regularizer_loss_fit = 0.0
    for i, data in tqdm(enumerate(train_dl), total=int(len(train_ds) / train_dl.batch_size)):
        counter += 1
        x, _ = data
        optimizer.zero_grad()
        reconstructions = model(x)
        loss, mse_loss1, l1_loss1 = loss_functions.old_loss(model_children=model_children, true_data=x, reconstructed_data=reconstructions,reg_param=regular_param)
        #loss, mse_loss1, l1_loss1 = loss_functions.l1_loss_func(model=model,lr=lr,reconstructed_data=reconstructions,true_data=x)
        loss.backward() # Computes the Gradient of the loss tensor
        optimizer.step()
        mse_loss_fit += mse_loss1.item()
        regularizer_loss_fit += l1_loss1.item()
        running_loss += loss.item() ## Running loss is not epoch loss.
    epoch_loss = running_loss / len(train_ds)
    print(f" Train Loss: {epoch_loss:.6f}")
    # save the reconstructed images every 5 epochs
    return epoch_loss,mse_loss_fit,regularizer_loss_fit, model

def validate(model, test_dl, test_ds, model_children,reg_param,l1,lr):
    print('Validating')
    model.eval()
    running_loss = 0.0
    counter = 0
    mse_loss_val = 0
    regularizer_loss_val = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dl), total=int(len(test_ds) / test_dl.batch_size)):
            counter += 1
            x, _ = data
            reconstructions = model(x)
            loss, mse_loss1, l1_loss1 = loss_functions.old_loss(model_children=model_children, true_data=x, reconstructed_data=reconstructions, reg_param=reg_param)
            #loss, mse_loss1, l1_loss1 = loss_functions.l1_loss_func(model=model,lr=lr,reconstructed_data=reconstructions,true_data=x)

            running_loss += loss.item()
            mse_loss_val += mse_loss1.item()
            regularizer_loss_val += l1_loss1.item()
    epoch_loss = running_loss / len(test_ds)
    print(f" Val Loss: {epoch_loss:.6f}")
    # save the reconstructed images every 5 epochs
    return epoch_loss, mse_loss_val, regularizer_loss_val

def train(model,variables, train_data, test_data, parent_path, config):
    learning_rate = config["lr"]
    bs = config["batch_size"]
    reg_param = config["reg_param"]
    RHO = config["RHO"]
    l1 = config["l1"]
    epochs = config["epochs"]
    latent_space_size = config["latent_space_size"]
    
    model_children = list(model.children())

    # Constructs a tensor object of the data and wraps them in a TensorDataset object.
    train_ds = TensorDataset(torch.tensor(train_data.values, dtype=torch.float64),
                             torch.tensor(train_data.values, dtype=torch.float64))
    valid_ds = TensorDataset(torch.tensor(test_data.values, dtype=torch.float64),
                             torch.tensor(test_data.values, dtype=torch.float64))

    # Converts the TensorDataset into a DataLoader object and combines into one DataLoaders object (a basic wrapper
    # around several DataLoader objects).
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs) ## Used to be batch_size = bs * 2

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.RMSprop(model.parameters(),lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

    # train and validate the autoencoder neural network
    train_loss = []
    val_loss = []
    mse_loss_val = []
    regularizer_loss_val = []
    mse_loss_fit = []
    regularizer_loss_fit = []
    start = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss,mse_loss_fit1, regularizer_loss_fit1, model_from_fit = fit(model=model, train_dl=train_dl, train_ds=train_ds, model_children=model_children,
                                optimizer=optimizer, RHO=RHO, regular_param=reg_param, l1=l1,lr=learning_rate)

        val_epoch_loss, mse_loss_val1, regularizer_loss_val1 = validate(model=model_from_fit, test_dl=valid_dl, test_ds=valid_ds, model_children=model_children,
                                reg_param=reg_param,l1=l1,lr=learning_rate)

        scheduler.step()
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        mse_loss_val.append(mse_loss_val1)
        mse_loss_fit.append(mse_loss_fit1)
        regularizer_loss_val.append(regularizer_loss_val1)
        regularizer_loss_fit.append(regularizer_loss_fit1)
    end = time.time()
    regularizer_string = 'l1'

    print(f"{(end - start) / 60:.3} minutes")
    pd.DataFrame({'Train Loss': train_loss,
                  'Val Loss': val_loss,
                  'mse_loss_val':mse_loss_val,
                  'mse_loss_fit':mse_loss_fit,
                  regularizer_string+'_loss_val':regularizer_loss_val,
                  regularizer_string+'_loss_fit':regularizer_loss_fit}).to_csv(parent_path+'loss_data_lrdecay.csv')
    #pd.DataFrame({'Values_val':values_val}).to_csv(parent_path + 'values_val.csv')
    #pd.DataFrame({'Values_fit':values_fit}).to_csv(parent_path + 'values_fit.csv')

    data_as_tensor = torch.tensor(test_data.values, dtype=torch.float64)
    pred_as_tensor = model(data_as_tensor)

    return data_as_tensor, pred_as_tensor, model_from_fit
