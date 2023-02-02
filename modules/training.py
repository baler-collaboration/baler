import torch
from torch.utils.data import DataLoader, TensorDataset
import modules.utils as utils
from tqdm import tqdm
import time
import pandas as pd

def fit(model, train_dl, train_ds, regular_param, optimizer, RHO, l1=True):
    print('Training')
    model.train()
    running_loss = 0.0
    counter = 0
    mse_loss_fit = 0.0
    regularizer_loss_fit = 0.0
    model_children = list(model.children())

    for data in tqdm(train_dl):

        counter += 1
        x, _ = data
        reconstructions = model(x)
        optimizer.zero_grad()

        if l1 == True:
            loss, mse_loss, l1_loss = utils.sparse_loss_function_L1(model_children=model_children, true_data=x,reg_param=regular_param,
                                                                    reconstructed_data=reconstructions, validate=False)
        else:
            # Implement KL here
            break
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        regularizer_loss_fit += l1_loss.item()
        mse_loss_fit += mse_loss.item()



    epoch_loss = running_loss/len(train_ds)
    print(f" Train Loss: {epoch_loss:.5E}")
    return epoch_loss,mse_loss_fit,regularizer_loss_fit, model

def validate(model, test_dl, test_ds,regular_param,l1=True):
    print('Validating')
    model.eval()
    running_loss = 0.0
    counter = 0
    model_children = list(model.children())

    with torch.no_grad():
        for data in tqdm(test_dl):

            counter += 1
            x, _ = data
            reconstructions = model(x)
            if l1 == True:
                loss, mse, l1_loss = utils.sparse_loss_function_L1(model_children=model_children, true_data=x,reg_param=regular_param,
                                                    reconstructed_data=reconstructions, validate=False)
                running_loss += loss.item()


            ## WIP
            else:
                # Implement KL Here also. Should however just return mse loss I think
                break

    epoch_loss = running_loss/len(test_ds)
    print(f" Val Loss: {epoch_loss:.5E}")
    return epoch_loss

def train(model,number_of_columns, train_data, test_data, parent_path, config):
    learning_rate = config["lr"]
    bs = config["batch_size"]
    reg_param = config["reg_param"]
    RHO = config["RHO"]
    l1 = config["l1"]
    epochs = config["epochs"]
    
    # Constructs a tensor object of the data and wraps them in a TensorDataset object.
    train_ds = TensorDataset(torch.tensor(train_data.values, dtype=torch.float64),
                             torch.tensor(train_data.values, dtype=torch.float64))
    valid_ds = TensorDataset(torch.tensor(test_data.values, dtype=torch.float64),
                             torch.tensor(test_data.values, dtype=torch.float64))

    # Converts the TensorDataset into a DataLoader object and combines into one DataLoaders object (a basic wrapper
    # around several DataLoader objects).
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs) ## Used to be batch_size = bs * 2


    ## Select Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ## Activate early stopping
    if config['early_stopping'] == True:
        early_stopping = utils.EarlyStopping(patience=config['patience'],min_delta=config['min_delta']) # Changes to patience & min_delta can be made in configs

    ## Activate LR Scheduler
    if config['lr_scheduler'] == True:
        lr_scheduler = utils.LRScheduler(optimizer=optimizer,patience=config['patience'])

    # train and validate the autoencoder neural network
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    mse_loss_fit = []
    regularizer_loss_fit = []
    start = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss,mse_loss_fit1,regularizer_loss_fit1, model_from_fit = fit(model=model, train_dl=train_dl, train_ds=train_ds,
                                                                                    optimizer=optimizer, RHO=RHO, regular_param=reg_param, l1=l1)

        val_epoch_loss = validate(model=model_from_fit, test_dl=valid_dl,
                                test_ds=valid_ds, regular_param=reg_param, l1 = l1)

        #accuracy_train = utils.accuracy(model,train_dl)
        #accuracy_val = utils.accuracy(model,valid_dl)

        #train_acc.append(accuracy_train)
        #val_acc.append(accuracy_val)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        mse_loss_fit.append(mse_loss_fit1)
        regularizer_loss_fit.append(regularizer_loss_fit1)
        if config['lr_scheduler'] == True:
            lr_scheduler(val_epoch_loss)
        if config['early_stopping'] == True:
            early_stopping(val_epoch_loss)
            if early_stopping.early_stop:
                break

    end = time.time()
    regularizer_string = 'l1'

    print(f"{(end - start) / 60:.3} minutes")
    pd.DataFrame({'Train Loss': train_loss,
                  'Val Loss': val_loss,
                  #'Val Acc.': val_acc,
                  #'Train Acc.': train_acc,
                  'mse_loss_fit':mse_loss_fit,
                  regularizer_string+'_loss_fit':regularizer_loss_fit}).to_csv(parent_path+'loss_data.csv')
    #pd.DataFrame({'Values_val':values_val}).to_csv(parent_path + 'values_val.csv')
    #pd.DataFrame({'Values_fit':values_fit}).to_csv(parent_path + 'values_fit.csv')

    data_as_tensor = torch.tensor(test_data.values, dtype=torch.float64)
    pred_as_tensor = model(data_as_tensor)

    return data_as_tensor, pred_as_tensor, model_from_fit
