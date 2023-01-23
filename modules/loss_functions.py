import torch
import torch.nn as nn
import torch.nn.functional as F


def l1_loss_func(model,lr,reconstructed_data,true_data):
    mse = nn.MSELoss()
    mse_loss = mse(reconstructed_data, true_data)

    l1_lambda = lr
    l1_norm = sum(torch.sum(torch.abs(p)) for p in model.parameters())
    l1_loss = l1_lambda * l1_norm
    loss = mse_loss + l1_loss
    return loss, mse_loss, l1_loss



def old_loss(self, model_children, true_data, reconstructed_data, reg_param):
    mse = nn.MSELoss()
    mse_loss = mse(reconstructed_data, true_data)
    l1_loss = 0
    values = true_data
    for i in range(len(model_children)):
        values = F.relu((model_children[i](values)))
        l1_loss += torch.mean(torch.abs(values))
    loss = mse_loss + reg_param * l1_loss
    return loss, mse_loss, l1_loss