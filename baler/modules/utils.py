import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import wasserstein_distance

###############################################
factor = 0.5
min_lr = 1e-6


###############################################
def sparse_loss_function_EMD_L1(
    model_children, true_data, reconstructed_data, reg_param, validate
):
    mse = nn.MSELoss()
    mse_loss = mse(reconstructed_data, true_data)
    Wassterstein_distance_list = [
        wasserstein_distance(
            true_data.detach().numpy()[i, :], reconstructed_data.detach().numpy()[i, :]
        )
        for i in range(len(true_data))
    ]
    emd_loss = sum(Wassterstein_distance_list)

    l1_loss = 0
    values = true_data
    if validate == False:
        for i in range(len(model_children)):
            values = model_children[i](values)
            l1_loss += torch.mean(torch.abs(values))

        loss = emd_loss + mse_loss + reg_param * l1_loss
        return loss, emd_loss, l1_loss
    else:
        return emd_loss


def sparse_loss_function_L1(
    model_children, true_data, reconstructed_data, reg_param, validate
):
    mse = nn.MSELoss()
    mse_loss = mse(reconstructed_data, true_data)

    l1_loss = 0
    values = true_data
    if not validate:
        for i in range(len(model_children)):
            values = model_children[i](values)
            l1_loss += torch.mean(torch.abs(values))

        loss = mse_loss + reg_param * l1_loss
        return loss, mse_loss, l1_loss
    else:
        return mse_loss


# Accuracy function still WIP. Not working properly.
# Probably has to do with total_correct counter.


def accuracy(model, dataloader):
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
    def __init__(self, patience, min_delta):
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
            self.counter = 0  ## Resets if val_loss improves

        elif self.best_loss - train_loss < self.min_delta:
            self.counter += 1

            print(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("Early Stopping")
                self.early_stop = True


class LRScheduler:
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
