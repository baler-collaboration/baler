import torch
from torch.utils.data import DataLoader, TensorDataset

from baler.modules.utils import (
    mse_loss_emd_l1,
    mse_loss_l1,
    accuracy,
    LRScheduler,
)


def test_mse_loss_emd_l1():
    # Generate random data
    batch_size, num_features = 32, 64
    true_data = torch.randn(batch_size, num_features)
    reconstructed_data = torch.randn(batch_size, num_features)

    # Create model children list
    model_children = [torch.nn.Linear(num_features, num_features)]

    # Compute loss with L1 regularization
    reg_param = 0.1
    loss, emd_loss, l1_loss = mse_loss_emd_l1(
        model_children, true_data, reconstructed_data, reg_param, validate=False
    )
    assert isinstance(loss, torch.Tensor)
    assert isinstance(emd_loss, float)
    assert isinstance(l1_loss, torch.Tensor)

    # Compute loss without L1 regularization
    loss = mse_loss_emd_l1(
        model_children, true_data, reconstructed_data, 0.0, validate=False
    )[0]
    assert isinstance(loss, torch.Tensor)

    # Compute EMD loss only
    emd_loss = mse_loss_emd_l1(
        model_children, true_data, reconstructed_data, 0.0, validate=True
    )
    assert isinstance(emd_loss, float)


def test_mse_loss_l1():
    model_children = [torch.nn.Linear(10, 10)]
    true_data = torch.randn(32, 10)
    reconstructed_data = torch.randn(32, 10)
    reg_param = 0.1
    validate = False

    loss, mse_loss, l1_loss = mse_loss_l1(
        model_children, true_data, reconstructed_data, reg_param, validate
    )

    assert isinstance(loss, torch.Tensor)
    assert isinstance(mse_loss, torch.Tensor)
    assert isinstance(l1_loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert mse_loss.shape == torch.Size([])
    assert l1_loss.shape == torch.Size([])

    validate = True
    mse_loss, _, _ = mse_loss_l1(
        model_children, true_data, reconstructed_data, reg_param, validate
    )

    assert isinstance(mse_loss, torch.Tensor)
    assert mse_loss.shape == torch.Size([])


def test_accuracy():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 2)
    )
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=10)
    accuracy_frac = accuracy(model, dataloader)
    assert isinstance(accuracy_frac, float)
    assert 0 <= accuracy_frac <= 1


def test_lr_scheduler():
    # Create a dummy model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Create a learning rate scheduler
    lr_scheduler = LRScheduler(optimizer, patience=2, min_lr=1e-5, factor=0.5)

    # Test that the learning rate is not reduced when the loss is improving
    train_losses = [10.0, 9.0, 8.0, 7.0]
    for train_loss in train_losses:
        lr_scheduler(train_loss)
        assert optimizer.param_groups[0]["lr"] == 0.1

    # Test that the learning rate is reduced when the loss is not improving
    train_losses = [10.0, 9.0, 10.0, 11.0, 12.0]
    for i, train_loss in enumerate(train_losses):
        lr_scheduler(train_loss)
        if i >= 2:
            assert optimizer.param_groups[0]["lr"] == 0.05

    # Test that the learning rate does not go below the minimum value
    train_losses = [10.0] * 100
    for train_loss in train_losses:
        lr_scheduler(train_loss)
    assert optimizer.param_groups[0]["lr"] == 1e-5
