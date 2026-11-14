import pytest
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
    """LR is reduced by `factor` only after more than `patience` consecutive
    non-improving steps, and never below `min_lr`."""
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    patience, factor, min_lr = 2, 0.5, 1e-5
    lr_scheduler = LRScheduler(optimizer, patience=patience, min_lr=min_lr, factor=factor)

    def lr():
        return optimizer.param_groups[0]["lr"]

    # Improving loss: no reduction
    for train_loss in [10.0, 9.0, 8.0, 7.0]:
        lr_scheduler(train_loss)
        assert lr() == pytest.approx(0.1)

    # `patience` non-improving steps are tolerated...
    for train_loss in [8.0] * patience:
        lr_scheduler(train_loss)
        assert lr() == pytest.approx(0.1)

    # ...the next one reduces the LR by exactly `factor`
    lr_scheduler(8.0)
    assert lr() == pytest.approx(0.1 * factor)

    # The counter restarts after a reduction: another `patience` bad steps
    # keep the LR, the one after reduces it again
    for train_loss in [8.0] * patience:
        lr_scheduler(train_loss)
        assert lr() == pytest.approx(0.1 * factor)
    lr_scheduler(8.0)
    assert lr() == pytest.approx(0.1 * factor**2)

    # The LR never goes below `min_lr`
    for train_loss in [10.0] * 100:
        lr_scheduler(train_loss)
    assert lr() == pytest.approx(min_lr)


def test_lr_scheduler_reports_reduction(capsys):
    optimizer = torch.optim.SGD(torch.nn.Linear(2, 1).parameters(), lr=0.1)
    lr_scheduler = LRScheduler(optimizer, patience=0, min_lr=1e-5, factor=0.5)
    lr_scheduler(1.0)
    assert capsys.readouterr().out == ""
    lr_scheduler(2.0)
    assert "reducing learning rate" in capsys.readouterr().out
