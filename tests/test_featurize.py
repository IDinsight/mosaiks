"""Tests for featurization functions."""
import pytest
import torch

from mosaiks.featurize import RCF


def test_random_conv_features():
    """Test random convolutional features."""
    model = RCF(num_features=666, kernel_size=5, num_input_channels=3)

    x1 = torch.empty(3, 10, 10)
    x2 = torch.ones(3, 10, 10) * torch.nan
    out1 = model(x1)
    out2 = model(x2)

    assert hasattr(model, "conv1")
    assert (
        model.conv1.weight.requires_grad == False
        and model.conv1.bias.requires_grad == False
    )
    assert out1.dim() == 1 and out2.dim() == 1
    assert out2.shape[0] == 666 and out2.shape[0] == 666
