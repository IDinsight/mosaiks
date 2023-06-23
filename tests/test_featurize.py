"""Tests for featurization functions."""
import os

import pytest
import torch

from mosaiks.featurize import RCF

os.environ["USE_PYGEOS"] = "0"


@pytest.fixture(scope="module")
def model():
    """Fixture for a random convolutional features model."""
    return RCF(num_features=666, kernel_size=5, num_input_channels=3)


def test_random_model_conv_weights_are_created(model):
    assert hasattr(model, "conv1")


def test_random_model_conv_weights_are_not_trainable(model):
    assert (
        model.conv1.weight.requires_grad == False
        and model.conv1.bias.requires_grad == False
    )


def test_random_conv_features_are_created(model):
    x1 = torch.empty(3, 10, 10)
    out1 = model(x1)
    assert out1.dim() == 1 and out1.shape[0] == 666


def test_random_conv_features_are_created_with_nans(model):
    x2 = torch.ones(3, 10, 10) * torch.nan
    out2 = model(x2)
    assert out2.dim() == 1 and out2.shape[0] == 666
