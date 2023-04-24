import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 41

# Disabling the benchmarking feature causes cuDNN to deterministically
# select an algorithm, possibly at the cost of reduced performance.
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)

__all__ = ["create_features"]


def create_features(
    dataloader: torch.utils.data.DataLoader,
    n_features: int,
    n_points: int,
    model: nn.Module,
    device: str,
    min_image_edge: str,
) -> np.ndarray:
    """
    Parameters:
    -----------
    dataloader: A dataloader object that yields batches of images.
    n_features: The number of features to extract from each image.
    n_points: The number of images to extract features from.
    model: A model that extracts features from images.
    device: The device to run the model on.
    min_image_edge: The minimum edge length of an image to extract features from.

    Returns:
    --------
    features_array: An array of shape (n_points, n_features) containing the extracted features.

    """

    features_array = np.full([n_points, n_features], np.nan, dtype=float)
    model.eval().to(device)

    i = -1
    for images in dataloader:
        for i, image in enumerate(images, start=i + 1):
            if image is not None:

                # in case of single-band image, force-add a band dimension
                if len(image.shape) == 2:
                    image = image.unsqueeze(0)

                if (image.shape[1] >= min_image_edge) and (
                    image.shape[2] >= min_image_edge
                ):
                    features_array[i] = featurize(image, model, device)
                else:
                    # print("warn", flush=True)
                    # warnings.warn("Image crop too small")
                    pass
            else:
                # print("warn", flush=True)
                # warnings.warn("No image found")
                pass

    return features_array


def featurize(image: torch.Tensor, model: nn.Module, device: str):
    """Helper method for running an image patch through a model.

    Parameters:
    -----------
    image: A tensor of shape (BANDS, X, Y) containing an image patch.
    model: A model that extracts features from images.
    device: The device to run the model on.

    Returns:
    --------
    feats: np.ndarray
        An array of shape (1, n_features) containing the extracted features.
    """
    image = image.to(device)
    # TODO: this causes torch NOT to save the computational graph.
    # Might need to change if we want to attribution analysis / add trainable layers.
    with torch.no_grad():
        feats = model(image).cpu().unsqueeze(0).numpy()
    return feats


class RCF(nn.Module):
    """
    A model for extracting Random Convolution Features (RCF) from input imagery.

    Parameters:
    -----------
    num_features: The number of features to extract from each image.
        NB: this should be an even number, since the features are produced by
        generating random filters, and concatenating the positive and negative
        convolutions from the filters.
    kernel_size: The size of the convolutional kernel.
    num_input_channels: The number of bands in the satellite image.

    """

    def __init__(
        self,
        num_features: int = 1000,
        kernel_size: int = 3,
        num_input_channels: int = 6,
    ):
        super().__init__()
        # We create `num_features / 2` filters so require `num_features` to be
        # divisible by 2
        assert num_features % 2 == 0, "Please enter an even number of features."
        # Applies a 2D convolution over an input image composed of several input planes.
        self.conv1 = nn.Conv2d(
            num_input_channels,
            num_features // 2,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
        )
        # Fills the input Tensor 'conv1.weight' with values drawn from the
        # normal distribution
        nn.init.normal_(self.conv1.weight, mean=0.0, std=1.0)
        # Fills the input Tensor 'conv1.bias' with the value 'val = -1'.
        nn.init.constant_(self.conv1.bias, -1.0)

        # Explicitly freeze convolutional weights and bias
        self.conv1.weight.requires_grad_(False)
        self.conv1.bias.requires_grad_(False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Parameters:
        -----------
        x: A tensor of shape (BANDS, X, Y) containing an image patch.
        """
        assert torch.all(
            x.shape[1:] > self.conv1.kernel_size
        ), "Image too small for kernel size"
        x1a = F.relu(self.conv1(x))
        x1b = F.relu(-self.conv1(x))

        x1a = F.adaptive_avg_pool2d(x1a, (1, 1)).squeeze()
        x1b = F.adaptive_avg_pool2d(x1b, (1, 1)).squeeze()

        if len(x1a.shape) == 1:
            return torch.cat((x1a, x1b), dim=0)
        elif len(x1a.shape) >= 2:
            return torch.cat((x1a, x1b), dim=1)
