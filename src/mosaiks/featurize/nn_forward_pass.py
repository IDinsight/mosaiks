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


def create_features(dataloader, n_features, n_points, model, device, min_image_edge):
    """
    Parameters:
    -----------
    dataloader: torch.utils.data.DataLoader
        A dataloader object that yields batches of images.
    n_features: int
        The number of features to extract from each image.
    n_points: int
        The number of images to extract features from.
    model: torch.nn.Module
        A model that extracts features from images.
    device: str
        The device to run the model on.
    min_image_edge: int
        The minimum edge length of an image to extract features from.

    Returns:
    --------
    features_array: np.ndarray
        An array of shape (n_points, n_features) containing the extracted features.

    """

    features_array = np.full([n_points, n_features], np.nan, dtype=float)
    torch_device = torch.device(device)
    model.eval().to(device)

    i = -1
    for images in dataloader:
        for i, image in enumerate(images, start=i + 1):
            if image is not None:
                if (image.shape[1] >= min_image_edge) and (
                    image.shape[2] >= min_image_edge
                ):
                    # image = normalize(image) # or
                    # image = image / 255
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


def normalize(image):
    
    img_min, img_max = image.min(), image.max()
    return (image - img_min) / (img_max - img_min)


def featurize(image, model, device):
    """Helper method for running an image patch through a model.

    Parameters:
    -----------
    image: torch.Tensor
        A tensor of shape (BANDS, X, Y) containing an image patch.
    model: torch.nn.Module
        A model that extracts features from images.
    device: str
        The device to run the model on.

    Returns:
    --------
    feats: np.ndarray
        An array of shape (1, n_features) containing the extracted features.
    """
    image = image.to(device)

    with torch.no_grad():
        feats = model(image).cpu().unsqueeze(0).numpy()
    return feats


class RCF(nn.Module):
    """
    A model for extracting Random Convolution Features (RCF) from input imagery.

    Parameters:
    -----------
    num_features: int
        The number of features to extract from each image.
    kernel_size: int
        The size of the convolutional kernel.
    num_input_channels: int
        The number of bands in the satellite image.

    """

    def __init__(self, num_features=1000, kernel_size=3, num_input_channels=6):
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

    def forward(self, x):
        """
        Parameters:
        -----------
        x: torch.Tensor
            A tensor of shape (BANDS, X, Y) containing an image patch.
        """
        x1a = F.relu(self.conv1(x))
        x1b = F.relu(-self.conv1(x))

        x1a = F.adaptive_avg_pool2d(x1a, (1, 1)).squeeze()
        x1b = F.adaptive_avg_pool2d(x1b, (1, 1)).squeeze()

        if len(x1a.shape) == 1:
            return torch.cat((x1a, x1b), dim=0)
        elif len(x1a.shape) >= 2:
            return torch.cat((x1a, x1b), dim=1)
