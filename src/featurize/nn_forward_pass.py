import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

SEED = 41

# Disabling the benchmarking feature causes cuDNN to deterministically
# select an algorithm, possibly at the cost of reduced performance.
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)

__all__ = ["create_features"]


def create_features(
    dataloader, n_features, n_points, model, torch_device, min_image_edge
):

    features_array = np.full([n_points, n_features], np.nan, dtype=float)
    # torch_device = torch.device(device)

    i = -1
    for images in tqdm(dataloader):
        for i, image in tqdm(enumerate(images, start=i + 1), leave=False):
            if image is not None:
                if (image.shape[1] >= min_image_edge) and (
                    image.shape[2] >= min_image_edge
                ):
                    features_array[i] = featurize(image, model, torch_device)
            else:
                print("warn", flush=True)
                warnings.warn("No image found")

    return features_array


def featurize(image, model, device):
    """Helper method for running an image patch through a model.

    Args:
        input_img (np.ndarray): Image in (C x H x W) format with a dtype of uint8.
        model (torch.nn.Module): Feature extractor network
    """
    image = image.to(device)
    with torch.no_grad():
        feats = model(image).cpu().unsqueeze(0).numpy()
    return feats


class RCF(nn.Module):
    """A model for extracting Random Convolution Features (RCF) from input imagery."""

    def __init__(self, num_features=1000, kernel_size=3, num_input_channels=6):
        super().__init__()
        # We create `num_features / 2` filters so require `num_features` to be divisible by 2
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
        # Fills the input Tensor 'conv1.weight' with values drawn from the normal distribution
        nn.init.normal_(self.conv1.weight, mean=0.0, std=1.0)
        # Fills the input Tensor 'conv1.bias' with the value 'val = -1'.
        nn.init.constant_(self.conv1.bias, -1.0)

    def forward(self, x):
        x1a = F.relu(self.conv1(x), inplace=True)
        x1b = F.relu(-self.conv1(x), inplace=True)

        x1a = F.adaptive_avg_pool2d(x1a, (1, 1)).squeeze()
        x1b = F.adaptive_avg_pool2d(x1b, (1, 1)).squeeze()

        if len(x1a.shape) == 1:
            return torch.cat((x1a, x1b), dim=0)
        elif len(x1a.shape) >= 2:
            return torch.cat((x1a, x1b), dim=1)
