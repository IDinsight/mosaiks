import torch
import torch.nn as nn
import torch.nn.functional as F

# Disabling the benchmarking feature causes cuDNN to deterministically
# select an algorithm, possibly at the cost of reduced performance.
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = False


__all__ = ["RCF"]


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
    random_seed_for_filters: The random seed to use when generating the filters.
        Defaults to 768.

    """

    def __init__(
        self,
        num_features: int = 1000,
        kernel_size: int = 3,
        num_input_channels: int = 6,
        random_seed_for_filters: int = 768,
    ):
        super().__init__()
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

        # Set random state to initialise filters same way every time
        torch.manual_seed(random_seed_for_filters)

        # Fill the input Tensor with random gaussian noise and bias of -1
        nn.init.normal_(self.conv1.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.conv1.bias, -1.0)

        # Explicitly freeze convolutional weights and bias
        self.conv1.weight.requires_grad_(False)
        self.conv1.bias.requires_grad_(False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Parameters
        -----------
        x: A tensor of shape (BANDS, X, Y) containing an image patch.

        Returns
        --------
        A tensor containing the features.
        """

        assert x.shape[1:] > self.conv1.kernel_size, "Image too small for kernel size"
        x1a = F.relu(self.conv1(x))
        x1b = F.relu(-self.conv1(x))

        x1a = F.adaptive_avg_pool2d(x1a, (1, 1)).squeeze()
        x1b = F.adaptive_avg_pool2d(x1b, (1, 1)).squeeze()

        if len(x1a.shape) == 1:
            return torch.cat((x1a, x1b), dim=0)
        elif len(x1a.shape) >= 2:
            return torch.cat((x1a, x1b), dim=1)
