import torch
import torch.nn as nn
import torch.nn.functional as F

# Disabling the benchmarking feature with torch.backends.cudnn.benchmark = False
# causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)


def featurize(image, model, device):
    """Helper method for running an image patch through a model.

    Args:
        input_img (np.ndarray): Image in (C x H x W) format with a dtype of uint8.
        model (torch.nn.Module): Feature extractor network
    """
    image = image.to(device)
    with torch.no_grad():
        feats = model(image).cpu().numpy()  # .unsqueeze(0)
    return feats


class RCF(nn.Module):
    """A model for extracting Random Convolution Features (RCF) from input imagery."""

    def __init__(self, num_features=16, kernel_size=3, num_input_channels=6):
        super(RCF, self).__init__()
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
        # The rectified linear activation function or ReLU for short is a piecewise linear function
        # that will output the input directly if it is positive, otherwise, it will output zero.
        x1a = F.relu(self.conv1(x), inplace=True)
        # The below step is where we take the inverse which is appended later
        x1b = F.relu(-self.conv1(x), inplace=True)
        # Applies a 2D adaptive average pooling over an input signal composed of several input planes.
        x1a = F.adaptive_avg_pool2d(x1a, (1, 1)).squeeze()
        x1b = F.adaptive_avg_pool2d(x1b, (1, 1)).squeeze()
        if len(x1a.shape) == 1:  # case where we passed a single input
            return torch.cat((x1a, x1b), dim=0)
        elif len(x1a.shape) == 2:  # case where we passed a batch of > 1 inputs
            return torch.cat((x1a, x1b), dim=1)
