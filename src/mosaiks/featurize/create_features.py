import numpy as np
import torch

__all__ = ["create_features_from_image_array"]


def create_features_from_image_array(
    dataloader: torch.utils.data.DataLoader,
    n_features: int,
    model: torch.nn.Module,
    device: str,
    min_image_edge: str,
) -> np.ndarray:
    """
    Create features for images fetched through a dataloader.

    Parameters:
    -----------
    dataloader: A dataloader object that lazily yields images.
    n_features: The number of features to extract from each image.
    model: A model that extracts features from images.
    device: The device to run the model on.
    min_image_edge: The minimum edge length of an image to extract features from.

    Returns:
    --------
    features_array: An array with n_features columns.
    """

    n_points = len(dataloader.dataset)
    features_array = np.full([n_points, n_features], np.nan, dtype=float)
    model.eval().to(device)

    for i, image in enumerate(dataloader):
        if image is not None:
            # in case of single-band image, force-add a band dimension
            if len(image.shape) == 2:
                image = image.unsqueeze(0)

            if (image.shape[1] >= min_image_edge) and (
                image.shape[2] >= min_image_edge
            ):
                features_array[i] = featurize(image, model, device)
            else:
                # logging.warn("Image crop too small")
                pass
        else:
            # logging.warn("No image found")
            pass
    return features_array


def featurize(image: torch.Tensor, model: torch.nn.Module, device: str):
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
    feats = model(image).cpu().unsqueeze(0).numpy()
    return feats
