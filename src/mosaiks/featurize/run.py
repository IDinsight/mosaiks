import geopandas as gpd
import numpy as np
import pandas as pd
import torch

__all__ = ["create_features", "make_result_df"]


def create_features(
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
    dataloader: A dataloader object that yields batches of images.
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


def make_result_df(
    features: np.ndarray,
    mosaiks_col_names: list[str],
    context_gdf: gpd.GeoDataFrame,
    selected_context_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Takes the features array and a context dataframe and returns a dataframe with the
    features and chosen context columns, with the same index as the context dataframe.

    Parameters
    -----------
    features : Array of features.
    mosaiks_col_names : List of column names to label the feature columns as.
    context_gdf : GeoDataFrame of context variables. Must have the same index size as
        the features array.
    selected_context_cols : List of context columns to include in final dataframe
        (optional). If not given, no context columns will be included.

    Returns
    --------
    DataFrame
    """

    features_df = pd.DataFrame(
        data=features, index=context_gdf.index, columns=mosaiks_col_names
    )
    if selected_context_cols is None:
        return features_df
    else:
        context_gdf = context_gdf[selected_context_cols]
        return pd.concat([context_gdf, features_df], axis=1)
