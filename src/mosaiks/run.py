import geopandas as gpd
import numpy as np
import pandas as pd
import torch.nn as nn

import mosaiks.utils as utl
from mosaiks.featurize import create_features, make_result_df
from mosaiks.fetch import create_data_loader, fetch_image_refs

__all__ = ["full_pipeline"]


def full_pipeline(
    points_gdf: gpd.GeoDataFrame,
    model: nn.Module,
    featurization_config: dict,
    satellite_config: dict,
    col_names: list,
    save_folder_path: str,
    save_filename: str,
    return_df: bool = False,
    selected_context_cols: list[str] = ["Lat", "Lon", "stac_item"],
) -> None:  # or DataFrame...
    """
    For a given GeoDataFrame of coordinate points, this function runs the necessary
    functions and saves resulting mosaiks features to file. No Dask is necessary.

    Parameters
    -----------
    points_gdf : GeoDataFrame of points to be featurized.
    model : PyTorch model to be used for featurization.
    featurization_config : Dictionary of featurization parameters.
    satellite_config : Dictionary of satellite parameters.
    col_names : List of column names to be used for saving the features.
    save_folder_path : Path to folder where features will be saved.
    save_filename : Name of file where features will be saved.
    return_df : Whether to return the features as a DataFrame.
    selected_context_cols : List of context columns to include in final dataframe
        (optional). If not given, no context columns will be included.

    Returns
    --------
    None or DataFrame
    """

    satellite_search_params = featurization_config["satellite_search_params"]

    points_gdf_with_stac = fetch_image_refs(points_gdf, satellite_search_params)

    data_loader = create_data_loader(
        points_gdf_with_stac=points_gdf_with_stac,
        satellite_params=satellite_config,
        batch_size=featurization_config["model"]["batch_size"],
    )

    X_features = create_features(
        dataloader=data_loader,
        n_features=featurization_config["model"]["num_features"],
        model=model,
        device=featurization_config["model"]["device"],
        min_image_edge=satellite_config["min_image_edge"],
    )

    df = make_result_df(
        features=X_features,
        mosaiks_col_names=col_names,
        context_gdf=points_gdf_with_stac,
        selected_context_cols=selected_context_cols,
    )

    if save_folder_path is not None:
        utl.save_dataframe(df=df, file_path=save_folder_path / save_filename)

    if return_df:
        return df
