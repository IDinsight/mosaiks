import pandas as pd

import mosaiks.utils as utl
from mosaiks.pipeline import get_features


def load_and_save_features(
    input_file_path: str,
    path_to_save_data: str,
    lat_col: str = "Lat",
    lon_col: str = "Lon",
    context_cols_to_keep_from_input: list = None,
    **kwargs,
) -> None:
    """
    Load and save features

    Parameters
    ----------
    input_file_path : Path to lat-lons data file, in either .csv or .parquet format.
    path_to_save_data : Path to save data, in either .csv or .parquet format.
    lat_col : Name of latitude column in input data, default is "Lat".
    lon_col : Name of longitude column in input data, default is "Lon".
    context_cols_to_keep_from_input : List of context columns to add to final dataframe from input data
    kwargs: config parameters for `get_features`. See `get_features` docstring for more details and default values.
    """
    # Load data
    points_df = utl.load_dataframe(input_file_path)

    # Get features
    features_df = get_features(points_df[lat_col], points_df[lon_col], **kwargs)

    # Add context columns
    if context_cols_to_keep_from_input:
        context_df = points_df[context_cols_to_keep_from_input]
        features_df = pd.concat([context_df, features_df], axis=1)

    # Save data
    utl.save_dataframe(features_df, path_to_save_data)
