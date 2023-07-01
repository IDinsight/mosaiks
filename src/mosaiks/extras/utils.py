from pathlib import Path

import geopandas as gpd
import pandas as pd

import mosaiks.utils as utl

_all_ = [
    "combine_results_df_with_context_df",
    "get_dataset_path",
    "save_geodataframe_as_dataframe",
]


def combine_results_df_with_context_df(
    features_df: pd.DataFrame,
    context_df: pd.DataFrame,
    context_cols_to_keep: list[str] = None,
) -> pd.DataFrame:
    """
    Takes the features array and a context dataframe and returns a dataframe with the
    features, the stac_id of the images used to create each row, and chosen context
    columns.

    Parameters
    -----------
    features : Array of features.
    context_df : DataFrame of context variables. Must have the same index size as
        the features array.
    context_cols_to_keep : List of context columns to include in final dataframe
        (optional). If not given, only "stac_id" will be included.

    Returns
    --------
    DataFrame
    """
    context_df = context_df[context_cols_to_keep]
    return pd.concat([context_df, features_df], axis=1)


def get_dataset_path(
    folder: str, filename: str, relative_path_is_root_folder: bool = True
) -> str:
    """
    Get data path in the form `relative path/folder/filename`.

    Parameters
    ----------
    folder : folder name.
    filename : file name.
    relative_path_is_root_folder: If True, keep the relative path fixed to root folder
        i.e. `repo root folder/data/folder/filename`. If False, path will be relative
        to the current working directory.

    Returns
    -------
    file_path : The path to the file.
    """
    if relative_path_is_root_folder:
        path = Path(__file__).resolve().parents[3] / "data" / folder / filename
    else:
        path = Path().resolve() / folder / filename

    return str(path)


def save_geodataframe_as_dataframe(
    gdf: gpd.GeoDataFrame,
    file_path: str,
    add_latlon_cols: bool = False,
    **kwargs,
) -> None:
    """
    Save GeoDataFrame as a DataFrame by dropping geometry column.
    Either dataset_name or file_path must be given.

    Parameters
    ----------
    gdf : The GeoDataFrame to save.
    file_path : The path to the file to load.
    add_latlon_cols : If True, add columns for latitude and longitude.
    """
    df = gdf.drop(columns="geometry")
    if add_latlon_cols:
        df["Lat"] = gdf.geometry.y
        df["Lon"] = gdf.geometry.x

    utl.save_dataframe(df, file_path=file_path, **kwargs)
