import os
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd


def load_dataframe(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load file with tabular data (csv or parquet) as a pandas DataFrame.

    Parameters
    ----------
    file_path : The path to the file to load.
    **kwargs : Keyword arguments to pass to the pandas read function.
    """

    file_path = str(file_path)

    if file_path.endswith(".csv"):
        return pd.read_csv(file_path, **kwargs)
    elif file_path.endswith(".parquet") or file_path.endswith(".parquet.gzip"):
        return pd.read_parquet(file_path, **kwargs)
    else:
        raise ValueError("File extension not recognized")


def load_and_combine_dataframes(folder_path: str, filenames: List[str]) -> pd.DataFrame:
    """
    Given folder path and filenames, load multiple dataframes and combine
    them, sording by the index.
    """

    dfs = []
    for filename in filenames:
        df = load_dataframe(file_path=folder_path / filename)
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0)
    combined_df.sort_index(inplace=True)

    return combined_df


def save_dataframe(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Save pandas dataframe to .csv, .parquet etc file based on extension.

    Parameters
    ----------
    df : The dataframe to save.
    file_path : If given, the path to the file to load.
    """
    file_path = str(file_path)

    if os.path.dirname(file_path) != "":
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if file_path.endswith(".csv"):
        return df.to_csv(file_path, **kwargs)
    elif file_path.endswith(".parquet") or file_path.endswith(".parquet.gzip"):
        return df.to_parquet(file_path, **kwargs)
    else:
        raise ValueError("File extension not recognized.")


def df_w_latlons_to_gdf(
    df: pd.DataFrame,
    lat_name: str = "Lat",
    lon_name: str = "Lon",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Convert DataFrame to GeoDataFrame by creating geometries from lat-lon columns.

    Parameters
    ----------
    df : The DataFrame to convert.
    lat_name, lon_name : The names of the columns containing the latitude and longitude values.
        Default is 'Lat' and 'Lon'.
    crs : The coordinate reference system of the lat-lon columns.
    """
    latlon_point_geoms = gpd.points_from_xy(x=df[lon_name], y=df[lat_name])
    gdf = gpd.GeoDataFrame(df.copy(), geometry=latlon_point_geoms, crs=crs)
    return gdf


def get_filtered_filenames(folder_path: str, prefix: str = "df_") -> List[str]:
    """
    Get the paths to all files in a folder that start with a given prefix.

    Parameters
    ----------
    folder_path : The path to the folder.
    prefix : The prefix that the filenames must start with.
    """

    all_filenames = os.listdir(folder_path)
    filtered_filenames = [file for file in all_filenames if file.startswith(prefix)]
    return sorted(filtered_filenames)


def make_result_df(
    features: np.ndarray,
    context_gdf: gpd.GeoDataFrame,
    mosaiks_col_names: list[str],
) -> pd.DataFrame:
    """
    Takes the features array and returns a dataframe with mosaiks features in the
    columns, plus the stac_id, with the same index as the context GeoDataFrame.

    Parameters
    -----------
    features : Array of features.
    context_gdf : GeoDataFrame of context variables. Must have the same index size as
        the features array. Must also have a "stac_item" column which contains
        pystac.item.Item objects since the "stac_id" is always saved.
    mosaiks_col_names : List of column names to label the feature columns as.

    Returns
    --------
    DataFrame
    """
    # Make features dataframe
    features_df = pd.DataFrame(
        data=features, index=context_gdf.index, columns=mosaiks_col_names
    )

    # Add stac_id to features dataframe
    if isinstance(context_gdf["stac_item"].iloc[0], list):
        features_df["stac_id"] = context_gdf["stac_item"].map(
            lambda item_list: [
                item.id if item is not None else None for item in item_list
            ]
        )
    else:
        features_df["stac_id"] = context_gdf["stac_item"].map(
            lambda item: item.id if item is not None else None
        )
    return features_df
