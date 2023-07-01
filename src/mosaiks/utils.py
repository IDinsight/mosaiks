import os
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

os.environ["USE_PYGEOS"] = "0"


def load_yaml_config(filename: str, config_subfolder: str = None):
    """Load generic yaml files from config and return dictionary."""

    if config_subfolder:
        full_path = (
            Path(__file__).resolve().parents[2] / "config" / config_subfolder / filename
        )
    else:
        full_path = Path(__file__).resolve().parents[2] / "config" / filename

    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    return yaml_dict


# def get_data_catalog_params(dataset_name: str) -> dict:
#     """Load data catalog yaml file and return dictionary."""
#     data_catalog = load_yaml_config("data_catalog.yaml")
#     return data_catalog[dataset_name]


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

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if file_path.endswith(".csv"):
        return df.to_csv(file_path, **kwargs)
    elif file_path.endswith(".parquet") or file_path.endswith(".parquet.gzip"):
        return df.to_parquet(file_path, **kwargs)
    else:
        raise ValueError("File extension not recognized.")


def load_df_w_latlons_to_gdf(
    file_path: str,
    lat_name: str = "Lat",
    lon_name: str = "Lon",
    crs: str = "EPSG:4326",
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Load CSV with Lat-Lon columns into a GeoDataFrame.

    Parameters
    ----------
    file_path : If given, the path to the file to load.
    lat_name, lon_name : The names of the columns containing the latitude and longitude
        values.
        Default is 'Lat' and 'Lon'.
    crs : The coordinate reference system of the lat-lon columns.
        Default is 'EPSG:4326'.
    """
    df = load_dataframe(file_path=file_path, **kwargs)
    return df_w_latlons_to_gdf(df, lat_name, lon_name, crs)


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


def make_output_folder_path(
    featurization_config: dict, dataset_name: str = "temp"
) -> Path:
    """
    Get the path to the folder where the mosaiks features should be saved.

    Parameters
    ----------
    featurization_config : The featurization configuration dictionary. Must contain the keys
    'satellite_search_params' and 'model'.
    dataset_name : The name of the dataset used for featurization. Default is 'temp'.
    """

    satellite = featurization_config["satellite_search_params"]["satellite_name"]
    year = featurization_config["satellite_search_params"]["search_start"].split("-")[0]
    n_features = str(featurization_config["model"]["num_features"])

    data_path = Path(__file__).parents[2].resolve() / "data"
    folder_path = (
        data_path
        / "00_raw/mosaiks"
        / satellite
        / str(year)
        / dataset_name
        / str(n_features)
    )

    return folder_path


def get_mosaiks_package_link(branch="main") -> str:
    """Get the link to the mosaiks package."""

    secrets = load_yaml_config("secrets.yaml")
    GITHUB_TOKEN = secrets["GITHUB_TOKEN"]
    return f"git+https://{GITHUB_TOKEN}@github.com/IDinsight/mosaiks@{branch}"


def make_result_df(
    features: np.ndarray,
    index: pd.RangeIndex,
    mosaiks_col_names: list[str],
) -> pd.DataFrame:
    """
    Takes the features array and returns a dataframe with mosaiks features in the
    columns, and a specified index.

    Parameters
    -----------
    features : Array of features.
    index: Indices for feature dataframe.
    mosaiks_col_names : List of column names to label the feature columns as.

    Returns
    --------
    DataFrame
    """
    return pd.DataFrame(data=features, index=index, columns=mosaiks_col_names)
