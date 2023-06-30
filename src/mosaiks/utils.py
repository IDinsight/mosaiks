import os
from pathlib import Path
from typing import List, Tuple

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


def get_data_catalog_params(dataset_name: str) -> dict:
    """Load data catalog yaml file and return dictionary."""
    data_catalog = load_yaml_config("data_catalog.yaml")
    return data_catalog[dataset_name]


def create_dataset_path(folder: str, filename: str) -> str:
    """Create data path from folder and filename."""

    path = Path(__file__).resolve().parents[2] / "data" / folder / filename
    return str(path)


def get_dataset_path_and_kwargs(dataset_name: str) -> Tuple[str, dict]:
    """Get data path and kwargs from data catalog."""

    data_catalog = get_data_catalog_params(dataset_name)
    folder = data_catalog.pop("folder")
    filename = data_catalog.pop("filename")
    kwargs = data_catalog

    file_path = create_dataset_path(folder, filename)

    return file_path, kwargs


def load_dataframe(
    file_path: str = None, dataset_name: str = None, **kwargs
) -> pd.DataFrame:
    """
    Load file with tabular data (csv or parquet) as a pandas DataFrame.
    Either dataset_name or file_path must be given.

    Parameters
    ----------
    dataset_name : The name of the dataset to load from the data catalog. If given, the
        load the filepath and kwargs from the data catalog.
    file_path : If given, the path to the file to load.
    **kwargs : Keyword arguments to pass to the pandas read function.
    """

    if dataset_name:
        file_path, kwargs = get_dataset_path_and_kwargs(dataset_name)
    elif file_path:
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


def save_dataframe(
    df: pd.DataFrame, file_path: str = None, dataset_name: str = None, **kwargs
) -> None:
    """
    Save pandas dataframe to .csv, .parquet etc file based on extension.
    Either dataset_name or file_path must be given.

    Parameters
    ----------
    df : The dataframe to save.
    dataset_name : The name of the dataset to load from the data catalog. If given, the
        load the filepath and kwargs from the data catalog.
    file_path : If given, the path to the file to load.
    """
    if dataset_name:
        file_path, kwargs = get_dataset_path_and_kwargs(dataset_name)
    elif file_path:
        file_path = str(file_path)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if file_path.endswith(".csv"):
        return df.to_csv(file_path, **kwargs)
    elif file_path.endswith(".parquet") or file_path.endswith(".parquet.gzip"):
        return df.to_parquet(file_path, **kwargs)
    else:
        raise ValueError("File extension not recognized.")


def save_geodataframe_as_dataframe(
    gdf: gpd.GeoDataFrame,
    file_path: str = None,
    dataset_name: str = None,
    add_latlon_cols: bool = False,
    **kwargs,
) -> None:
    """
    Save GeoDataFrame as a DataFrame by dropping geometry column.
    Either dataset_name or file_path must be given.

    Parameters
    ----------
    gdf : The GeoDataFrame to save.
    dataset_name : The name of the dataset to load from the data catalog. If given, the
        load the filepath and kwargs from the data catalog.
    file_path : If given, the path to the file to load.
    add_latlon_cols : If True, add columns for latitude and longitude.
    """
    df = gdf.drop(columns="geometry")
    if add_latlon_cols:
        df["Lat"] = gdf.geometry.y
        df["Lon"] = gdf.geometry.x

    save_dataframe(df, file_path=file_path, dataset_name=dataset_name, **kwargs)


def load_df_w_latlons_to_gdf(
    file_path: str = None,
    dataset_name: str = None,
    lat_name: str = "Lat",
    lon_name: str = "Lon",
    crs: str = "EPSG:4326",
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Load CSV with Lat-Lon columns into a GeoDataFrame.
    Either dataset_name or file_path must be given.

    Parameters
    ----------
    dataset_name : The name of the dataset to load from the data catalog. If given, the
        load the filepath and kwargs from the data catalog.
    file_path : If given, the path to the file to load.
    lat_name, lon_name : The names of the columns containing the latitude and longitude
        values.
        Default is 'Lat' and 'Lon'.
    crs : The coordinate reference system of the lat-lon columns.
        Default is 'EPSG:4326'.
    """
    df = load_dataframe(file_path=file_path, dataset_name=dataset_name, **kwargs)
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


def make_output_folder_path(featurization_config: dict) -> Path:
    """
    Get the path to the folder where the mosaiks features should be saved.

    Parameters
    ----------
    featurization_config : The featurization configuration dictionary. Must contain the keys
        'satellite_search_params' and 'model'.
    """

    # coord_set_name = featurization_config["coord_set"]["coord_set_name"]
    satellite = featurization_config["satellite_search_params"]["satellite_name"]
    year = featurization_config["satellite_search_params"]["search_start"].split("-")[0]
    n_features = str(featurization_config["model"]["num_features"])

    data_path = Path(__file__).parents[2].resolve() / "data"
    folder_path = (
        data_path
        / "00_raw/mosaiks"
        / satellite
        / str(year)
        # / coord_set_name
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


def combine_results_df_with_context_df(
    features_df: pd.DataFrame,
    context_gdf: gpd.GeoDataFrame,
    context_cols_to_keep: list[str] = None,
) -> pd.DataFrame:
    """
    Takes the features array and a context dataframe and returns a dataframe with the
    features, the stac_id of the images used to create each row, and chosen context
    columns.

    Note: context_gdf must have a "stac_item" column which contains pystac.item.Item
    objects since the "stac_id" is always saved.

    Parameters
    -----------
    features : Array of features.
    context_gdf : GeoDataFrame of context variables. Must have the same index size as
        the features array. Must also have a "stac_item" column which contains
        pystac.item.Item objects since the "stac_id" is always saved.
    context_cols_to_keep : List of context columns to include in final dataframe
        (optional). If not given, only "stac_id" will be included.

    Returns
    --------
    DataFrame
    """
    if isinstance(context_gdf["stac_item"].iloc[0], list):
        context_gdf["stac_id"] = context_gdf["stac_item"].map(
            lambda item_list: [
                item.id if item is not None else None for item in item_list
            ]
        )
    else:
        context_gdf["stac_id"] = context_gdf["stac_item"].map(
            lambda item: item.id if item is not None else None
        )
    context_cols_to_keep = context_cols_to_keep + ["stac_id"]
    context_gdf = context_gdf[context_cols_to_keep]

    return pd.concat([context_gdf, features_df], axis=1)
