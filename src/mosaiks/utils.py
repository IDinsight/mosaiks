import functools
import logging
import os
from pathlib import Path
from time import time

import pandas as pd
import yaml

os.environ["USE_PYGEOS"] = "0"

import geopandas as gpd  # noqa: E402


def load_yaml_config(filename, config_subfolder=None):
    """
    Load generic yaml files from config and return dictionary
    """

    if config_subfolder:
        full_path = (
            Path(__file__).resolve().parents[2] / "config" / config_subfolder / filename
        )
    else:
        full_path = Path(__file__).resolve().parents[2] / "config" / filename

    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    return yaml_dict


def get_data_catalog_params(dataset_name):
    """
    Load data catalog yaml file and return dictionary
    """

    data_catalog = load_yaml_config("data_catalog.yaml")
    return data_catalog[dataset_name]


def create_dataset_path(folder, filename):
    """
    Create data path from folder and filename
    """

    path = Path(__file__).resolve().parents[2] / "data" / folder / filename
    return str(path)


def get_dataset_path_and_kwargs(dataset_name):
    """
    Get data path and kwargs from data catalog.
    """

    data_catalog = get_data_catalog_params(dataset_name)
    folder = data_catalog.pop("folder")
    filename = data_catalog.pop("filename")
    kwargs = data_catalog

    file_path = create_dataset_path(folder, filename)

    return file_path, kwargs


def load_dataframe(dataset_name=None, file_path=None, **kwargs):
    """
    Load file with tabular data (csv or parquet) as a pandas DataFrame.
    Either dataset_name or file_path must be given.

    Parameters
    ----------
    dataset_name : str, optional
        The name of the dataset to load from the data catalog. If given, the
        load the filepath and kwargs from the data catalog.
    file_path : str, optional
        If given, the path to the file to load.


    Returns
    -------
    pd.DataFrame
    """

    if dataset_name:
        file_path, kwargs = get_dataset_path_and_kwargs(dataset_name)
    elif file_path:
        file_path = file_path

    if file_path.endswith(".csv"):
        return pd.read_csv(file_path, **kwargs)
    elif file_path.endswith(".parquet") or file_path.endswith(".parquet.gzip"):
        return pd.read_parquet(file_path, **kwargs)
    else:
        raise ValueError("File extension not recognized")


def save_dataframe(df, dataset_name=None, file_path=None, **kwargs):
    """
    Save pandas dataframe to .csv .parquet etc file based on extension.
    Either dataset_name or file_path must be given.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to save.
    dataset_name : str, optional
        The name of the dataset to load from the data catalog. If given, the
        load the filepath and kwargs from the data catalog.
    file_path : str, optional
        If given, the path to the file to load.
    """

    if dataset_name:
        file_path, kwargs = get_dataset_path_and_kwargs(dataset_name)
    elif file_path:
        file_path = file_path

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if file_path.endswith(".csv"):
        return df.to_csv(file_path, **kwargs)
    elif file_path.endswith(".parquet") or file_path.endswith(".parquet.gzip"):
        return df.to_parquet(file_path, **kwargs)
    else:
        raise ValueError("File extension not recognized.")


def save_geodataframe_as_dataframe(
    gdf, dataset_name=None, file_path=None, add_latlon_cols=False, **kwargs
):
    """
    Save GeoDataFrame as a DataFrame by dropping geometry column.
    Either dataset_name or file_path must be given.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to save.
    dataset_name : str, optional
        The name of the dataset to load from the data catalog. If given, the
        load the filepath and kwargs from the data catalog.
    file_path : str, optional
        If given, the path to the file to load.
    add_latlon_cols : bool, optional
        If True, add columns for latitude and longitude.

    Returns
    -------
    None
    """

    df = gdf.drop(columns="geometry")
    if add_latlon_cols:
        df["Lat"] = gdf.geometry.y
        df["Lon"] = gdf.geometry.x

    save_dataframe(df, dataset_name, file_path, **kwargs)


def load_df_w_latlons_to_gdf(
    dataset_name=None,
    file_path=None,
    lat_name="Lat",
    lon_name="Lon",
    crs="EPSG:4326",
    **kwargs
):
    """
    Load CSV with Lat-Lon columns into a GeoDataFrame.
    Either dataset_name or file_path must be given.

    dataset_name : str, optional
        The name of the dataset to load from the data catalog. If given, the
        load the filepath and kwargs from the data catalog.
    file_path : str, optional
        If given, the path to the file to load.
    lat_name, lon_name : str, optional
        The names of the columns containing the latitude and longitude values.
        Default is 'Lat' and 'Lon'.
    crs : str, optional
        The coordinate reference system of the lat-lon columns.
        Default is 'EPSG:4326'.

    Returns
    -------
    gpd.GeoDataFrame
    """

    df = load_dataframe(dataset_name, file_path, **kwargs)
    return df_w_latlons_to_gdf(df, lat_name, lon_name, crs)


def df_w_latlons_to_gdf(df, lat_name="Lat", lon_name="Lon", crs="EPSG:4326"):
    """
    Convert DataFrame to GeoDataFrame by creating geometries from lat-lon columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to convert.
    lat_name, lon_name : str, optional
        The names of the columns containing the latitude and longitude values.
        Default is 'Lat' and 'Lon'.
    crs : str, optional
        The coordinate reference system of the lat-lon columns.

    Returns
    -------
    gpd.GeoDataFrame
    """

    latlon_point_geoms = gpd.points_from_xy(x=df[lon_name], y=df[lat_name])
    gdf = gpd.GeoDataFrame(df.copy(), geometry=latlon_point_geoms, crs=crs)
    return gdf


def make_mosaiks_filepath(
    satellite,
    year,
    coord_set_name,
    n_features,
    filename="features.parquet.gzip",
):
    """
    Creates path to mosaiks features from a given
    satellite, year, number of features, and filename.
    """

    data_path = Path(__file__).parents[2].resolve() / "data"
    file_path = (
        data_path
        / "00_raw/mosaiks"
        / satellite
        / str(year)
        / coord_set_name
        / n_features
        / filename
    )

    return file_path


def log_progress(func):
    """
    Decorator to log the start and end of a function
    """

    @functools.wraps(func)
    def log_wrapper(*args, **kwargs):
        """
        Inner function
        """
        logging.info(" >>> Starting {:s} ... ".format(func.__name__))
        start_time = time()
        result = func(*args, **kwargs)
        time_diff = time() - start_time
        logging.info(
            " <<< Exiting {:s} in {:.2f} secs ... ".format(func.__name__, time_diff)
        )
        return result

    return log_wrapper
