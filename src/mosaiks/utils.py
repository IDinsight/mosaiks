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
        full_path = Path(__file__).resolve().parents[2] / "config/{}/{}".format(
            config_subfolder, filename
        )
    else:
        full_path = Path(__file__).resolve().parents[2] / "config/{}".format(filename)

    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    return yaml_dict


def get_data_catalog_params(dataset_name):

    data_catalog = load_yaml_config("data_catalog.yaml")
    return data_catalog[dataset_name]


def load_points_gdf(filename, folder, lat_name="Lat", lon_name="Lon", crs="EPSG:4326"):
    """Load CSV with LatLon columns into a GeoDataFrame"""

    full_path = Path(__file__).resolve().parents[2] / f"data/{folder}/{filename}"

    points_df = pd.read_csv(full_path)
    points_gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df[lon_name], points_df[lat_name]),
        crs=crs,
    )
    del points_df

    return points_gdf


def latlon_df_to_gdf(df, lat_name="Lat", lon_name="Lon"):
    """Convert df to GeoDataFrame using latlon columns"""

    latlon_point_geoms = gpd.points_from_xy(
        x=df[lon_name],
        y=df[lat_name],
    )
    gdf = gpd.GeoDataFrame(
        df,
        geometry=latlon_point_geoms,
        crs="EPSG:4326",
    )
    return gdf


def save_gdf(gdf, folder_name, file_name):
    """
    Save gdf to file as .shp .dbf etc.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to save.
    folder_name : str
        The name of the folder to save the file to.
    file_name : str
        The name of the file to save.

    Returns
    -------
    None

    """
    folder_path = Path(__file__).resolve().parents[2] / "data" / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    gdf.to_file(folder_path / file_name)


def load_csv_dataset(dataset_name):

    data_catalog = get_data_catalog_params(dataset_name)
    folder_path = data_catalog["folder"]
    filename = data_catalog["filename"]

    file_path = Path(__file__).resolve().parents[2] / "data" / folder_path / filename

    return pd.read_csv(file_path)


def save_csv_dataset(dataset, dataset_name):

    data_catalog = get_data_catalog_params(dataset_name)
    filepath = (
        Path(__file__).resolve().parents[2]
        / "data"
        / data_catalog["folder"]
        / data_catalog["filename"]
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(filepath)


def load_gdf(folder_name, file_name):
    """
    Load gdf from shapefile.

    Parameters
    ----------
    folder_name : str
        The name of the folder to load the file from.
    file_name : str
        The name of the file to load.

    Returns
    -------
    gpd.GeoDataFrame

    """
    file_path = Path(__file__).parents[2] / "data" / folder_name / file_name
    return gpd.read_file(file_path)


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
