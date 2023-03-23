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
            Path(__file__).resolve().parents[1] / "config" / config_subfolder / filename
        )
    else:
        full_path = Path(__file__).resolve().parents[1] / "config" / filename

    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    return yaml_dict


def get_data_catalog_params(dataset_name):
    """
    Load data catalog yaml file and return dictionary
    """

    data_catalog = load_yaml_config("data_catalog.yaml")
    return data_catalog[dataset_name]


def load_points_gdf(filename, folder, lat_name="Lat", lon_name="Lon", crs="EPSG:4326"):
    """Load CSV with LatLon columns into a GeoDataFrame"""

    full_path = Path(__file__).resolve().parents[2] / "data" / folder / filename

    points_df = pd.read_csv(full_path, index_col=0)
    points_gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df[lon_name], points_df[lat_name]),
        crs=crs,
    )

    return points_gdf


def minmax_normalize_image(image):

    img_min, img_max = image.min(), image.max()
    return (image - img_min) / (img_max - img_min)


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
