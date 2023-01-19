import os
from pathlib import Path

import pandas as pd
import yaml

os.environ["USE_PYGEOS"] = "0"

import geopandas as gpd  # noqa: E402


def load_yaml_config(filename, config_subfolder=None):
    """
    Load generic yaml files from config and return dictionary
    """
    if config_subfolder:
        full_path = Path(__file__).parents[1] / "config/{}/{}".format(
            config_subfolder, filename
        )
    else:
        full_path = Path(__file__).parents[1] / "config/{}".format(filename)

    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    return yaml_dict


def load_points_gdf(filename, folder, lat_name="Lat", lon_name="Lon", crs="EPSG:4326"):
    """Load CSV with LatLon columns into a GeoDataFrame"""

    full_path = Path(__file__).parents[1] / f"data/{folder}/{filename}"

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
    folder_path = Path(__file__).parents[2] / "data" / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    gdf.to_file(folder_path / file_name)


def load_dataset(dataset_name):

    data_catalog = load_yaml_config("data_catalog.yaml")
    folder_path = data_catalog[dataset_name]["folder"]
    filename = data_catalog[dataset_name]["filename"]

    file_path = Path(__file__).parents[2] / "data" / folder_path / filename

    return pd.read_csv(file_path)


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
