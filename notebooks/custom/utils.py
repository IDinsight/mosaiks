from pathlib import Path
import logging

import geopandas as gpd
import pandas as pd
import numpy as np


def import_shapefiles():
    """
    Import shapefiles from the data directory and merge into one GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame

    """
    NASA_data_path = Path(__file__).parents[2] / "data" / "NASA"
    dbf_filepaths = NASA_data_path.glob("**/*.dbf")

    dbf_list = []

    for filepath in dbf_filepaths:
        print("Importing: ", filepath.name)
        dbf = gpd.read_file(filepath)
        # convert to Lat-Long coords
        dbf = dbf.to_crs(epsg=4326)
        dbf_list.append(dbf)

    dbf = pd.concat(dbf_list, ignore_index=True)

    return dbf


def get_bounds(gdf):
    """
    Get the bounds of a GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame

    Returns
    -------
    dict

    """
    min_long, min_lat, max_long, max_lat = gdf.total_bounds.round(2)

    return {
        "min_long": min_long,
        "min_lat": min_lat,
        "max_long": max_long,
        "max_lat": max_lat,
    }


def create_coords_list(min_long, min_lat, max_long, max_lat, step=0.05):
    """
    Create a list of coordinates to use for the NASA API.

    Parameters
    ----------
    min_long, min_lat, max_long, max_lat : float
        The bounds of the area of interest.
    step : float, optional
        The step size to use for the coordinates.
        The default is 0.05.
        Max resolution is 0.01.

    Returns
    -------
    list of lat-long coordinate tuples

    """

    lat_list = np.arange(min_lat, max_lat, step).round(2)
    long_list = np.arange(min_long, max_long, step).round(2)

    coords_list = np.array(
        [(lat, long) for lat in lat_list for long in long_list]
    ).round(2)

    return coords_list


def coords_list_to_gdf(coords_list):

    # Convert coords pairs to geopandas points
    geometry = gpd.points_from_xy(x=coords_list[:, 1], y=coords_list[:, 0])
    coords_gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

    return coords_gdf
