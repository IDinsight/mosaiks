from pathlib import Path
import logging

import geopandas as gpd
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, kendalltau, spearmanr

import seaborn as sns


def import_shrug_shapefiles():

    shapefile_path = (
        Path(__file__).parents[2]
        / "data"
        / "SHRUG"
        / "geometries_shrug-v1.5.samosa-open-polygons-shp"
        / "village.shp"
    )
    shrug_shp = gpd.read_file(shapefile_path)

    shrug_shp.rename(
        columns={
            "pc11_s_id": "pc11_state_id",
            "pc11_d_id": "pc11_district_id",
            "pc11_sd_id": "pc11_subdistrict_id",
            "pc11_tv_id": "pc11_village_id",
        },
        inplace=True,
    )

    # change to int (also removes trailing zeros)
    shrug_shp = shrug_shp.astype(
        {
            "pc11_state_id": int,
            "pc11_district_id": int,
            "pc11_subdistrict_id": int,
            "pc11_village_id": int,
        }
    )

    return shrug_shp


def import_nasa_shapefiles():
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


def import_shrug_r_keys():
    """
    Import the SHRUG rural 2011 keys.

    Returns
    -------
    pd.DataFrame

    """
    shrug_r_keys = pd.read_csv(
        Path(__file__).parents[2]
        / "data"
        / "SHRUG"
        / "shrug-v1.5.samosa-keys-csv"
        / "shrug_pc11r_key.csv"
    )

    # drop any entries that are missing IDs
    shrug_r_key_clean = shrug_r_keys.dropna(
        subset=[
            "pc11_state_id",
            "pc11_district_id",
            "pc11_subdistrict_id",
            "pc11_village_id",
        ]
    )

    return shrug_r_key_clean


def import_shrug_u_keys():
    """
    Import the SHRUG urban 2011 keys.

    Returns
    -------
    pd.DataFrame

    """
    shrug_u_keys = pd.read_csv(
        Path(__file__).parents[2]
        / "data"
        / "SHRUG"
        / "shrug-v1.5.samosa-keys-csv"
        / "shrug_pc11u_key.csv"
    )

    # drop any entries that are missing IDs
    shrug_u_key_clean = shrug_u_keys.dropna(
        subset=[
            "pc11_state_id",
            "pc11_town_id",
        ]
    )

    return shrug_u_key_clean


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


def show_results(y_test, y_pred):
    """
    Print stats and plot true vs predicted values.

    Parameters
    ----------
    y_test : array-like
        The true values
    y_pred : array-like
        The predicted values

    Returns
    -------
    None

    """
    print("r2: %f" % r2_score(y_test, y_pred))
    print("pearson: %f" % pearsonr(y_test, y_pred)[0])
    print("kendall: %f" % kendalltau(y_test, y_pred)[0])
    print("spearman: %f" % spearmanr(y_test, y_pred)[0])

    # scatterplot
    ax = sns.scatterplot(x=y_pred, y=y_test)
    sns.lineplot(x=[0, 1], y=[0, 1], color="black", linestyle="--", ax=ax)
    ax.set(xlabel="Predicted", ylabel="Observed")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-0.2, 1.1)
