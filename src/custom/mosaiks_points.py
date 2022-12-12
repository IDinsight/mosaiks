import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


def load_points_gdf(filepath, lat_name="Lat", lon_name="Lon", crs="EPSG:4326"):
    """Load CSV with LatLon columns into a GeoDataFrame"""

    points_df = pd.read_csv(filepath)
    points_gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df[lon_name], points_df[lat_name]),
        crs=crs,
    )
    del points_df

    return points_gdf


def create_gdf_of_enclosed_points(shapes_gdf, step=0.05, pre_calc_bounds=None):
    """
    Create a GeoDataFrame of grid point coordinates enclosed by shapes in shapes_gdf.

    Parameters
    ----------
    shapes_gdf : gpd.GeoDataFrame
        A GeoDataFrame containing shapes in the geometry column
    step : float, default 0.05
        The step size to use for the point coordinates. Min = 0.01.
    pre_calc_bounds : {None, "india"}
        If given, returns pre-calculated bounds. If None, the bounds are calculated.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the selected points

    """
    bounds = _get_total_bounds(shapes_gdf, pre_calc_bounds=pre_calc_bounds)
    points_grid_gdf = _create_grid_of_points(*bounds, step=step)
    selected_points_gdf = _inner_join_points(points_grid_gdf, shapes_gdf)

    logging.info(f"Number of point coords in grid:{points_grid_gdf.shape[0]}")
    logging.info(f"Number of point coords selected:{selected_points_gdf.shape[0]}")

    return selected_points_gdf


def points_to_latlon_df(points_geometry, file_name=None):
    """
    Convert a GeoDataFrame of points to a DataFrame with lat and long columns and save to csv (optional).

    Parameters
    ----------
    points_geometry : gpd.GeoDataFrame
        A GeoDataFrame containing point coordinates in the geometry column
    filename : str, default None
        The filename to save the DataFrame to. If None, the DataFrame is not saved.

    Returns
    -------
    pd.DataFrame

    """
    latlon_list_df = pd.DataFrame({"Lat": points_geometry.y, "Lon": points_geometry.x})

    if file_name:
        file_path = (
            Path(__file__).parents[2]
            / "data"
            / "01_preprocessed"
            / "mosaiks_request_points"
            / f"{file_name}.csv"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        latlon_list_df.to_csv(file_path, index=False)

    return latlon_list_df


def plot_selected_points(
    selected_points_gdf, color_column, file_name="selected_points"
):
    """
    Plot the selected points on a map.

    Parameters
    ----------
    selected_points_gdf : gpd.GeoDataFrame
        A GeoDataFrame containing the selected points
    color_column : string
        Name of the column to color the dots by
    filename : str, default None
        The filename to save the plot to.

    Returns
    -------
    None

    """
    selected_points_gdf.plot(
        column=selected_points_gdf[color_column].astype(str),
        figsize=(10, 10),
        markersize=0.1,
    )

    plt.axis("off")
    plt.title("Chosen point coordinates")
    plt.tight_layout()

    file_path = (
        Path(__file__).parents[2]
        / "data"
        / "01_preprocessed"
        / "mosaiks_request_points"
        / f"{file_name}.png"
    )
    plt.savefig(file_path)

    plt.show()


def _get_total_bounds(gdf, pre_calc_bounds=None):
    """
    Returns the bounds of a GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        A GeoDataFrame containing shapes in the geometry column
    pre_calc_bounds : {None, "india"}
        If given, returns pre-calculated bounds. If None, the bounds are calculated.

    Returns
    -------
    list of floats
        [min_long, min_lat, max_long, max_lat]

    """
    if pre_calc_bounds == "india":
        logging.info("Note: Using pre-calculated bounds for India.")
        return [68.48448624, 6.75487781, 97.20992258, 35.49455663]
    else:
        return gdf.total_bounds


def _create_grid_of_points(
    min_long,
    min_lat,
    max_long,
    max_lat,
    step=0.05,
):
    """
    Create a grid of point coordinates with given bounds.

    Parameters
    ----------
    min_long, min_lat, max_long, max_lat : float
        The bounds of the area of interest
    step : float, default 0.05
        The step size to use for the point coordinates. Min = 0.01.

    Returns
    -------
    gpd.GeoDataFrame with point coordinates in the geometry column

    """
    # create a grid of point coordinates
    lat_list = np.arange(min_lat, max_lat, step)
    long_list = np.arange(min_long, max_long, step)
    points_grid_list = np.array([(lat, long) for lat in lat_list for long in long_list])

    # create a GeoDataFrame from the point coordinates
    points_geom = gpd.points_from_xy(x=points_grid_list[:, 1], y=points_grid_list[:, 0])
    points_grid_gdf = gpd.GeoDataFrame(geometry=points_geom, crs="EPSG:4326")

    return points_grid_gdf


def _inner_join_points(points_grid_gdf, shapes_gdf):
    """
    Select points that are enclosed by the shapes in the shapes_gdf.

    Parameters
    ----------
    points_grid_gdf : gpd.GeoDataFrame
        A GeoDataFrame containing point coordinates in the geometry column
    shapes_gdf : gpd.GeoDataFrame
        A GeoDataFrame containing shapes in the geometry column

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the selected points

    """
    selected_points_gdf = points_grid_gdf.sjoin(shapes_gdf)
    selected_points_gdf.drop(columns="index_right", inplace=True)
    selected_points_gdf.sort_values(by=["shrid"], inplace=True)

    return selected_points_gdf
