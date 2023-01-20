# TO-DO

# Contents of main() should be a separate function

# Add
# 1. path_to_shapes
# 2. step
# 3. pre_calc_bounds
# 4. subset to 6 states var
# as script arguments

import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.utils as utl

logging.basicConfig(level=logging.INFO)

__all__ = ["create_mosaiks_points", "load_points_gdf", "plot_selected_points"]


@utl.log_progress
def create_mosaiks_points(points_step, pre_calc_bounds=None):
    """Create a GeoDataFrame of points enclosed by the SHRUG shapes and save to file."""

    shrug_key_geoms = load_shrug_keys_w_shape()
    selected_points_gdf = create_gdf_of_enclosed_points(
        shrug_key_geoms,
        step=points_step,
        pre_calc_bounds=pre_calc_bounds,
    )
    save_request_points(selected_points_gdf.geometry)


@utl.log_progress
def load_shrug_keys_w_shape():

    data_catalog = utl.get_data_catalog_params("shrug_all_keys_with_shapes")
    return utl.load_gdf(
        data_catalog["folder"],
        data_catalog["filename"],
    )


def save_request_points(points_gdf):

    latlon_list_df = pd.DataFrame({"Lat": points_gdf.y, "Lon": points_gdf.x})
    utl.save_csv_dataset(latlon_list_df, "request_points")


@utl.log_progress
def create_gdf_of_enclosed_points(shapes_gdf, step, pre_calc_bounds):
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
    bounds = _get_total_bounds(shapes_gdf, pre_calc_bounds)
    points_grid_gdf = _create_grid_of_points(*bounds, step=step)
    selected_points_gdf = _inner_join_points(points_grid_gdf, shapes_gdf)

    logging.info(f"Number of point coords in grid:{points_grid_gdf.shape[0]}")
    logging.info(f"Number of point coords selected:{selected_points_gdf.shape[0]}")

    return selected_points_gdf


def _get_total_bounds(gdf, geo_region=None):
    """
    Returns the bounds of a GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        A GeoDataFrame containing shapes in the geometry column
    geo_region : None or str
        If given, returns pre-calculated bounds. If None, the bounds are calculated.

    Returns
    -------
    list of floats
        [min_long, min_lat, max_long, max_lat]

    """

    pre_calculated_bounds = utl.load_yaml_config("geographic_bounds.yaml")
    if geo_region is None:
        return gdf.total_bounds
    else:
        bounds = pre_calculated_bounds.get(geo_region)
        if bounds is None:
            raise NotImplementedError(f"bounds for `{geo_region}` has not been defined")
        else:
            logging.info(f"Note: Using pre-calculated bounds for {geo_region}.")
            return bounds


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
    # drop duplicated points (points that sit inside multiple shapes)
    # IMPROVE THIS PROCESS
    selected_points_gdf.drop_duplicates(subset=["geometry"], inplace=True)
    selected_points_gdf.drop(columns="index_right", inplace=True)
    selected_points_gdf.sort_values(by=["shrid"], inplace=True)

    return selected_points_gdf


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
    plt.show()


if __name__ == "__main__":

    preprocessing_config = utl.load_yaml_config("preprocessing.yaml")[
        "point_coordinates"
    ]
    create_mosaiks_points(
        preprocessing_config["step"], preprocessing_config["pre_calc_bounds"]
    )
