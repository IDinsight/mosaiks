from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_mosaiks_data(filename):
    """
    Import MOSAIKS data as GeoDataFrame.

    Parameters
    ----------
    filename : str
        The name of the file to import.

    Returns
    -------
    gpd.GeoDataFrame

    """
    file_path = Path(__file__).parents[2] / "data" / "00_raw" / "MOSAIKS" / filename
    mosaiks_features = pd.read_csv(file_path)

    # Convert to GeoDataFrame using point coordinates
    mosaiks_coords_points = gpd.points_from_xy(
        x=mosaiks_features["Lon"],
        y=mosaiks_features["Lat"],
    )
    mosaiks_features_gdf = gpd.GeoDataFrame(
        mosaiks_features,
        geometry=mosaiks_coords_points,
        crs="EPSG:4326",
    )

    return mosaiks_features_gdf
