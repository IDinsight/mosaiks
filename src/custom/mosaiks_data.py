from pathlib import Path

import geopandas as gpd
import pandas as pd

from .utils import latlon_df_to_gdf

def load_mosaiks_data(folder_name):
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
    file_path = (
        Path(__file__).parents[2]
        / "data"
        / "00_raw"
        / "MOSAIKS"
        / folder_name
        / "Mosaiks_features.csv"
    )
    mosaiks_features = pd.read_csv(file_path)
    
    mosaiks_features_gdf = latlon_df_to_gdf(
        mosaiks_features,
        lat_name="Lat",
        lon_name="Lon"
    )
    
    return mosaiks_features_gdf
