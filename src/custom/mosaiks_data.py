import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .utils import latlon_df_to_gdf


def save_features_to_parquet(array, points_list, start_index=None, end_index=None, output_folder_path="data/", filename_prefix="mosaiks"):
    """Add latlons to a numpy array of MOSAIKS features and save to file as a gzipped parquet."""
    
    if start_index==None and end_index==None:
        start_index = 0
        end_index = array.shape[0]
    
    df = pd.DataFrame(
        array, 
        index=range(start_index, end_index), 
        columns=[str(c) for c in range(array.shape[1])]
    )
    
    df.insert(0, "Lat", points_list[start_index:end_index, 1])
    df.insert(1, "Lon", points_list[start_index:end_index, 0])

    filename = f"{filename_prefix}_{str(start_index)}_to_{str(end_index-1)}"
    os.makedirs(output_folder_path, exist_ok=True)
    df.to_parquet(output_folder_path+filename+".parquet.gzip", compression="gzip")


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
