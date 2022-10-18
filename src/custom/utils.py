from pathlib import Path

import geopandas as gpd


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
