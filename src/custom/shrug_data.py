from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_shrug_secc():
    """
    Import the SHRUG SECC data.

    Returns
    -------
    pd.DataFrame

    """

    file_path = (
        Path(__file__).parents[2]
        / "data"
        / "00_raw"
        / "SHRUG"
        / "shrug-v1.5.samosa-secc-csv"
        / "shrug_secc.csv"
    )

    return pd.read_csv(file_path)


def load_shrug_shapefiles(level="village"):
    """
    Import shapefiles from the data directory.

    Parameters
    ----------
    level : {"village", "subdistrict", "district", "state"}
        The level of the shapefile to import. Note: The village
        level also includes towns.

    Returns
    -------
    gpd.GeoDataFrame

    """
    levels = ["state", "district", "subdistrict", "village"]
    if level not in levels:
        raise ValueError(
            "level must be one of 'state', 'district', 'subdistrict', or 'village'"
        )

    # load the shapefile
    shapefile_path = (
        Path(__file__).parents[2]
        / "data"
        / "00_raw"
        / "SHRUG"
        / "geometries_shrug-v1.5.samosa-open-polygons-shp"
        / f"{level}.shp"
    )

    return gpd.read_file(shapefile_path)


def load_shrug_urban_keys():
    """
    Import the SHRUG urban 2011 keys.

    Returns
    -------
    pd.DataFrame

    """
    shrug_u_keys = pd.read_csv(
        Path(__file__).parents[2]
        / "data"
        / "00_raw"
        / "SHRUG"
        / "shrug-v1.5.samosa-keys-csv"
        / "shrug_pc11u_key.csv"
    )

    return shrug_u_keys


def load_shrug_rural_keys():
    """
    Import the SHRUG rural 2011 keys.

    Returns
    -------
    pd.DataFrame

    """
    shrug_r_keys = pd.read_csv(
        Path(__file__).parents[2]
        / "data"
        / "00_raw"
        / "SHRUG"
        / "shrug-v1.5.samosa-keys-csv"
        / "shrug_pc11r_key.csv"
    )

    return shrug_r_keys


def clean_shrug_rural_keys(shrug_r_keys):
    """Drop any entries that are missing IDs."""

    shrug_r_keys_clean = shrug_r_keys.dropna(
        subset=[
            "pc11_state_id",
            "pc11_district_id",
            "pc11_subdistrict_id",
            "pc11_village_id",
        ]
    )

    return shrug_r_keys_clean


def change_shapefile_IDs_to_int(shrug_shp):
    """
    Change the ID columns of a SHRUG shapefile to integers. Also removes trailing zeros.

    Parameters
    ----------
    shrug_shp : gpd.GeoDataFrame
        The shapefile to process the columns of.

    Returns
    -------
    shrug_shp : gpd.GeoDataFrame

    """
    ID_col_names = [
        "pc11_s_id",
        "pc11_d_id",
        "pc11_sd_id",
        "pc11_tv_id",
    ]

    # Note: only rename columns that are present in the loaded shapefile
    # since we may be loading shapefiles at different levels.
    ID_col_names_present = [
        col_name for col_name in ID_col_names if col_name in shrug_shp.columns
    ]

    for col_name in ID_col_names_present:
        shrug_shp[col_name] = shrug_shp[col_name].astype(int)

    return shrug_shp


# function below is unused but kept for reference
# def load_nasa_shapefiles():
#     """
#     Import shapefiles from the data directory and merge into one GeoDataFrame.

#     Returns
#     -------
#     gpd.GeoDataFrame

#     """
#     NASA_data_path = Path(__file__).parents[2] / "data" / "00_raw" / "NASA"
#     dbf_filepaths = NASA_data_path.glob("**/*.dbf")

#     dbf_list = []

#     for filepath in dbf_filepaths:
#         print("Importing: ", filepath.name)
#         dbf = gpd.read_file(filepath)
#         # convert to Lat-Long coords
#         dbf = dbf.to_crs(epsg=4326)
#         dbf_list.append(dbf)

#     dbf = pd.concat(dbf_list, ignore_index=True)

#     return dbf
