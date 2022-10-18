from pathlib import Path

import geopandas as gpd
import pandas as pd


def merge_shapes_and_keys(shrug_shp, shrug_r_keys):
    """
    Match the shapefile geometries to the SHRUG keys.

    Parameters
    ----------
    shrug_shp : gpd.GeoDataFrame
        The shapefile to match the keys to.
    shrug_r_keys : pd.DataFrame
        The keys to match the shapefile to.

    Returns
    -------
    gpd.GeoDataFrame

    """
    return pd.merge(
        shrug_shp,
        shrug_r_keys,
        on=[
            "pc11_s_id",
            "pc11_d_id",
            "pc11_sd_id",
            "pc11_tv_id",
        ],
        how="inner",
    )


def load_shrug_shapefiles(level="village"):
    """
    Import shapefiles from the data directory.

    Parameters
    ----------
    level : {"village", "subdistrict", "district", "state"}
        The level of the shapefile to import.

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


def lengthen_shapefile_ID_names(gdf):
    """
    Rename SHRUG shapefile ID column names to match the keys ID names.

    Use this when loading shrug shapefiles.
    This is necessary since shapefile column names are limited to 10 characters.

    """
    full_col_rename_dict = {
        "pc11_s_id": "pc11_state_id",
        "pc11_d_id": "pc11_district_id",
        "pc11_sd_id": "pc11_subdistrict_id",
        "pc11_tv_id": "pc11_village_id",
    }

    # Note: only rename columns that are present in the loaded shapefile
    # since we may be loading shapefiles at different levels.
    col_rename_dict = {
        col_name: new_col_name
        for col_name, new_col_name in full_col_rename_dict.items()
        if col_name in gdf.columns
    }

    return gdf.rename(columns=col_rename_dict)


def shorten_keys_ID_names(df):
    """
    Shorten shrug keys ID column names to match the shapefile column names.

    This is necessary since shapefile column names are limited to 10 characters.

    """
    shorten_ID_cols_dict = {
        "pc11_state_id": "pc11_s_id",
        "pc11_district_id": "pc11_d_id",
        "pc11_subdistrict_id": "pc11_sd_id",
        "pc11_village_id": "pc11_tv_id",
    }
    return df.rename(columns=shorten_ID_cols_dict)


# urban keys functions below not yet used
def clean_shrug_urban_keys(shrug_u_keys):
    """Drop any entries that are missing IDs."""
    shrug_u_keys_clean = shrug_u_keys.dropna(
        subset=[
            "pc11_state_id",
            "pc11_town_id",
        ]
    )

    return shrug_u_keys_clean


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


# function below is unused but kept for reference
def load_nasa_shapefiles():
    """
    Import shapefiles from the data directory and merge into one GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame

    """
    NASA_data_path = Path(__file__).parents[2] / "data" / "00_raw" / "NASA"
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
