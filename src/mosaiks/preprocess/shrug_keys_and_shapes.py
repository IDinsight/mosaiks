import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

import mosaiks.utils as utl

logging.basicConfig(level=logging.INFO)

__all__ = ["create_shapefile_w_keys"]


def create_shapefile_w_keys(geographic_level="village"):
    """Preprocess SHRUG keys (rural and urban) by adding shapes and saving to file."""

    shrug_shapes = preprocess_shape_files(geographic_level)
    keys = preprocess_keys()
    save_shrug_keys_w_shape(merge_shapes_and_keys(shrug_shapes, keys))

    logging.info("Created shape file with keys")


@utl.log_progress
def preprocess_shape_files(geographic_level):

    shrug_shapes = load_shrug_shapefiles(level=geographic_level)
    shrug_shapes = change_shapefile_IDs_to_int(shrug_shapes)

    return shrug_shapes


def preprocess_keys():

    rural_keys = preprocess_rural_keys()
    urban_keys = preprocess_urban_keys()

    shrug_all_keys = pd.concat([rural_keys, urban_keys])
    shrug_all_keys = shrug_all_keys.sort_values(by="pc11_tv_id")

    return shrug_all_keys


@utl.log_progress
def preprocess_rural_keys():

    shrug_r_keys = utl.load_csv_dataset("shrug_rural_keys")
    shrug_r_keys_clean = clean_rural_keys(shrug_r_keys)

    return shrug_r_keys_clean


def clean_rural_keys(shrug_r_keys):

    shrug_r_keys_clean = shrug_r_keys.dropna(
        subset=[
            "pc11_state_id",
            "pc11_district_id",
            "pc11_subdistrict_id",
            "pc11_village_id",
        ]
    )

    shrug_r_keys_clean = shrug_r_keys_clean.rename(
        columns={"pc11_village_id": "pc11_tv_id"}
    )
    shrug_r_keys_clean["is_urban"] = 0
    shrug_r_keys_clean = shrug_r_keys_clean[["pc11_tv_id", "is_urban", "shrid"]]

    return shrug_r_keys_clean


@utl.log_progress
def preprocess_urban_keys():
    shrug_u_keys = utl.load_csv_dataset("shrug_urban_keys")
    shrug_u_keys = clean_urban_keys(shrug_u_keys)

    return shrug_u_keys


def clean_urban_keys(shrug_u_keys):

    shrug_u_keys = shrug_u_keys.rename(columns={"pc11_town_id": "pc11_tv_id"})
    shrug_u_keys["is_urban"] = 1
    shrug_u_keys = shrug_u_keys[["pc11_tv_id", "is_urban", "shrid"]]

    return shrug_u_keys


@utl.log_progress
def merge_shapes_and_keys(shrug_shapes, shrug_keys):

    shrug_all_keys_with_shapes = pd.merge(
        shrug_shapes,
        shrug_keys,
        on="pc11_tv_id",
        how="inner",
    )

    return shrug_all_keys_with_shapes


def save_shrug_keys_w_shape(shrug_all_keys_with_shapes):

    utl.save_gdf(
        gdf=shrug_all_keys_with_shapes,
        folder_name="01_preprocessed/SHRUG/shrug_all_keys_with_shapes",
        file_name="shrug_all_keys_with_shapes.shp",
    )

    return


def load_shrug_shapefiles(level="village"):
    """
    Import shapefiles from the data directory.
    """

    levels = ["state", "district", "subdistrict", "village"]
    if level not in levels:
        raise NotImplementedError(
            "level must be one of 'state', 'district', 'subdistrict', or 'village'"
        )

    dataset_name = f"{level}_shrug_shapefiles"

    data_catalog = utl.load_yaml_config("data_catalog.yaml")[dataset_name]
    shapefile_path = (
        Path(__file__).parents[2]
        / "data"
        / data_catalog["folder"]
        / data_catalog["filename"]
    )

    return gpd.read_file(shapefile_path)


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
    ID_COL_NAMES = [
        "pc11_s_id",
        "pc11_d_id",
        "pc11_sd_id",
        "pc11_tv_id",
    ]

    # Note: only rename columns that are present in the loaded shapefile
    # since we may be loading shapefiles at different levels.
    ID_col_names_present = [
        col_name for col_name in ID_COL_NAMES if col_name in shrug_shp.columns
    ]

    for col_name in ID_col_names_present:
        shrug_shp[col_name] = shrug_shp[col_name].astype(int)

    return shrug_shp


if __name__ == "__main__":

    preprocessing_config = utl.load_yaml_config("preprocessing.yaml")
    create_shapefile_w_keys(preprocessing_config["geographic_level"])
