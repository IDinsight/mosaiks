import pandas as pd

from custom.mosaiks_data import load_mosaiks_data
from custom.shrug_data import load_shrug_secc
from custom.utils import load_gdf


def load_and_merge_data(moasaiks_folder_name="INDIA_SHRUG_Mosaiks_features"):

    mosaiks_features_gdf = load_mosaiks_data(moasaiks_folder_name)
    shrug_key_geoms = load_gdf(
        "01_preprocessed/SHRUG/shrug_pc11r_key_with_shapes",
        "shrug_pc11r_key_with_shapes.shp",
    )
    shrug_key_geoms = preserve_geometry(shrug_key_geoms, level="town_village")
    mosaiks_features_gdf = _add_shrid_to_mosaiks(mosaiks_features_gdf, shrug_key_geoms)
    shrug_secc = load_shrug_secc()

    return _merge_mosaiks_and_secc(mosaiks_features_gdf, shrug_secc)


def preserve_geometry(gdf, level="village"):
    """Preserve geometry in a GeoDataFrame by adding it to a new geometry column."""
    gdf["geometry_" + level] = gdf["geometry"].copy()

    return gdf


def _add_shrid_to_mosaiks(mosaiks_features_gdf, shrug_key_geoms):
    """Add SHRID to Mosaiks features.

    Parameters
    ----------
    mosaiks_features_gdf : geopandas.GeoDataFrame
        Mosaiks features GeoDataFrame.
    shrug_key_geoms : geopandas.GeoDataFrame
        SHRUG keys with shapes GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame

    """
    mosaiks_features_gdf = mosaiks_features_gdf.sjoin(shrug_key_geoms).drop(
        columns=["index_right"]
    )

    return mosaiks_features_gdf


def _merge_mosaiks_and_secc(mosaiks_features_gdf, shrug_secc):
    """Merge Mosaiks features with SECC data.

    Parameters
    ----------
    mosaiks_features_gdf : geopandas.GeoDataFrame
        Mosaiks features GeoDataFrame.
    shrug_secc : pandas.DataFrame
        SHRUG SECC data.

    Returns
    -------
    geopandas.GeoDataFrame

    """
    return pd.merge(mosaiks_features_gdf, shrug_secc, on="shrid")
