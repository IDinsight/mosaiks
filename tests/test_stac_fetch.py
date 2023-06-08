"""Test stac item fetching."""
import geopandas as gpd
import numpy as np
import pytest
from pystac.item import Item

import mosaiks.utils as utl
from mosaiks.fetch.stacs import fetch_seasonal_stac_items, fetch_stac_items


@pytest.fixture(scope="module")
def test_points_with_stac(sample_test_data: gpd.GeoDataFrame):
    """Sample test data with stac IDs."""
    points_gdf = utl.df_w_latlons_to_gdf(sample_test_data)
    return fetch_stac_items(
        points_gdf, "landsat-8-c2-l2", "2013-04-01", "2014-03-31", "planetary-compute"
    )


@pytest.fixture(scope="module")
def test_points_with_stac_null(sample_test_null_data: gpd.GeoDataFrame):
    """Sample test data with stac IDs."""
    points_gdf = utl.df_w_latlons_to_gdf(sample_test_null_data)
    return fetch_stac_items(
        points_gdf, "landsat-8-c2-l2", "2013-04-01", "2014-03-31", "planetary-compute"
    )


@pytest.fixture(scope="module")
def test_points_with_seasonal_stac(sample_test_data: gpd.GeoDataFrame):
    """Sample test data with stac IDs."""
    points_gdf = utl.df_w_latlons_to_gdf(sample_test_data)
    return fetch_seasonal_stac_items(
        points_gdf, "landsat-8-c2-l2", 2013, "planetary-compute"
    )


# -----Tests for stac ID fetching-----
def test_if_stac_ids_are_added_to_test_df(test_points_with_stac: gpd.GeoDataFrame):
    assert "stac_item" in test_points_with_stac.columns and type(
        test_points_with_stac["stac_item"].tolist()[0] == Item
    )


def test_if_df_with_stac_ids_has_correct_shape(
    test_points_with_stac: gpd.GeoDataFrame, sample_test_data: gpd.GeoDataFrame
):
    assert len(test_points_with_stac) == len(sample_test_data)


# -----Tests for stac ID fetching with null data-----
def test_if_stac_ids_are_added_to_test_null_df(
    test_points_with_stac_null: gpd.GeoDataFrame,
):
    print(test_points_with_stac_null.columns)
    assert "stac_item" in test_points_with_stac_null.columns


def test_if_null_df_with_stac_ids_has_correct_shape(
    test_points_with_stac_null: gpd.GeoDataFrame, sample_test_data: gpd.GeoDataFrame
):
    assert len(test_points_with_stac_null) == len(sample_test_data)


def test_if_stac_ids_from_test_null_df_are_empty(
    test_points_with_stac_null: gpd.GeoDataFrame,
):
    assert test_points_with_stac_null["stac_item"].isnull().all()


# -----Tests for stac ID fetching with seasonal data-----
def test_if_seasonal_stac_ids_are_added_to_test_df(
    test_points_with_seasonal_stac: gpd.GeoDataFrame,
):
    assert "stac_item" in test_points_with_seasonal_stac.columns and type(
        test_points_with_seasonal_stac["stac_item"].tolist()[0] == Item
    )


def test_if_season_information_is_added_to_test_df(
    test_points_with_seasonal_stac: gpd.GeoDataFrame,
):
    assert "season" in test_points_with_seasonal_stac.columns


def test_if_season_information_in_test_df_is_correct(
    test_points_with_seasonal_stac: gpd.GeoDataFrame,
):
    assert sorted(np.unique(test_points_with_seasonal_stac["season"].values)) == [
        "autumn",
        "spring",
        "summer",
        "winter",
    ]


def test_if_df_with_seasonal_stac_ids_has_correct_shape(
    test_points_with_seasonal_stac: gpd.GeoDataFrame, sample_test_data: gpd.GeoDataFrame
):
    assert len(test_points_with_seasonal_stac) == 4 * len(sample_test_data)


def test_if_seasonal_stac_df_correctly_duplicates_points(
    test_points_with_seasonal_stac: gpd.GeoDataFrame, sample_test_data: gpd.GeoDataFrame
):
    assert np.all(test_points_with_seasonal_stac["Lat"].value_counts() == 4) and len(
        test_points_with_seasonal_stac["Lat"].value_counts()
    ) == len(sample_test_data)
