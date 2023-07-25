"""Test stac item fetching."""
import geopandas as gpd
import numpy as np
import pytest
from pystac.item import Item

import mosaiks.utils as utl
from mosaiks.fetch.stacs import fetch_image_refs, fetch_stac_item_from_id, get_stac_api


# -----Tests for stac item fetching-----
@pytest.fixture(scope="module")
def test_points_with_stac_with_all_composite(sample_test_data: gpd.GeoDataFrame):
    """Sample test data with stac items."""
    points_gdf = utl.df_w_latlons_to_gdf(sample_test_data)
    return fetch_image_refs(
        points_gdf,
        satellite_name="landsat-8-c2-l2",
        datetime=["2013-04-01", "2014-03-31"],
        stac_api_name="planetary-compute",
        image_composite_method="all",
    )


@pytest.fixture(scope="module")
def test_points_with_stac_with_least_cloudy_composite(
    sample_test_data: gpd.GeoDataFrame,
):
    """Sample test data with stac items."""
    points_gdf = utl.df_w_latlons_to_gdf(sample_test_data)
    return fetch_image_refs(
        points_gdf,
        satellite_name="landsat-8-c2-l2",
        datetime=["2013-04-01", "2014-03-31"],
        stac_api_name="planetary-compute",
        image_composite_method="least_cloudy",
    )


@pytest.mark.parametrize(
    "test_points_with_stac",
    [
        pytest.lazy_fixture("test_points_with_stac_with_all_composite"),
        pytest.lazy_fixture("test_points_with_stac_with_least_cloudy_composite"),
    ],
)
def test_if_stac_items_are_added_to_test_df(test_points_with_stac: gpd.GeoDataFrame):
    assert "stac_item" in test_points_with_stac.columns and type(
        test_points_with_stac["stac_item"].tolist()[0] == Item
    )


@pytest.mark.parametrize(
    "test_points_with_stac",
    [
        pytest.lazy_fixture("test_points_with_stac_with_all_composite"),
        pytest.lazy_fixture("test_points_with_stac_with_least_cloudy_composite"),
    ],
)
def test_if_df_with_stac_items_has_correct_shape(
    test_points_with_stac: gpd.GeoDataFrame, sample_test_data: gpd.GeoDataFrame
):
    assert len(test_points_with_stac) == len(sample_test_data)


# -----Tests for stac item fetching with null data-----
@pytest.fixture(scope="module")
def test_points_with_stac_null(sample_test_null_data: gpd.GeoDataFrame):
    """Sample test data with stac items."""
    points_gdf = utl.df_w_latlons_to_gdf(sample_test_null_data)
    return fetch_image_refs(
        points_gdf,
        satellite_name="landsat-8-c2-l2",
        datetime=["2013-04-01", "2014-03-31"],
        stac_api_name="planetary-compute",
        image_composite_method="least_cloudy",
    )


def test_if_stac_items_are_added_to_test_null_df(
    test_points_with_stac_null: gpd.GeoDataFrame,
):
    assert "stac_item" in test_points_with_stac_null.columns


def test_if_null_df_with_stac_items_has_correct_shape(
    test_points_with_stac_null: gpd.GeoDataFrame, sample_test_data: gpd.GeoDataFrame
):
    assert len(test_points_with_stac_null) == len(sample_test_data)


def test_if_stac_items_from_test_null_df_are_empty(
    test_points_with_stac_null: gpd.GeoDataFrame,
):
    assert test_points_with_stac_null["stac_item"].isnull().all()


# -----Tests for stac item fetch from stac ID-----
@pytest.fixture(scope="module")
def stac_item_from_stac_id():
    """Stac Item fetched from stac ID."""
    id = ["LC08_L2SP_143046_20151208_02_T1"]
    return fetch_stac_item_from_id(id, "planetary-compute")


def test_if_stac_item_is_returned_from_id(stac_item_from_stac_id: Item):
    assert type(stac_item_from_stac_id[0]) == Item


def test_if_stac_item_list_has_correct_shape(stac_item_from_stac_id: Item):
    assert len(stac_item_from_stac_id) >= 1


# -----Tests for stac item fetch for null stac ID-----
@pytest.fixture(scope="module")
def stac_item_from_null_stac_id():
    """Stac Item fetched from stac ID."""
    id = [None]
    return fetch_stac_item_from_id(id, "planetary-compute")


def test_if_null_stac_item_is_returned_from_null_id(stac_item_from_null_stac_id: Item):
    assert stac_item_from_null_stac_id[0] is None


def test_if_null_stac_item_list_has_correct_shape(stac_item_from_null_stac_id: Item):
    assert len(stac_item_from_null_stac_id) == 1
