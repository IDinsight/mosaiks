"""Test data loading and image fetching."""
import math
import os

import geopandas as gpd
import numpy as np
import pytest

import mosaiks.utils as utl
from mosaiks.fetch.images import fetch_image_crop, fetch_image_crop_from_stac_id
from mosaiks.fetch.stacs import fetch_stac_items, get_stac_api

os.environ["USE_PYGEOS"] = "0"


@pytest.fixture(scope="module")
def image_crop(sample_test_data: gpd.GeoDataFrame, satellite_config: dict):
    """Test image crop."""
    points_gdf = utl.df_w_latlons_to_gdf(sample_test_data)
    point_with_stac = fetch_stac_items(
        points_gdf.iloc[:1],
        "landsat-8-c2-l2",
        "2013-04-01",
        "2014-03-31",
        "planetary-compute",
    )
    return fetch_image_crop(
        point_with_stac.iloc[0]["Lon"],
        point_with_stac.iloc[0]["Lat"],
        point_with_stac.iloc[0]["stac_item"],
        satellite_config["buffer_distance"],
        satellite_config["bands"],
        satellite_config["resolution"],
    )


@pytest.fixture(scope="module")
def image_crop_from_stac_id(sample_test_data: gpd.GeoDataFrame, satellite_config: dict):
    """Stac Item fetched from stac ID."""
    stac_api = get_stac_api("planetary-compute")
    id = stac_api.search(
        collections=["landsat-8-c2-l2"],
        intersects=sample_test_data["geometry"][0],
        datetime=["2015-01-01", "2015-12-31"],
        query={"eo:cloud_cover": {"lt": 10}},
        limit=500,
    ).item_collection_as_dict()["features"][0]["id"]
    return fetch_image_crop_from_stac_id(
        id,
        sample_test_data["Lon"][0],
        sample_test_data["Lat"][0],
        satellite_config,
        "planetary-compute",
    )


@pytest.mark.parametrize(
    "test_image_crop",
    [pytest.lazy_fixture("image_crop"), pytest.lazy_fixture("image_crop_from_stac_id")],
)
def test_image_crop_shape(test_image_crop: np.ndarray, satellite_config: dict):
    image_size = math.ceil(
        satellite_config["buffer_distance"] * 2 / satellite_config["resolution"]
    )
    assert test_image_crop.shape == (
        len(satellite_config["bands"]),
        image_size + 1,
        image_size + 1,
    )


@pytest.mark.parametrize(
    "test_image_crop",
    [pytest.lazy_fixture("image_crop"), pytest.lazy_fixture("image_crop_from_stac_id")],
)
def test_if_image_crop_is_normalised(test_image_crop: np.ndarray):
    assert np.nanmin(test_image_crop) >= 0 and np.nanmax(test_image_crop) <= 1
