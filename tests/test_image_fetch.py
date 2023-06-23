"""Test data loading and image fetching."""
import math
import os

import geopandas as gpd
import numpy as np
import pytest

import mosaiks.utils as utl
from mosaiks.fetch.images import fetch_image_crop, fetch_image_crop_from_stac_id
from mosaiks.fetch.stacs import fetch_stac_item_from_id, fetch_stac_items

os.environ["USE_PYGEOS"] = "0"


@pytest.fixture(scope="module")
def image_crop(satellite_config: dict):
    """Test image crop."""
    lon, lat, id = (
        80.99266676800818,
        20.696269834517118,
        "LC08_L2SP_143046_20151208_02_T1",
    )
    stac_item = fetch_stac_item_from_id(id)[0]

    return fetch_image_crop(
        lon,
        lat,
        stac_item,
        satellite_config["buffer_distance"],
        satellite_config["bands"],
        satellite_config["resolution"],
    )


@pytest.fixture(scope="module")
def image_crop_from_stac_id(satellite_config: dict):
    """Stac Item fetched from stac ID."""
    lon, lat, id = (
        80.99266676800818,
        20.696269834517118,
        "LC08_L2SP_143046_20151208_02_T1",
    )
    return fetch_image_crop_from_stac_id(
        id,
        lon,
        lat,
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
