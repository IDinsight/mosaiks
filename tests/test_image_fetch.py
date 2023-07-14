"""Test data loading and image fetching."""
import math

import numpy as np
import pytest

from mosaiks.fetch.images import fetch_image_crop, fetch_image_crop_from_stac_id
from mosaiks.fetch.stacs import fetch_stac_item_from_id


@pytest.fixture(scope="module")
def image_crop(config_dict: dict):
    """Test image crop."""
    lon, lat, id = (
        80.99266676800818,
        20.696269834517118,
        "LC08_L2SP_143046_20151208_02_T1",
    )
    stac_items = fetch_stac_item_from_id([id])

    return fetch_image_crop(
        lon,
        lat,
        stac_items,
        config_dict["image_width"],
        config_dict["image_bands"],
        config_dict["image_resolution"],
    )


@pytest.fixture(scope="module")
def image_crop_from_stac_id(config_dict: dict):
    """Stac Item fetched from stac ID."""
    lon, lat, id = (
        80.99266676800818,
        20.696269834517118,
        ["LC08_L2SP_143046_20151208_02_T1"],
    )
    return fetch_image_crop_from_stac_id(
        id,
        lon,
        lat,
        config_dict["image_width"],
        config_dict["image_bands"],
        config_dict["image_resolution"],
        config_dict["image_dtype"],
        config_dict["image_composite_method"],
        True,
        "planetary-compute",
    )


@pytest.fixture(scope="module")
def image_crop_from_nans(config_dict: dict):
    """Test image crop."""
    lon, lat, id = (np.nan, np.nan, None)
    stac_items = fetch_stac_item_from_id([id])

    return fetch_image_crop(
        lon,
        lat,
        stac_items,
        config_dict["image_width"],
        config_dict["image_bands"],
        config_dict["image_resolution"],
    )


@pytest.mark.parametrize(
    "test_image_crop",
    [
        pytest.lazy_fixture("image_crop"),
        pytest.lazy_fixture("image_crop_from_stac_id"),
        pytest.lazy_fixture("image_crop_from_nans"),
    ],
)
def test_image_crop_shape(test_image_crop: np.ndarray, config_dict: dict):
    image_size = math.ceil(
        config_dict["image_width"] / config_dict["image_resolution"] + 1
    )
    assert test_image_crop.shape == (
        len(config_dict["image_bands"]),
        image_size,
        image_size,
    )


@pytest.mark.parametrize(
    "test_image_crop",
    [pytest.lazy_fixture("image_crop"), pytest.lazy_fixture("image_crop_from_stac_id")],
)
def test_if_image_crop_is_normalised(test_image_crop: np.ndarray):
    assert np.nanmin(test_image_crop) >= 0 and np.nanmax(test_image_crop) <= 1


def test_if_image_crop_for_null_ids_are_nans(image_crop_from_nans: np.ndarray):
    assert ~np.isfinite(image_crop_from_nans).all()
