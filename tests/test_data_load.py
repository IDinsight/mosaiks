"""Test data loading and stc/image fetching."""
import math

import geopandas as gpd
import numpy as np
import pytest
from pystac.item import Item

import mosaiks.utils as utl
from mosaiks.featurize.stacs import (
    create_data_loader,
    fetch_stac_items,
    sort_by_hilbert_distance,
)


def test_hilbert_distance_sort(sample_test_data: gpd.GeoDataFrame):
    """Check that sorting by hilbert distance works."""
    points_gdf = utl.df_w_latlons_to_gdf(sample_test_data)
    points_gdf_with_hilbert = sort_by_hilbert_distance(points_gdf)

    assert len(points_gdf_with_hilbert) == len(points_gdf)
    assert "hilbert_distance" in points_gdf_with_hilbert.columns
    assert np.all(
        points_gdf_with_hilbert["hilbert_distance"].sort_values()
        == points_gdf_with_hilbert["hilbert_distance"]
    )
    assert np.all(points_gdf_with_hilbert["hilbert_distance"] >= 0)


@pytest.mark.parametrize(
    "test_data",
    [
        pytest.lazy_fixture("sample_test_data"),
        pytest.lazy_fixture("sample_test_null_data"),
    ],
)
def test_stac_item_fetch(test_data: gpd.GeoDataFrame):
    """Check if stac references can be fetched."""
    points_gdf = utl.df_w_latlons_to_gdf(test_data)
    points_gdf_with_stac = fetch_stac_items(
        points_gdf, "landsat-8-c2-l2", "2013-04-01", "2014-03-31", "planetary-compute"
    )

    assert len(points_gdf_with_stac) == len(points_gdf)
    assert hasattr(points_gdf_with_stac, "stac_item")
    assert type(points_gdf_with_stac["stac_item"].tolist()[0] == Item)
    if points_gdf["Lat"].isnull().all():
        assert points_gdf_with_stac["stac_item"].isnull().all()


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_data",
    [
        pytest.lazy_fixture("sample_test_data"),
        pytest.lazy_fixture("sample_test_null_data"),
    ],
)
def test_data_loader(test_data: gpd.GeoDataFrame, satellite_config: dict):
    """Check data loader creation and image fetch."""
    points_gdf = utl.df_w_latlons_to_gdf(test_data)
    points_gdf_with_stac = fetch_stac_items(
        points_gdf, "landsat-8-c2-l2", "2013-04-01", "2014-03-31", "planetary-compute"
    )
    data_loader = create_data_loader(points_gdf_with_stac, satellite_config, 1)

    bands = len(satellite_config["bands"])
    buffer = satellite_config["buffer_distance"]
    resolution = satellite_config["resolution"]
    image_size = math.ceil(buffer * 2 / resolution)

    image = data_loader.dataset.__getitem__(5)

    assert len(data_loader) == len(points_gdf_with_stac)
    if image is not None:
        assert image.shape == (bands, image_size + 1, image_size + 1)
        assert image.min() == 0.0 and image.max() == 1.0


# TODO: test seasonal image fetch
