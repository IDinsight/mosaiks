import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac_client
import pytest
from dask import delayed
from dask.distributed import as_completed
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
    image_size = math.floor(buffer * 2 / resolution)

    image = data_loader.dataset.__getitem__(5)

    assert len(data_loader) == len(points_gdf_with_stac)
    if image is not None:
        assert image.shape == (bands, image_size + 1, image_size + 1)
        assert image.min() == 0.0 and image.max() == 1.0


# @delayed
# def partition_run(
#     df, satellite_config, featurization_params, mosaiks_column_names, model, device
# ):

#     data_loader = create_data_loader(
#         df, satellite_config, featurization_params["model"]["batch_size"]
#     )
#     X_features = create_features(
#         data_loader,
#         featurization_params["model"]["num_features"],
#         len(df),
#         model,
#         device,
#         satellite_config["min_image_edge"],
#     )

#     df = pd.DataFrame(X_features, index=df.index.copy(), columns=mosaiks_column_names)

#     return df


# def test_image_fetch(
#     sample_test_data, featurization_params, satellite_config, local_cluster_client
# ):
#     """
#     Check images can be fetched, features generated and basic summary stats are as expected
#     """
#     points_gdf_with_stac = fetch_image_refs(
#         sample_test_data,
#         featurization_params["dask"]["n_partitions"],
#         featurization_params["satellite_search_params"],
#     )

#     mosaiks_column_names = [
#         f"mosaiks_{i}" for i in range(featurization_params["model"]["num_features"])
#     ]

#     partitions = points_gdf_with_stac.to_delayed()

#     model = RCF(
#         featurization_params["model"]["num_features"],
#         featurization_params["model"]["kernel_size"],
#         len(satellite_config["bands"]),
#     )

#     i = 0
#     p = partitions[i]
#     f = partition_run(
#         p,
#         satellite_config,
#         featurization_params,
#         mosaiks_column_names,
#         model,
#         featurization_params["model"]["device"],
#         dask_key_name=f"run_{i}",
#     )
#     df_future = local_cluster_client.compute(f)
#     for f in as_completed([df_future]):
#         df = f.result()

#     # n features
#     assert len(df.columns) == 4000
#     # non negative
#     assert df.min().min() >= 0
