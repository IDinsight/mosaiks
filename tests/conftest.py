import os

os.environ["USE_PYGEOS"] = "0"  # must happen before geopandas import

import geopandas as gpd
import pandas as pd
import pytest
from dask.distributed import Client, LocalCluster


@pytest.fixture(scope="session")
def sample_test_data():
    """Sample test data to use with tests."""
    df = pd.read_csv("tests/data/test_points.csv")
    geometries = gpd.points_from_xy(x=df["Lon"], y=df["Lat"])
    return gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")


@pytest.fixture(scope="session")
def sample_test_null_data():
    """Sample test data with null points to use with tests."""
    df = pd.read_csv("tests/data/test_points_null.csv")
    geometries = gpd.points_from_xy(x=df["Lon"], y=df["Lat"])
    return gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")


@pytest.fixture(scope="session")
def config_dict():
    """Featurization configuration for testing."""
    params = {
        "satellite_name": "landsat-8-c2-l2",
        "image_resolution": 53,
        "image_dtype": "int16",
        "image_bands": ["SR_B2", "SR_B3", "SR_B6", "SR_B7"],
        "image_width": 2000,
        "min_image_edge": 23,
        "sort_points_by_hilbert_distance": True,
        "datetime": "2013",
        "image_composite_method": "least_cloudy",
        "stac_api": "planetary-compute",
        "n_mosaiks_features": 4,
        "mosaiks_kernel_size": 3,
        "mosaiks_batch_size": 1,
        "model_device": "cpu",
        "dask_client_type": "local",
        "dask_n_concurrent_tasks": 1,
        "dask_chunksize": 5,
        "dask_n_workers": 1,
        "dask_threads_per_worker": 1,
    }
    return params


@pytest.fixture(scope="session")
def local_cluster_client():
    """Local cluster client for testing."""
    cluster = LocalCluster(n_workers=2, processes=True, threads_per_worker=4)
    client = Client(cluster)
    return client


def test_column_correctness(sample_test_data):
    """
    Check required columns exist in test data
    """
    assert all(x in sample_test_data.columns for x in ["Lon", "Lat", "geometry"])
