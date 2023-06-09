import geopandas as gpd
import pandas as pd
import pytest
from dask.distributed import Client, LocalCluster

import mosaiks.utils as utl


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
def featurization_params():
    """Featurization configuration for testing."""
    params = {
        "coord_set": {
            "coord_set_name": "india_grid_05",
            "sort_points": True,
            "context_cols_to_keep": ["shrid", "Lat", "Lon"],
        },
        "satellite_search_params": {
            "satellite_name": "landsat-8-c2-l2",
            "seasonal": False,
            "year": 2014,
            "search_start": "2013-04-01",
            "search_end": "2014-03-31",
            "stac_output": "least_cloudy",
            "stac_api": "planetary-compute",
        },
        "model": {
            "num_features": 4,
            "kernel_size": 3,
            "batch_size": 1,
            "device": "cpu",
        },
        "dask": {
            "n_concurrent": 8,
            "chunksize": 5,
            "n_workers": 4,
            "threads_per_worker": 4,
        },
    }
    return params


@pytest.fixture(scope="session")
def satellite_config():
    """Satellite configuration for testing."""
    satellite_config = {
        "resolution": 53,
        "dtype": "int16",
        "bands": ["SR_B2", "SR_B3", "SR_B6", "SR_B7"],
        "buffer_distance": 1001,
        "min_image_edge": 23,
    }

    return satellite_config


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
