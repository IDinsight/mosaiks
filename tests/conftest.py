import geopandas as gpd
import pandas as pd
import pytest
from dask.distributed import Client, LocalCluster

import mosaiks.utils as utl


@pytest.fixture(scope="session")
def sample_test_data():
    df = pd.read_csv("tests/data/test_points.csv")
    geometries = gpd.points_from_xy(x=df["Lon"], y=df["Lat"])
    return gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")


@pytest.fixture(scope="session")
def featurization_params():
    return utl.load_yaml_config("featurisation.yaml")


@pytest.fixture(scope="session")
def satellite_config(featurization_params):
    satellite_config = utl.load_yaml_config("satellite_config.yaml")
    return satellite_config[
        featurization_params["satellite_search_params"]["satellite_name"]
    ]


@pytest.fixture(scope="session")
def local_cluster_client():
    cluster = LocalCluster(n_workers=2, processes=True, threads_per_worker=4)
    client = Client(cluster)
    return client
