import geopandas as gpd
import pandas as pd
import pytest
from dask import delayed
from dask.distributed import as_completed

import mosaiks.utils as utl
from mosaiks.featurize import *


@delayed
def partition_run(
    df, satellite_config, featurization_params, mosaiks_column_names, model, device
):

    data_loader = create_data_loader(
        df, satellite_config, featurization_params["batch_size"]
    )
    X_features = create_features(
        data_loader,
        featurization_params["num_features"],
        len(df),
        model,
        device,
        satellite_config["min_image_edge"],
    )

    df = pd.DataFrame(X_features, index=df.index.copy(), columns=mosaiks_column_names)

    return df


def test_column_correctness(sample_test_data):
    assert all(x in sample_test_data.columns for x in ["Lon", "Lat", "geometry"])


def test_image_fetch(
    sample_test_data, featurization_params, satellite_config, local_cluster_client
):
    points_gdf_with_stac = fetch_image_refs(
        sample_test_data,
        featurization_params["dask"]["n_partitions"],
        featurization_params["satellite_search_params"],
    )

    mosaiks_column_names = [
        f"mosaiks_{i}" for i in range(featurization_params["num_features"])
    ]

    partitions = points_gdf_with_stac.to_delayed()

    model = RCF(
        featurization_params["num_features"],
        featurization_params["kernel_size"],
        len(satellite_config["bands"]),
    )

    i = 0
    p = partitions[i]
    f = partition_run(
        p,
        satellite_config,
        featurization_params,
        mosaiks_column_names,
        model,
        featurization_params["device"],
        dask_key_name=f"run_{i}",
    )
    df_future = local_cluster_client.compute(f)
    for f in as_completed([df_future]):
        df = f.result()

    # n features
    assert len(df.columns) == 4000
    # non negative
    assert df.min().min() >= 0
