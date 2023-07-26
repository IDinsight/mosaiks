"""Test dask code."""
from os import listdir
from pathlib import Path
from shutil import rmtree

import pandas as pd
import pytest

from mosaiks import utils as utl
from mosaiks.featurize import RCF
from mosaiks.pipeline.parallel import (
    _run_batched_pipeline,
    get_local_dask_cluster_and_client,
)


# -----Test local dask client-----
@pytest.mark.slow
def test__run_batched_pipeline(
    sample_test_data: pd.DataFrame,
    config_dict: dict,
):
    points_gdf = utl.df_w_latlons_to_gdf(sample_test_data)

    model = RCF(
        config_dict["n_mosaiks_features"],
        config_dict["mosaiks_kernel_size"],
        len(config_dict["image_bands"]),
    )
    columns = ["feature_%d" % i for i in range(config_dict["n_mosaiks_features"])]
    folder_path = Path("tests/data/test_output_batch_delayed/")
    folder_path.mkdir(parents=True, exist_ok=True)
    cluster, client = get_local_dask_cluster_and_client(2, 2)

    _run_batched_pipeline(
        points_gdf,
        client,
        model,
        satellite_name=config_dict["satellite_name"],
        image_resolution=config_dict["image_resolution"],
        image_dtype=config_dict["image_dtype"],
        image_bands=config_dict["image_bands"],
        image_width=config_dict["image_width"],
        min_image_edge=config_dict["min_image_edge"],
        sort_points_by_hilbert_distance=config_dict["sort_points_by_hilbert_distance"],
        datetime=config_dict["datetime"],
        image_composite_method=config_dict["image_composite_method"],
        stac_api_name=config_dict["stac_api_name"],
        num_features=config_dict["n_mosaiks_features"],
        device=config_dict["model_device"],
        n_concurrent_tasks=config_dict["dask_n_concurrent_tasks"],
        chunksize=config_dict["dask_chunksize"],
        col_names=columns,
        temp_folderpath=folder_path,
    )

    num_files = len(listdir(folder_path))
    rmtree(folder_path)
    client.close()
    cluster.close()

    assert num_files == 2
