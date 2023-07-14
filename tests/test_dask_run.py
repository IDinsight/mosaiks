"""Test dask code."""
from os import listdir
from pathlib import Path
from shutil import rmtree
from typing import Generator

import pandas as pd
import pytest

from mosaiks import utils as utl
from mosaiks.dask import (
    get_local_dask_cluster_and_client,
    get_partitions_generator,
    run_batched_delayed_pipeline,
    run_queued_futures_pipeline,
    run_unbatched_delayed_pipeline,
)
from mosaiks.featurize import RCF


# -----Test partitions generator-----
def test_if_get_partitions_generator_returns_generator(sample_test_data: pd.DataFrame):
    test_points_gdf = utl.df_w_latlons_to_gdf(sample_test_data)
    partitions_generator = get_partitions_generator(test_points_gdf, 5)
    assert isinstance(partitions_generator, Generator)


def test_if_get_partitions_generator_returns_correct_number_of_partitions(
    sample_test_data: pd.DataFrame,
):
    test_points_gdf = utl.df_w_latlons_to_gdf(sample_test_data)
    partitions_generator = get_partitions_generator(test_points_gdf, 5)
    assert len(list(partitions_generator)) == 2


def test_if_get_partitions_generator_returns_correct_type_of_partitions(
    sample_test_data: pd.DataFrame,
):
    test_points_gdf = utl.df_w_latlons_to_gdf(sample_test_data)
    partitions_generator = get_partitions_generator(test_points_gdf, 5)
    assert isinstance(next(partitions_generator), pd.DataFrame)


def test_if_get_partitions_generator_returns_correct_number_of_points_in_partitions(
    sample_test_data: pd.DataFrame,
):
    test_points_gdf = utl.df_w_latlons_to_gdf(sample_test_data)
    partitions_generator = get_partitions_generator(test_points_gdf, 5)
    assert len(next(partitions_generator)) == 5


# -----Test local dask client-----


@pytest.mark.slow
def test_run_queued_futures(
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
    folder_path = Path("tests/data/test_output_futures/")
    folder_path.mkdir(parents=True, exist_ok=True)
    cluster, client = get_local_dask_cluster_and_client(1, 1)

    run_queued_futures_pipeline(
        points_gdf,
        client,
        model,
        satellite_name=config_dict["satellite_name"],
        image_resolution=config_dict["image_resolution"],
        image_dtype=config_dict["image_dtype"],
        image_bands=config_dict["image_bands"],
        image_width=config_dict["image_width"],
        min_image_edge=config_dict["min_image_edge"],
        sort_points=config_dict["sort_points_by_hilbert_distance"],
        seasonal=config_dict["seasonal"],
        year=config_dict["year"],
        search_start=config_dict["search_start"],
        search_end=config_dict["search_end"],
        image_composite_method=config_dict["image_composite_method"],
        stac_api_name=config_dict["stac_api"],
        num_features=config_dict["n_mosaiks_features"],
        batch_size=config_dict["mosaiks_batch_size"],
        device=config_dict["model_device"],
        col_names=columns,
        n_concurrent=config_dict["dask_n_concurrent_tasks"],
        chunksize=config_dict["dask_chunksize"],
        save_folder_path=folder_path,
    )
    num_files = len(listdir(folder_path))
    rmtree(folder_path)
    client.shutdown()
    assert num_files == 2


@pytest.mark.slow
def test_run_batched_delayed_pipeline(
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
    cluster, client = get_local_dask_cluster_and_client(1, 1)

    run_batched_delayed_pipeline(
        points_gdf,
        client,
        model,
        satellite_name=config_dict["satellite_name"],
        image_resolution=config_dict["image_resolution"],
        image_dtype=config_dict["image_dtype"],
        image_bands=config_dict["image_bands"],
        image_width=config_dict["image_width"],
        min_image_edge=config_dict["min_image_edge"],
        sort_points=config_dict["sort_points_by_hilbert_distance"],
        seasonal=config_dict["seasonal"],
        year=config_dict["year"],
        search_start=config_dict["search_start"],
        search_end=config_dict["search_end"],
        image_composite_method=config_dict["image_composite_method"],
        stac_api_name=config_dict["stac_api"],
        num_features=config_dict["n_mosaiks_features"],
        batch_size=config_dict["mosaiks_batch_size"],
        device=config_dict["model_device"],
        n_concurrent=config_dict["dask_n_concurrent_tasks"],
        chunksize=config_dict["dask_chunksize"],
        col_names=columns,
        save_folder_path=folder_path,
    )
    num_files = len(listdir(folder_path))
    rmtree(folder_path)
    client.shutdown()

    assert num_files == 2


@pytest.mark.slow
def test_run_unbatched_delayed_pipeline(
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
    folder_path = Path("tests/data/test_output_unbatch_delayed/")
    folder_path.mkdir(parents=True, exist_ok=True)
    cluster, client = get_local_dask_cluster_and_client(1, 1)

    run_unbatched_delayed_pipeline(
        points_gdf,
        client,
        model,
        config_dict["sort_points_by_hilbert_distance"],
        config_dict["satellite_name"],
        config_dict["search_start"],
        config_dict["search_end"],
        config_dict["stac_api"],
        config_dict["seasonal"],
        config_dict["year"],
        config_dict["image_composite_method"],
        config_dict["n_mosaiks_features"],
        config_dict["model_device"],
        config_dict["min_image_edge"],
        config_dict["mosaiks_batch_size"],
        config_dict["image_width"],
        config_dict["image_bands"],
        config_dict["image_resolution"],
        config_dict["image_dtype"],
        col_names=columns,
        chunksize=config_dict["dask_chunksize"],
        save_folder_path=folder_path,
    )

    num_files = len(listdir(folder_path))
    rmtree(folder_path)
    client.shutdown()

    assert num_files == 2
