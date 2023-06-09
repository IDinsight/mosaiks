"""Test dask code."""
from os import listdir
from pathlib import Path
from shutil import rmtree
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from dask.delayed import Delayed
from dask.distributed import Client

from mosaiks import utils as utl
from mosaiks.dask import (
    get_local_dask_client,
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
    featurization_params: dict,
    satellite_config: dict,
):
    points_gdf = utl.df_w_latlons_to_gdf(sample_test_data)
    client = get_local_dask_client(1, 1)
    model = RCF(
        featurization_params["model"]["num_features"],
        featurization_params["model"]["kernel_size"],
        len(satellite_config["bands"]),
    )
    columns = [
        "feature_%d" % i for i in range(featurization_params["model"]["num_features"])
    ]
    folder_path = Path("tests/data/test_output/")
    folder_path.mkdir(parents=True, exist_ok=True)
    run_queued_futures_pipeline(
        points_gdf,
        client,
        model,
        featurization_params,
        satellite_config,
        columns,
        folder_path,
    )
    num_files = len(listdir(folder_path))
    rmtree(folder_path)
    assert num_files == 2


@pytest.mark.slow
def test_run_batched_delayed_pipeline(
    sample_test_data: pd.DataFrame,
    featurization_params: dict,
    satellite_config: dict,
):
    points_gdf = utl.df_w_latlons_to_gdf(sample_test_data)
    client = get_local_dask_client()
    model = RCF(
        featurization_params["model"]["num_features"],
        featurization_params["model"]["kernel_size"],
        len(satellite_config["bands"]),
    )
    columns = [
        "feature_%d" % i for i in range(featurization_params["model"]["num_features"])
    ]
    folder_path = Path("tests/data/test_output/")
    folder_path.mkdir(parents=True, exist_ok=True)
    run_batched_delayed_pipeline(
        points_gdf,
        client,
        model,
        featurization_params,
        satellite_config,
        columns,
        folder_path,
    )
    num_files = len(listdir(folder_path))
    rmtree(folder_path)

    assert num_files == 2


@pytest.mark.slow
def test_run_unbatched_delayed_pipeline(
    sample_test_data: pd.DataFrame,
    featurization_params: dict,
    satellite_config: dict,
):
    points_gdf = utl.df_w_latlons_to_gdf(sample_test_data)
    client = get_local_dask_client(1, 1)
    model = RCF(
        featurization_params["model"]["num_features"],
        featurization_params["model"]["kernel_size"],
        len(satellite_config["bands"]),
    )
    columns = [
        "feature_%d" % i for i in range(featurization_params["model"]["num_features"])
    ]
    folder_path = Path("tests/data/test_output/")
    folder_path.mkdir(parents=True, exist_ok=True)
    run_unbatched_delayed_pipeline(
        points_gdf,
        client,
        model,
        featurization_params,
        satellite_config,
        columns,
        folder_path,
    )
    num_files = len(listdir(folder_path))
    rmtree(folder_path)

    assert num_files == 2
