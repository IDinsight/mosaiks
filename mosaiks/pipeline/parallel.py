import logging
import shutil
from datetime import datetime as dt
from pathlib import Path
from typing import List, Optional

import dask.delayed
import dask_geopandas as dask_gpd
import geopandas as gpd
import numpy as np
import pandas as pd
import torch.nn as nn
from dask.distributed import Client, LocalCluster, as_completed

import mosaiks.utils as utl
from mosaiks.pipeline.standard import run_pipeline

__all__ = [
    "run_parallel_pipeline",
    "get_local_dask_cluster_and_client",
]


def run_parallel_pipeline(
    points_gdf: gpd.GeoDataFrame,
    model: nn.Module,
    satellite_name: str,
    image_resolution: int,
    image_dtype: str,
    image_bands: List[str],
    image_width: int,
    min_image_edge: int,
    datetime: str or list[str] or callable,
    image_composite_method: str,
    stac_api_name: str,
    n_mosaiks_features: int,
    model_device: str,
    mosaiks_col_names: list,
    n_concurrent_tasks: int,
    chunksize: int,
    client: Optional[Client],
    n_workers: Optional[int],
    threads_per_worker: Optional[int],
    sort_points_by_hilbert_distance: bool,
) -> pd.DataFrame:
    """
    For a given DataFrame of coordinate points, this function runs the `run_pipeline()`
    function on batches of datapoints in parallel using Dask.

    Parameters:
    -----------
    points_gdf : GeoDataFrame of points to be featurized.
    model: PyTorch model to be used for featurization.
    satellite_name: name of the satellite to use. Options are "landsat-8-c2-l2" or "sentinel-2-l2a".
    image_resolution: resolution of the satellite images in meters. Set depending on satellite.
    image_dtype: data type of the satellite images. Suggested "int16". All options - "int16", "int32", and "float"
    image_bands: list of bands to use for the satellite images. Suggested ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]. For options, read the satellite docs
    image_width: Desired width of the image to be fetched (in meters). Suggested 3000m for landsat.
    min_image_edge: minimum image edge in meters. Suggested 1000.
    datetime : date/times for fetching satellite images. See STAC API docs for `pystac.Client.search`'s `datetime` parameter for more details
    image_composite_method: how to composite multiple images for same GPS location. Options are "least_cloudy" (pick least cloudy image) or "all" (get all images and average across them). Suggested "least_cloudy" for speed.
    stac_api_name: which STAC API to use. Options are "planetary-compute" or "earth-search". Suggested "planetary-compute".
    n_mosaiks_features: number of mosaiks features to generate. Suggested 1000-4000.
    model_device: compute device for mosaiks model. Options are "cpu" or "cuda".
    mosaiks_col_names: column names for the mosaiks features.
    n_concurrent_tasks: number of concurrent tasks to run in Dask. Suggested None, which sets the total number of tasks to number of threads.
    chunksize: number of datapoints per data partition in Dask. Suggested 500.
    client : Dask client. If None, create a local cluster and client.
    threads_per_worker : Number of threads per worker. If None, let Dask decide (uses all available threads per core).
    sort_points_by_hilbert_distance: Whether to sort points by Hilbert distance before partitioning them. Suggested True.

    Returns
    --------
    DataFrame
    """

    # create a temporary directory
    date_time_now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    temp_dir = Path.cwd() / f"dask_{date_time_now}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Temporary directory: {temp_dir}")

    try:
        # Create a temporary dask client if one not given
        if client is None:
            logging.info("Creating a local Dask cluster and client...")
            cluster, client = get_local_dask_cluster_and_client(
                n_workers=n_workers, threads_per_worker=threads_per_worker
            )
            temp_client = True
        else:
            temp_client = False

        failed_partitions = _run_batched_pipeline(
            points_gdf=points_gdf,
            client=client,
            model=model,
            satellite_name=satellite_name,
            image_resolution=image_resolution,
            image_dtype=image_dtype,
            image_bands=image_bands,
            image_width=image_width,
            min_image_edge=min_image_edge,
            sort_points_by_hilbert_distance=sort_points_by_hilbert_distance,
            datetime=datetime,
            image_composite_method=image_composite_method,
            stac_api_name=stac_api_name,
            num_features=n_mosaiks_features,
            device=model_device,
            n_concurrent_tasks=n_concurrent_tasks,
            chunksize=chunksize,
            col_names=mosaiks_col_names,
            temp_folderpath=temp_dir,
        )
        if failed_partitions:
            logging.warn(f"Failed partitions: {failed_partitions}.")

        # IMPORTANT: Close dask client
        if temp_client:
            client.close()
            cluster.close()

        # Load checkpoint files and combine
        logging.info("Loading and combining checkpoint files...")
        checkpoint_filenames = utl.get_filtered_filenames(
            folder_path=temp_dir, prefix="df_"
        )
        combined_df = utl.load_and_combine_dataframes(
            folder_path=temp_dir, filenames=checkpoint_filenames
        )

    finally:
        # Delete temporary directory
        shutil.rmtree(temp_dir)

    return combined_df


def _run_batched_pipeline(
    points_gdf: gpd.GeoDataFrame,
    client: Client,
    model: nn.Module,
    satellite_name: str,
    image_resolution: int,
    image_dtype: str,
    image_bands: list[str],
    image_width: int,
    min_image_edge: int,
    sort_points_by_hilbert_distance: bool,
    datetime: str or list[str] or callable,
    image_composite_method: bool,
    stac_api_name: str,
    num_features: int,
    device: str,
    n_concurrent_tasks: int,
    chunksize: int,
    col_names: list[str],
    temp_folderpath: str,
    partition_ids: list[int] = None,
) -> list[int]:
    """
    Run partitions in batches and save the result for each partition
    to a parquet file.

    Parameters
    -----------
    points_gdf : GeoDataFrame of points to be featurized.
    client : Dask client.
    model : PyTorch model to be used for featurization.
    satellite_name : Name of satellite to be used for featurization.
    image_resolution : Resolution of satellite images to be used for featurization.
    image_dtype : Data type of satellite images to be used for featurization.
    image_bands : List of satellite image bands to be used for featurization.
    image_width : Desired width of the image to be fetched (in meters).
    min_image_edge : Minimum image edge size.
    sort_points_by_hilbert_distance: Whether to sort points by Hilbert distance before partitioning them. Suggested True.
    datetime: date/times for fetching satellite images. Same as datetime parameter in pystac.Client.search.
    image_composite_method : Mosaic composite to be used for featurization.
    stac_api_name : Name of STAC API to be used for satellite image search.
    num_features : number of mosaiks features.
    device : Device to be used for featurization.
    col_names : List of column names to be used for saving the features.
    temp_folderpath : Path to folder where features will be saved.
    partition_ids : Optional. List of partition indexes to be used for featurization.

    Returns
    --------
    list of failed partition ids
    """
    # make delayed gdf partitions
    dask_gdf = _get_dask_gdf(
        points_gdf,
        chunksize,
        sort_points_by_hilbert_distance,
    )
    partitions = dask_gdf.to_delayed()

    # only use the partitions specified by partition_ids if given
    if partition_ids is None:
        n_partitions = len(partitions)
        partition_ids = list(range(n_partitions))
    else:
        partitions = [partitions[i] for i in partition_ids]
        n_partitions = len(partitions)

    # if n_concurrent_tasks is not specified, use all available threads
    if n_concurrent_tasks is None:
        n_concurrent_tasks = sum(client.nthreads().values())
    # if there are less partitions to run than concurrent tasks, run all partitions
    n_concurrent_tasks = min(n_partitions, n_concurrent_tasks)

    logging.info(
        f"Running {n_partitions} partitions in batches of {n_concurrent_tasks} at a time."
    )

    failed_ids = []
    checkpoint_indices = list(np.arange(0, n_partitions, n_concurrent_tasks)) + [
        n_partitions
    ]
    for p_start_id, p_end_id in zip(checkpoint_indices[:-1], checkpoint_indices[1:]):
        now = dt.now().strftime("%d-%b %H:%M:%S")
        logging.info(f"{now} Running batch: {p_start_id} to {p_end_id - 1}")

        batch_indices = list(range(p_start_id, p_end_id))
        batch_p_ids = [partition_ids[i] for i in batch_indices]
        batch_partitions = [partitions[i] for i in batch_indices]

        failed_ids += _run_batch(
            partitions=batch_partitions,
            partition_ids=batch_p_ids,
            client=client,
            model=model,
            satellite_name=satellite_name,
            image_resolution=image_resolution,
            image_dtype=image_dtype,
            image_bands=image_bands,
            image_width=image_width,
            min_image_edge=min_image_edge,
            datetime=datetime,
            image_composite_method=image_composite_method,
            stac_api_name=stac_api_name,
            num_features=num_features,
            device=device,
            col_names=col_names,
            temp_folderpath=temp_folderpath,
        )

    return failed_ids


def _run_batch(
    partitions: list,
    partition_ids: list,
    client: Client,
    model: nn.Module,
    satellite_name: str,
    image_resolution: int,
    image_dtype: str,
    image_bands: list[str],
    image_width: int,
    min_image_edge: int,
    datetime: str or list[str] or callable,
    image_composite_method: str,
    stac_api_name: str,
    num_features: int,
    device: str,
    col_names: list,
    temp_folderpath: str,
) -> list[int]:
    """
    Run a batch of partitions and save the result for each partition to a parquet file.

    Parameters
    ----------
    partitions :List of dataframes to process.
    partition_ids : List containing IDs corresponding to the partitions passed (to be
        used for naming saved files and reference in case of failure).
    client : Dask client.
    model : PyTorch model to be used for featurization.
    satellite_name : Name of satellite to be used for featurization.
    image_resolution : Resolution of satellite images to be used for featurization.
    image_dtype : Data type of satellite images to be used for featurization.
    image_bands : List of satellite image bands to be used for featurization.
    image_width : Desired width of the image to be fetched (in meters).
    min_image_edge : Minimum image edge size.
    datetime : date/times for fetching satellite images. See STAC API docs for `pystac.Client.search`'s `datetime` parameter for more details
    image_composite_method : Mosaic composite to be used for featurization.
    stac_api_name : Name of STAC API to be used for satellite image search.
    num_features : number of mosaiks features.
    device : Device to be used for featurization.
    col_names : List of column names to be used for the output dataframe.
    temp_folderpath : Path to folder where features will be saved.

    Returns
    -------
    failed_ids : List of partition labels that failed to be featurized.
    """

    failed_ids = []
    delayed_tasks = []

    # collect delayed tasks
    for partition_id, partition in zip(partition_ids, partitions):
        str_id = str(partition_id).zfill(3)  # makes 1 into '001'

        # this can be swapped for delayed_pipeline(...)
        delayed_task = dask.delayed(run_pipeline)(
            points_gdf=partition,
            model=model,
            satellite_name=satellite_name,
            image_resolution=image_resolution,
            image_dtype=image_dtype,
            image_bands=image_bands,
            image_width=image_width,
            min_image_edge=min_image_edge,
            datetime=datetime,
            image_composite_method=image_composite_method,
            stac_api_name=stac_api_name,
            num_features=num_features,
            device=device,
            col_names=col_names,
            output_filepath=temp_folderpath / f"df_{str_id}.parquet",
            dask_key_name=f"features_{str_id}",
            return_df=False,
        )
        delayed_tasks.append(delayed_task)

    # compute delayed tasks
    futures = client.compute(delayed_tasks)

    # check for errors
    for completed_future in as_completed(futures):
        if completed_future.status == "error":
            f_key = completed_future.key
            partition_id = int(f_key.split("features_")[1])
            failed_ids.append(partition_id)

    return failed_ids


def get_local_dask_cluster_and_client(
    n_workers: Optional[int] = None,
    threads_per_worker: Optional[int] = None,
) -> tuple:
    """
    Get a local dask cluster and client.

    Parameters
    -----------
    threads_per_worker : Number of threads per worker. If None, let Dask decide (uses all available threads per core).
    dask_sort_points_by_hilbert_distance: Whether to sort points by Hilbert distance before partitioning them. Suggested True.

    Returns
    --------
    Dask cluster and client.
    """

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        silence_logs=logging.ERROR,
    )
    client = Client(cluster)

    # get number of workers and threads actually used by the client
    logging.info(
        f"\n\nDask client created with "
        f"{len(client.scheduler_info()['workers'])} workers and "
        f"{sum(client.nthreads().values())} total threads.\n"
        f"Dashboard link: {client.dashboard_link}\n\n"
    )

    return cluster, client


def _get_dask_gdf(
    points_gdf: gpd.GeoDataFrame,
    chunksize: int,
    sort_points_by_hilbert_distance: bool = True,
) -> dask_gpd.GeoDataFrame:
    """
    Split the gdf up by the given chunksize. To be used to create Dask Delayed jobs.

    Parameters
    ----------
    points_dgdf : A GeoDataFrame with a column named "geometry" containing shapely
        Point objects.
    chunksize : The number of points per partition to use creating the Dask
        GeoDataFrame.
    sort_points_by_hilbert_distance : Whether to sort the points by their Hilbert distance before splitting the dataframe into partitions.

    Returns
    -------
    points_dgdf: Dask GeoDataFrame split into partitions of size `chunksize`.
    """

    if sort_points_by_hilbert_distance:
        points_gdf = _sort_points_by_hilbert_distance(points_gdf)

    points_dgdf = dask_gpd.from_geopandas(
        points_gdf,
        chunksize=chunksize,
        sort=False,
    )

    logging.info(
        f"Created {points_dgdf.npartitions} partitions of {chunksize} points each."
    )

    return points_dgdf


def _sort_points_by_hilbert_distance(points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sort the points in the GeoDataFrame by their Hilbert distance."""

    ddf = dask_gpd.from_geopandas(points_gdf, npartitions=1)
    hilbert_distance = ddf.hilbert_distance().compute()
    points_gdf["hilbert_distance"] = hilbert_distance
    points_gdf = points_gdf.sort_values("hilbert_distance")

    return points_gdf
