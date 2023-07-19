import logging
import math
from datetime import datetime
from typing import Generator, List, Optional

import dask.delayed
import dask_geopandas as dask_gpd
import geopandas as gpd
import numpy as np
import pandas as pd
import torch.nn as nn
from dask import delayed
from dask.distributed import Client, LocalCluster, as_completed, wait

import mosaiks.utils as utl

# for fully-delayed pipeline
from mosaiks.featurize import create_features_from_image_array
from mosaiks.fetch import create_data_loader, fetch_image_refs
from mosaiks.pipeline import run_pipeline

__all__ = [
    "get_local_dask_cluster_and_client",
    "run_pipeline_with_parallelization",
    "run_queued_futures_pipeline",
    "run_batched_pipeline",
    "run_unbatched_delayed_pipeline",
    "delayed_pipeline",
]


def run_pipeline_with_parallelization(
    points_gdf: gpd.GeoDataFrame,
    model: nn.Module,
    satellite_name: str,
    image_resolution: int,
    image_dtype: str,
    image_bands: List[str],
    image_width: int,
    min_image_edge: int,
    seasonal: bool,
    year: int,
    search_start: str,
    search_end: str,
    image_composite_method: str,
    stac_api_name: str,
    n_mosaiks_features: int,
    model_device: str,
    n_concurrent_tasks: int,
    chunksize: int,
    n_workers: Optional[int],
    threads_per_worker: Optional[int],
    sort_points_by_hilbert_distance: bool,
    mosaiks_col_names: list,
    save_folder_path: str = None,
    save_filename: str = "features.csv",
    return_df: bool = True,
) -> pd.DataFrame:  # or None
    """
    For a given DataFrame of coordinate points, this function runs the `run_pipeline()`
    function on batches of datapoints in parallel using Dask.

    Parameters:
    -----------
    points_gdf : GeoDataFrame of points to be featurized.
    model: PyTorch model to be used for featurization.
    satellite_name: name of the satellite to use. Options are "landsat-8-c2-l2" or "sentinel-2-l2a". Defaults to "landsat-8-c2-l2".
    image_resolution: resolution of the satellite images in meters. Defaults to 30.
    image_dtype: data type of the satellite images. Defaults to "int16". All options - "int16", "int32", and "float"
    image_bands: list of bands to use for the satellite images. Defaults to ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]. For options, read the satellite docs
    image_width: Desired width of the image to be fetched (in meters). Default 3000m.
    min_image_edge: minimum image edge in meters. Defaults to 1000.
    seasonal: whether to get seasonal images. Defaults to False.
    year: year to get seasonal images for in format YYYY. Only needed if seasonal = True. Defaults to None.
    search_start: start date for image search in format YYYY-MM-DD. Defaults to "2013-01-01".
    search_end: end date for image search in format YYYY-MM-DD. Defaults to "2013-12-31".
    image_composite_method: how to composite multiple images for same GPS location. Options are "least_cloudy" (pick least cloudy image) or "all" (get all images and average across them). Defaults to "least_cloudy".
    stac_api_name: which STAC API to use. Options are "planetary-compute" or "earth-search". Defaults to "planetary-compute".
    n_mosaiks_features: number of mosaiks features to generate. Defaults to 4000.
    model_device: compute device for mosaiks model. Options are "cpu" or "cuda". Defaults to "cpu".
    n_concurrent_tasks: number of concurrent tasks to run in Dask. Defaults to 8.
    chunksize: number of datapoints per data partition in Dask. Defaults to 500.
    threads_per_worker : Number of threads per worker. If None, let Dask decide (uses all available threads per core).
    sort_points_by_hilbert_distance: Whether to sort points by Hilbert distance before partitioning them. Defaults to True.
    mosaiks_col_names: column names for the mosaiks features. Defaults to None.
    save_folder_path : Path to folder where features will be saved. Default is None.
    save_filename : Name of file where features will be saved. Default is "features.csv".
    return_df : Whether to return the features as a DataFrame. Default is True.

    Returns
    --------
    None or DataFrame

    """
    # TODO - just make this a temp folder inside cwd without any structure
    save_folder_path_temp = utl.make_output_folder_path(
        satellite_name=satellite_name,
        year=search_start.split("-")[0],
        n_mosaiks_features=n_mosaiks_features,
        root_folder=None,
        coords_dataset_name="temp",
    )
    save_folder_path_temp.mkdir(parents=True, exist_ok=True)

    # Create dask client
    cluster, client = get_local_dask_cluster_and_client(
        n_workers=n_workers, threads_per_worker=threads_per_worker
    )

    # Run in batches in parallel
    run_batched_pipeline(
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
        seasonal=seasonal,
        year=year,
        search_start=search_start,
        search_end=search_end,
        image_composite_method=image_composite_method,
        stac_api_name=stac_api_name,
        num_features=n_mosaiks_features,
        device=model_device,
        n_concurrent=n_concurrent_tasks,
        chunksize=chunksize,
        col_names=mosaiks_col_names,
        save_folder_path=save_folder_path_temp,
    )

    # Close dask client
    client.close()
    cluster.close()

    # Load checkpoint files and combine
    logging.info("Loading and combining checkpoint files...")
    checkpoint_filenames = utl.get_filtered_filenames(
        folder_path=save_folder_path_temp, prefix="df_"
    )
    combined_df = utl.load_and_combine_dataframes(
        folder_path=save_folder_path_temp, filenames=checkpoint_filenames
    )

    if save_folder_path is not None:
        utl.save_dataframe(df=combined_df, file_path=save_folder_path / save_filename)

    if return_df:
        return combined_df


def get_local_dask_cluster_and_client(
    n_workers: Optional[int] = None,
    threads_per_worker: Optional[int] = None,
) -> tuple:
    """
    Get a local dask cluster and client.

    Parameters
    -----------
    threads_per_worker : Number of threads per worker. If None, let Dask decide (uses all available threads per core).
    dask_sort_points_by_hilbert_distance: Whether to sort points by Hilbert distance before partitioning them. Defaults to True.

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


def get_partitions_generator(
    points_gdf: gpd.GeoDataFrame,
    chunksize: int,
    sort_points_by_hilbert_distance: bool = False,
) -> Generator[gpd.GeoDataFrame, None, None]:
    """
    Given a GeoDataFrame, this function creates a generator that returns chunksize
    number of rows per iteration.

    To be used for submitting Dask Futures.

    Parameters
    -----------
    points_gdf : GeoDataFrame of points to be featurized.
    chunksize : Number of points to be featurized per iteration.
    sort_points_by_hilbert_distance : Whether to sort the points by their Hilbert distance.

    Returns
    --------
    Generator
    """

    if sort_points_by_hilbert_distance:
        points_gdf = _sort_points_by_hilbert_distance(points_gdf)

    num_chunks = math.ceil(len(points_gdf) / chunksize)

    logging.info(
        f"Distributing {len(points_gdf)} points across {chunksize}-point partitions "
        f"results in {num_chunks} partitions."
    )

    for i in range(num_chunks):
        yield points_gdf.iloc[i * chunksize : (i + 1) * chunksize]


def _sort_points_by_hilbert_distance(points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sort the points in the GeoDataFrame by their Hilbert distance."""

    ddf = dask_gpd.from_geopandas(points_gdf, npartitions=1)
    hilbert_distance = ddf.hilbert_distance().compute()
    points_gdf["hilbert_distance"] = hilbert_distance
    points_gdf = points_gdf.sort_values("hilbert_distance")

    return points_gdf


def run_queued_futures_pipeline(
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
    seasonal: bool,
    year: int,
    search_start: str,
    search_end: str,
    image_composite_method: str,
    stac_api_name: str,
    num_features: int,
    device: str,
    col_names: list,
    n_concurrent: int,
    chunksize: int,
    save_folder_path: str,
) -> None:
    """
    For a given GeoDataFrame of coordinate points, this function partitions it
    and submit each partition to be processed as a Future on the Dask client.

    Initially, only as many partitions are submitted as there are threads. As each
    partition is completed, another partition is submitted to the client.

    Parameters
    -----------
    points_gdf : GeoDataFrame of points to be featurized.
    client : Dask client.
    model : PyTorch model to be used for featurization.
    satellite_name : Name of the satellite to be used for featurization.
    image_resolution : Resolution of the image to be generated.
    image_dtype : Data type of the image to be generated.
    image_bands : List of bands to be used for generating the image.
    image_width : Desired width of the image to be fetched (in meters).
    min_image_edge : Minimum edge length of the image to be generated.
    sort_points_by_hilbert_distance : Whether to sort the points by their Hilbert distance.
    seasonal : Whether to use seasonal imagery.
    year : Year of imagery to be used.
    search_start : Start date of imagery to be used.
    search_end : End date of imagery to be used.
    image_composite_method : Mosaic composite to be used.
    stac_api_name : Name of the STAC API to be used.
    num_features : Number of features to be extracted from the model.
    device : Device to be used for featurization.
    col_names : List of column names to be used for saving the features.
    n_concurrent : Number of concurrent partitions to be submitted to the client.
    chunksize : Number of points to be featurized per partition.
    save_folder_path : Path to folder where features will be saved.

    Returns
    --------
    None
    """

    # make generator for spliting up the data into partitions
    partitions = get_partitions_generator(
        points_gdf,
        chunksize,
        sort_points_by_hilbert_distance,
    )

    # kickoff "n_concurrent" number of tasks. Each of these will be replaced by a new
    # task upon completion.
    now = datetime.now().strftime("%d-%b %H:%M:%S")
    logging.info(f"{now} Trying to kick off initial {n_concurrent} partitions...")
    initial_futures_list = []
    for i in range(n_concurrent):
        try:
            partition = next(partitions)
        except StopIteration:
            logging.info(
                f"There are less partitions than processors. All {i} partitions running."
            )
            wait(initial_futures_list)
            break

        future = client.submit(
            run_pipeline,
            points_gdf=partition,
            model=model,
            satellite_name=satellite_name,
            image_resolution=image_resolution,
            image_dtype=image_dtype,
            image_bands=image_bands,
            image_width=image_width,
            min_image_edge=min_image_edge,
            seasonal=seasonal,
            year=year,
            search_start=search_start,
            search_end=search_end,
            image_composite_method=image_composite_method,
            stac_api_name=stac_api_name,
            num_features=num_features,
            device=device,
            col_names=col_names,
            save_folder_path=save_folder_path,
            save_filename=f"df_{str(i).zfill(3)}.parquet.gzip",
            return_df=False,
        )
        initial_futures_list.append(future)

    # get generator that returns futures as they are completed
    as_completed_generator = as_completed(initial_futures_list)

    # only run each remaining partitions once a previous task has completed
    for partition in partitions:
        i += 1
        completed_future = next(as_completed_generator)
        logging.info(f"Adding partition {i}")

        new_future = client.submit(
            run_pipeline,
            points_gdf=partition,
            model=model,
            satellite_name=satellite_name,
            image_resolution=image_resolution,
            image_dtype=image_dtype,
            image_bands=image_bands,
            image_width=image_width,
            min_image_edge=min_image_edge,
            seasonal=seasonal,
            year=year,
            search_start=search_start,
            search_end=search_end,
            image_composite_method=image_composite_method,
            stac_api_name=stac_api_name,
            num_features=num_features,
            device=device,
            col_names=col_names,
            save_folder_path=save_folder_path,
            save_filename=f"df_{str(i).zfill(3)}.parquet.gzip",
            return_df=False,
        )
        as_completed_generator.add(new_future)

    # wait for all futures to process
    for completed_future in as_completed_generator:
        pass

    now = datetime.now().strftime("%d-%b %H:%M:%S")
    logging.info(f"{now} Finished.")


# ALTERNATIVE - BATCHED DASK DELAYED ###################################################


def get_dask_gdf(
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
        f"Distributing {len(points_gdf)} points across {chunksize}-point partitions results in {points_dgdf.npartitions} partitions."
    )

    return points_dgdf


def run_batched_pipeline(
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
    seasonal: bool,
    year: int,
    search_start: str,
    search_end: str,
    image_composite_method: bool,
    stac_api_name: str,
    num_features: int,
    device: str,
    n_concurrent: int,
    chunksize: int,
    col_names: list[str],
    save_folder_path: str,
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
    seasonal : Whether to use seasonal satellite images for featurization.
    year : Year to be used for featurization.
    search_start : Start date for satellite image search.
    search_end : End date for satellite image search.
    image_composite_method : Mosaic composite to be used for featurization.
    stac_api_name : Name of STAC API to be used for satellite image search.
    num_features : number of mosaiks features.
    device : Device to be used for featurization.
    col_names : List of column names to be used for saving the features.
    save_folder_path : Path to folder where features will be saved.
    partition_ids : Optional. List of partition indexes to be used for featurization.

    Returns
    --------
    list of failed partition ids
    """
    # make delayed gdf partitions
    dask_gdf = get_dask_gdf(
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

    if n_partitions < n_concurrent:
        logging.info(
            f"n_partitions is smaller than n_concurrent. Running all {n_partitions} partitions."
        )
        n_concurrent = n_partitions

    failed_ids = []
    checkpoint_indices = list(np.arange(0, n_partitions, n_concurrent)) + [n_partitions]
    for p_start_id, p_end_id in zip(checkpoint_indices[:-1], checkpoint_indices[1:]):
        now = datetime.now().strftime("%d-%b %H:%M:%S")
        logging.info(f"{now} Running batch: {p_start_id} to {p_end_id - 1}")

        batch_indices = list(range(p_start_id, p_end_id))
        batch_p_ids = [partition_ids[i] for i in batch_indices]
        batch_partitions = [partitions[i] for i in batch_indices]

        failed_ids += run_batch(
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
            seasonal=seasonal,
            year=year,
            search_start=search_start,
            search_end=search_end,
            image_composite_method=image_composite_method,
            stac_api_name=stac_api_name,
            num_features=num_features,
            device=device,
            col_names=col_names,
            save_folder_path=save_folder_path,
        )

    return failed_ids


def run_batch(
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
    seasonal: bool,
    year: int,
    search_start: str,
    search_end: str,
    image_composite_method: str,
    stac_api_name: str,
    num_features: int,
    device: str,
    col_names: list,
    save_folder_path: str,
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
    seasonal : Whether to use seasonal satellite images for featurization.
    year : Year to be used for featurization.
    search_start : Start date for satellite image search.
    search_end : End date for satellite image search.
    image_composite_method : Mosaic composite to be used for featurization.
    stac_api_name : Name of STAC API to be used for satellite image search.
    num_features : number of mosaiks features.
    device : Device to be used for featurization.
    col_names : List of column names to be used for the output dataframe.
    save_folder_path : Path to folder where features will be saved.

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
            seasonal=seasonal,
            year=year,
            search_start=search_start,
            search_end=search_end,
            image_composite_method=image_composite_method,
            stac_api_name=stac_api_name,
            num_features=num_features,
            device=device,
            col_names=col_names,
            save_folder_path=save_folder_path,
            save_filename=f"df_{str_id}.parquet.gzip",
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

    # prep for next run
    # client.restart()
    # sleep(5)

    return failed_ids


# ALTERNATIVE - UNBATCHED DASK DELAYED  ################################################


def run_unbatched_delayed_pipeline(
    points_gdf: gpd.GeoDataFrame,
    client: Client,
    model: nn.Module,
    sort_points_by_hilbert_distance: bool,
    satellite_name: str,
    search_start: str,
    search_end: str,
    stac_api_name: str,
    seasonal: bool,
    year: int,
    image_composite_method: str,
    num_features: int,
    device: str,
    min_image_edge: int,
    image_width: int,
    image_bands: list[str],
    image_resolution: int,
    image_dtype: str,
    col_names: list,
    chunksize: int,
    save_folder_path: str,
) -> list[delayed]:
    """
    Given a GeoDataFrame of coordinate points, partitions it, creates a list of each
    partition's respective Dask Delayed tasks, and runs the tasks.

    Parameters
    ----------
    points_gdf : GeoDataFrame of coordinate points.
    client : Dask client.
    model : PyTorch model to be used for featurization.
    sort_points_by_hilbert_distance : Whether to sort the points by their Hilbert distance.
    satellite_name : Name of satellite to be used for featurization.
    search_start : Start date for satellite image search.
    search_end : End date for satellite image search.
    stac_api_name : Name of STAC API to be used for satellite image search.
    seasonal : Whether to use seasonal satellite images for featurization.
    year : Year to be used for featurization.
    image_composite_method : Mosaic composite to be used for featurization.
    num_features : number of mosaiks features.
    device : Device to be used for featurization.
    min_image_edge : Minimum image edge size.
    image_width : Desired width of the image to be fetched (in meters).
    image_bands : List of satellite image bands to be used for featurization.
    image_resolution : Resolution of satellite images to be used for featurization.
    image_dtype : Data type of satellite images to be used for featurization.
    col_names : List of column names to be used for the output dataframe.
    chunksize : Number of points per partition.
    save_folder_path : Path to folder where features will be saved.

    Returns
    -------
    None
    """

    dask_gdf = get_dask_gdf(
        points_gdf,
        chunksize,
        sort_points_by_hilbert_distance,
    )
    partitions = dask_gdf.to_delayed()

    delayed_tasks = []
    for i, partition in enumerate(partitions):
        str_i = str(i).zfill(3)
        # this can be swapped for dask.delayed(full_pipeline)(...)
        delayed_task = delayed_pipeline(
            points_gdf=partition,
            model=model,
            satellite_name=satellite_name,
            search_start=search_start,
            search_end=search_end,
            stac_api_name=stac_api_name,
            seasonal=seasonal,
            year=year,
            image_composite_method=image_composite_method,
            num_features=num_features,
            device=device,
            min_image_edge=min_image_edge,
            image_width=image_width,
            image_bands=image_bands,
            image_resolution=image_resolution,
            image_dtype=image_dtype,
            col_names=col_names,
            save_folder_path=save_folder_path,
            save_filename=f"df_{str_i}.parquet.gzip",
        )
        delayed_tasks.append(delayed_task)

    persist_tasks = client.persist(delayed_tasks)
    wait(persist_tasks)


def delayed_pipeline(
    points_gdf: gpd.GeoDataFrame,
    model: nn.Module,
    satellite_name: str,
    search_start: str,
    search_end: str,
    stac_api_name: str,
    seasonal: bool,
    year: int,
    image_composite_method: str,
    num_features: int,
    device: str,
    min_image_edge: int,
    image_width: int,
    image_bands: list[str],
    image_resolution: int,
    image_dtype: str,
    col_names: list,
    save_folder_path: str,
    save_filename: str,
) -> dask.delayed:
    """
    For a given GeoDataFrame of coordinate points, this function creates the necesary
    chain of delayed dask operations for image fetching and feaurisation.

    Parameters
    ----------
    points_gdf : GeoDataFrame of coordinate points.
    model : PyTorch model to be used for featurization.
    satellite_name : Name of satellite to be used for featurization.
    search_start : Start date for satellite image search.
    search_end : End date for satellite image search.
    stac_api_name : Name of STAC API to be used for satellite image search.
    seasonal : Whether to use seasonal satellite images for featurization.
    year : Year to be used for featurization.
    image_composite_method : Mosaic composite to be used for featurization.
    num_features : number of mosaiks features.
    device : Device to be used for featurization.
    min_image_edge : Minimum image edge size.
    image_width : Desired width of the image to be fetched (in meters).
    image_bands : List of satellite image bands to be used for featurization.
    image_resolution : Resolution of satellite images to be used for featurization.
    image_dtype : Data type of satellite images to be used for featurization.
    col_names : List of column names to be used for the output dataframe.
    save_folder_path : Path to folder where features will be saved.
    save_filename : Name of file to save features to.

    Returns:
        Dask.Delayed object
    """

    points_gdf_with_stac = dask.delayed(fetch_image_refs)(
        points_gdf=points_gdf,
        satellite_name=satellite_name,
        seasonal=seasonal,
        year=year,
        search_start=search_start,
        search_end=search_end,
        image_composite_method=image_composite_method,
        stac_api_name=stac_api_name,
    )

    data_loader = dask.delayed(create_data_loader)(
        points_gdf_with_stac=points_gdf_with_stac,
        image_width=image_width,
        image_bands=image_bands,
        image_resolution=image_resolution,
        image_dtype=image_dtype,
        image_composite_method=image_composite_method,
    )

    X_features = dask.delayed(create_features_from_image_array)(
        dataloader=data_loader,
        n_features=num_features,
        model=model,
        device=device,
        min_image_edge=min_image_edge,
    )

    df = dask.delayed(utl.make_result_df)(
        features=X_features,
        context_gdf=points_gdf_with_stac,
        mosaiks_col_names=col_names,
    )
    delayed_task = dask.delayed(utl.save_dataframe)(
        df=df, file_path=save_folder_path / save_filename
    )
    return delayed_task
