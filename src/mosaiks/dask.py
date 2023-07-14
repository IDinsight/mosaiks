import logging
import math
from datetime import datetime
from typing import Generator

import dask.delayed
import dask_geopandas as dask_gpd
import geopandas as gpd
import numpy as np
import pandas as pd
import torch.nn
import torch.nn as nn
from dask import delayed
from dask.distributed import Client, LocalCluster, as_completed, wait

import mosaiks.utils as utl
from mosaiks.featurize import create_features_from_image_array

# for fully-delayd pipeline
from mosaiks.fetch import create_data_loader, fetch_image_refs

__all__ = [
    "get_local_dask_cluster_and_client",
    "run_queued_futures_pipeline",
    "run_batched_delayed_pipeline",
    "run_unbatched_delayed_pipeline",
    "delayed_pipeline",
    "get_features_without_parallelization",
]


def get_local_dask_cluster_and_client(
    n_workers: int = 4, threads_per_worker: int = 4
) -> tuple:
    """
    Get a local dask cluster and client.

    Parameters
    -----------
    n_workers : Number of workers to use.
    threads_per_worker : Number of threads per worker.

    Returns
    --------
    Dask client.
    """

    cluster = LocalCluster(
        n_workers=n_workers,
        processes=True,  # TODO - is this necessary?
        threads_per_worker=threads_per_worker,
        silence_logs=logging.ERROR,
    )
    client = Client(cluster)
    return cluster, client


def get_partitions_generator(
    points_gdf: gpd.GeoDataFrame,
    chunksize: int,
    sort_by_hilbert: bool = False,
) -> Generator[gpd.GeoDataFrame, None, None]:
    """
    Given a GeoDataFrame, this function creates a generator that returns chunksize
    number of rows per iteration.

    To be used for submitting Dask Futures.

    Parameters
    -----------
    points_gdf : GeoDataFrame of points to be featurized.
    chunksize : Number of points to be featurized per iteration.
    sort_by_hilbert : Whether to sort the points by their Hilbert distance.

    Returns
    --------
    Generator
    """

    if sort_by_hilbert:
        points_gdf = _sort_by_hilbert_distance(points_gdf)

    num_chunks = math.ceil(len(points_gdf) / chunksize)

    logging.info(
        f"Distributing {len(points_gdf)} points across {chunksize}-point partitions "
        f"results in {num_chunks} partitions."
    )

    for i in range(num_chunks):
        yield points_gdf.iloc[i * chunksize : (i + 1) * chunksize]


def _sort_by_hilbert_distance(points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
    buffer_distance: int,
    min_image_edge: int,
    sort_points: bool,
    seasonal: bool,
    year: int,
    search_start: str,
    search_end: str,
    image_composite_method: str,
    stac_api_name: str,
    num_features: int,
    batch_size: int,
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
    buffer_distance : Buffer distance to be used for generating the image.
    min_image_edge : Minimum edge length of the image to be generated.
    sort_points : Whether to sort the points by their Hilbert distance.
    seasonal : Whether to use seasonal imagery.
    year : Year of imagery to be used.
    search_start : Start date of imagery to be used.
    search_end : End date of imagery to be used.
    image_composite_method : Mosaic composite to be used.
    stac_api_name : Name of the STAC API to be used.
    num_features : Number of features to be extracted from the model.
    batch_size : Batch size to be used for featurization.
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
        sort_points,
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
            get_features_without_parallelization,
            points_gdf=partition,
            model=model,
            satellite_name=satellite_name,
            image_resolution=image_resolution,
            image_dtype=image_dtype,
            image_bands=image_bands,
            buffer_distance=buffer_distance,
            min_image_edge=min_image_edge,
            seasonal=seasonal,
            year=year,
            search_start=search_start,
            search_end=search_end,
            image_composite_method=image_composite_method,
            stac_api_name=stac_api_name,
            num_features=num_features,
            batch_size=batch_size,
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
            get_features_without_parallelization,
            points_gdf=partition,
            model=model,
            satellite_name=satellite_name,
            image_resolution=image_resolution,
            image_dtype=image_dtype,
            image_bands=image_bands,
            buffer_distance=buffer_distance,
            min_image_edge=min_image_edge,
            seasonal=seasonal,
            year=year,
            search_start=search_start,
            search_end=search_end,
            image_composite_method=image_composite_method,
            stac_api_name=stac_api_name,
            num_features=num_features,
            batch_size=batch_size,
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
    points_gdf: gpd.GeoDataFrame, chunksize: int, sort_by_hilbert: bool = True
) -> dask_gpd.GeoDataFrame:
    """
    Split the gdf up by the given chunksize. To be used to create Dask Delayed jobs.

    Parameters
    ----------
    points_dgdf : A GeoDataFrame with a column named "geometry" containing shapely
        Point objects.
    chunksize : The number of points per partition to use creating the Dask
        GeoDataFrame.
    sort_by_hilbert : Whether to sort the points by their Hilbert distance before

    Returns
    -------
    points_dgdf: Dask GeoDataFrame split into partitions of size `chunksize`.
    """

    if sort_by_hilbert:
        points_gdf = _sort_by_hilbert_distance(points_gdf)

    points_dgdf = dask_gpd.from_geopandas(
        points_gdf,
        chunksize=chunksize,
        sort=False,
    )

    logging.info(
        f"Distributing {len(points_gdf)} points across {chunksize}-point partitions results in {points_dgdf.npartitions} partitions."
    )

    return points_dgdf


def run_batched_delayed_pipeline(
    points_gdf: gpd.GeoDataFrame,
    client: Client,
    model: nn.Module,
    satellite_name: str,
    image_resolution: int,
    image_dtype: str,
    image_bands: list[str],
    buffer_distance: int,
    min_image_edge: int,
    sort_points: bool,
    seasonal: bool,
    year: int,
    search_start: str,
    search_end: str,
    image_composite_method: bool,
    stac_api_name: str,
    num_features: int,
    batch_size: int,
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
    buffer_distance : Buffer distance for fetching satellite images.
    min_image_edge : Minimum image edge size.
    seasonal : Whether to use seasonal satellite images for featurization.
    year : Year to be used for featurization.
    search_start : Start date for satellite image search.
    search_end : End date for satellite image search.
    image_composite_method : Mosaic composite to be used for featurization.
    stac_api_name : Name of STAC API to be used for satellite image search.
    num_features : number of mosaiks features.
    batch_size : Batch size for featurization.
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
        sort_points,
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
            buffer_distance=buffer_distance,
            min_image_edge=min_image_edge,
            seasonal=seasonal,
            year=year,
            search_start=search_start,
            search_end=search_end,
            image_composite_method=image_composite_method,
            stac_api_name=stac_api_name,
            num_features=num_features,
            batch_size=batch_size,
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
    buffer_distance: int,
    min_image_edge: int,
    seasonal: bool,
    year: int,
    search_start: str,
    search_end: str,
    image_composite_method: str,
    stac_api_name: str,
    num_features: int,
    batch_size: int,
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
    buffer_distance : Buffer distance for fetching satellite images.
    min_image_edge : Minimum image edge size.
    seasonal : Whether to use seasonal satellite images for featurization.
    year : Year to be used for featurization.
    search_start : Start date for satellite image search.
    search_end : End date for satellite image search.
    image_composite_method : Mosaic composite to be used for featurization.
    stac_api_name : Name of STAC API to be used for satellite image search.
    num_features : number of mosaiks features.
    batch_size : Batch size for featurization.
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
        delayed_task = dask.delayed(get_features_without_parallelization)(
            points_gdf=partition,
            model=model,
            satellite_name=satellite_name,
            image_resolution=image_resolution,
            image_dtype=image_dtype,
            image_bands=image_bands,
            buffer_distance=buffer_distance,
            min_image_edge=min_image_edge,
            seasonal=seasonal,
            year=year,
            search_start=search_start,
            search_end=search_end,
            image_composite_method=image_composite_method,
            stac_api_name=stac_api_name,
            num_features=num_features,
            batch_size=batch_size,
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
    sort_points: bool,
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
    batch_size: int,
    buffer_distance: int,
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
    sort_points : Whether to sort the points by their Hilbert distance.
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
    batch_size : Batch size for featurization.
    buffer_distance : Buffer distance for fetching satellite images.
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
        sort_points,
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
            batch_size=batch_size,
            buffer_distance=buffer_distance,
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
    batch_size: int,
    buffer_distance: int,
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
    batch_size : Batch size for featurization.
    buffer_distance : Buffer distance for fetching satellite images.
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
        buffer_distance=buffer_distance,
        image_bands=image_bands,
        image_resolution=image_resolution,
        image_dtype=image_dtype,
        batch_size=batch_size,
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


# TODO: Not the best name for this function, and also maybe not the best location?
def get_features_without_parallelization(
    points_gdf: gpd.GeoDataFrame,
    model: torch.nn.Module,
    satellite_name: str,
    image_resolution: int,
    image_dtype: str,
    image_bands: list,
    buffer_distance: int,
    min_image_edge: int,
    seasonal: bool,
    year: int,
    search_start: str,
    search_end: str,
    image_composite_method: str,
    stac_api_name: str,
    num_features: int,
    batch_size: int,
    device: str,
    col_names: list,
    save_folder_path: str = None,
    save_filename: str = "",
    return_df: bool = True,
) -> pd.DataFrame:  # or None
    """
    For a given DataFrame of coordinate points, this function runs the necessary
    functions and optionally saves resulting mosaiks features to file. No Dask is necessary.

    Parameters
    -----------
    points_gdf : GeoDataFrame of points to be featurized.
    model: PyTorch model to be used for featurization.
    satellite_name : Name of satellite to be used for featurization.
    image_resolution : Resolution of satellite images to be used for featurization.
    image_dtype : Data type of satellite images to be used for featurization.
    image_bands : List of satellite image bands to be used for featurization.
    buffer_distance : Buffer distance for fetching satellite images.
    min_image_edge : Minimum image edge size.
    seasonal : Whether to use seasonal satellite images for featurization.
    year : Year to be used for featurization.
    search_start : Start date for satellite image search.
    search_end : End date for satellite image search.
    image_composite_method : Mosaic composite to be used for featurization.
    stac_api_name : Name of STAC API to be used for satellite image search.
    num_features : number of mosaiks features.
    device : Device to be used for featurization.
    batch_size : Batch size for featurization.
    col_names : List of column names to be used for saving the features. Default is None, in which case the column names will be "mosaiks_0", "mosaiks_1", etc.
    save_folder_path : Path to folder where features will be saved. Default is None.
    save_filename : Name of file where features will be saved. Default is "".
    return_df : Whether to return the features as a DataFrame. Default is True.

    Returns
    --------
    None or DataFrame
    """

    try:
        points_gdf_with_stac = fetch_image_refs(
            points_gdf=points_gdf,
            satellite_name=satellite_name,
            seasonal=seasonal,
            year=year,
            search_start=search_start,
            search_end=search_end,
            image_composite_method=image_composite_method,
            stac_api_name=stac_api_name,
        )

        data_loader = create_data_loader(
            points_gdf_with_stac=points_gdf_with_stac,
            image_bands=image_bands,
            image_resolution=image_resolution,
            image_dtype=image_dtype,
            buffer_distance=buffer_distance,
            batch_size=batch_size,
            image_composite_method=image_composite_method,
        )

        X_features = create_features_from_image_array(
            dataloader=data_loader,
            n_features=num_features,
            model=model,
            device=device,
            min_image_edge=min_image_edge,
        )

        df = utl.make_result_df(
            features=X_features,
            context_gdf=points_gdf_with_stac,
            mosaiks_col_names=col_names,
        )

        if save_folder_path is not None:
            utl.save_dataframe(df=df, file_path=save_folder_path / save_filename)

        if return_df:
            return df

    except Exception as e:
        logging.warn(e)
