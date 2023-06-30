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
from dask.distributed import Client, LocalCluster, as_completed, wait
from dask_gateway import Gateway

import mosaiks.utils as utl
from mosaiks.featurize import create_features_from_image_array

# for fully-delayd pipeline
from mosaiks.fetch import create_data_loader, fetch_image_refs

__all__ = [
    "get_dask_client",
    "get_local_dask_client",
    "get_gateway_cluster_client",
    "run_queued_futures_pipeline",
    "run_batched_delayed_pipeline",
    "run_unbatched_delayed_pipeline",
    "delayed_pipeline",
    "get_features_without_parallelization",
]


def get_dask_client(client_type: str = "local", **client_kwargs) -> tuple:
    """
    Get dask client.

    Parameters:
    -----------
    client_type: "local" or "gateway". "local" spins up a local Dask cluster; "gateway"
        spins up a cluster on a remote computing platform. Default is "local".
    client_kwargs: Keyword arguments to create client. See `get_local_dask_client` and
        `get_gateway_dask_client` for details.

    Returns:
    --------
    Dask cluster and client
    """
    if client_type == "local":
        return 0, get_local_dask_client(**client_kwargs)
    elif client_type == "gateway":
        return get_gateway_cluster_client(**client_kwargs)
    else:
        raise NotImplementedError


def get_local_dask_client(
    n_workers: int = 4, threads_per_worker: int = 4, **kwargs
) -> Client:
    """
    Get a local dask client.

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
        processes=True,
        threads_per_worker=threads_per_worker,
        silence_logs=logging.ERROR,
    )
    logging.info(cluster.dashboard_link)
    client = Client(cluster)
    return client


def get_gateway_cluster_client(
    worker_cores: int = 4, worker_memory: int = 2, pip_install: bool = False, **kwargs
) -> tuple:
    """
    Get gateway cluster.

    NOTE: This spins up a remote Dask cluster on a remote computing platform and allows
        us to centrally manage workers deployed across multiple machines
        (local / remote).

    Parameters
    -----------
    worker_cores : Number of cores per worker.
    worker_memory : Amount of memory per worker in GB.
    pip_install : Whether to install mosaiks on each worker.

    Returns
    --------
    Dask cluster and client.
    """
    gateway = Gateway()

    # shutdown running clusters (if any)
    for cluster_info in gateway.list_clusters():
        cluster = gateway.connect(cluster_info.name)
        cluster.shutdown()

    # spin up new cluster
    gateway = Gateway()
    options = gateway.cluster_options()
    options.worker_cores = worker_cores  # this doesn't seem to work - set with .scale()
    options.worker_memory = worker_memory  # in GB

    cluster = gateway.new_cluster(options)
    cluster.scale(worker_cores)
    client = cluster.get_client()

    # install mosaiks on the workers
    if pip_install:
        from dask.distributed import PipInstall

        mosaiks_package_link = utl.get_mosaiks_package_link("dask-improvements")
        plugin = PipInstall(
            packages=[mosaiks_package_link], pip_options=["--upgrade"], restart=False
        )
        client.register_worker_plugin(plugin)

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
    featurization_config: dict,
    satellite_config: dict,
    col_names: list,
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
    featurization_config : Dictionary of featurization parameters.
    satellite_config : Dictionary of satellite parameters.
    col_names : List of column names to be used for saving the features.
    save_folder_path : Path to folder where features will be saved.

    Returns
    --------
    None
    """

    n_concurrent = featurization_config["dask"]["n_concurrent"]

    # make generator for spliting up the data into partitions
    partitions = get_partitions_generator(
        points_gdf,
        featurization_config["dask"]["chunksize"],
        featurization_config["fetch"]["sort_points"],
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
            featurization_config=featurization_config,
            satellite_config=satellite_config,
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
            featurization_config=featurization_config,
            satellite_config=satellite_config,
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
    featurization_config: dict,
    satellite_config: dict,
    col_names: list,
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
    featurization_config : Dictionary of featurization parameters.
    satellite_config : Dictionary of satellite parameters.
    col_names : List of column names to be used for saving the features.
    save_folder_path : Path to folder where features will be saved.
    partition_ids : Optional. List of partition indexes to be used for featurization.

    Returns
    --------
    list of failed partition ids
    """

    n_concurrent = featurization_config["dask"]["n_concurrent"]

    # make delayed gdf partitions
    dask_gdf = get_dask_gdf(
        points_gdf,
        featurization_config["dask"]["chunksize"],
        featurization_config["fetch"]["sort_points"],
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
            featurization_config=featurization_config,
            satellite_config=satellite_config,
            col_names=col_names,
            save_folder_path=save_folder_path,
        )

    return failed_ids


def run_batch(
    partitions: list,
    partition_ids: list,
    client: Client,
    model: nn.Module,
    featurization_config: dict,
    satellite_config: dict,
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
    featurization_config : Dictionary of featurization parameters.
    satellite_config : Dictionary of satellite parameters.
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
            featurization_config=featurization_config,
            satellite_config=satellite_config,
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
    featurization_config: dict,
    satellite_config: dict,
    col_names: list,
    save_folder_path: str,
) -> list[dask.delayed]:
    """
    Given a GeoDataFrame of coordinate points, partitions it, creates a list of each
    partition's respective Dask Delayed tasks, and runs the tasks.

    Parameters
    ----------
    points_gdf : GeoDataFrame of coordinate points.
    client : Dask client.
    model : PyTorch model to be used for featurization.
    featurization_config : Dictionary of featurization parameters.
    satellite_config : Dictionary of satellite parameters.
    col_names : List of column names to be used for the output dataframe.
    save_folder_path : Path to folder where features will be saved.

    Returns
    -------
    None
    """

    dask_gdf = get_dask_gdf(
        points_gdf,
        featurization_config["dask"]["chunksize"],
        featurization_config["fetch"]["sort_points"],
    )
    partitions = dask_gdf.to_delayed()

    delayed_tasks = []
    for i, partition in enumerate(partitions):

        str_i = str(i).zfill(3)
        # this can be swapped for dask.delayed(full_pipeline)(...)
        delayed_task = delayed_pipeline(
            points_gdf=partition,
            model=model,
            featurization_config=featurization_config,
            satellite_config=satellite_config,
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
    featurization_config: dict,
    satellite_config: dict,
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
    featurization_config : Dictionary of featurization parameters.
    satellite_config : Dictionary of satellite parameters.
    col_names : List of column names to be used for the output dataframe.
    save_folder_path : Path to folder where features will be saved.
    save_filename : Name of file to save features to.

    Returns:
        Dask.Delayed object
    """

    satellite_search_params = featurization_config["satellite_search_params"]

    points_gdf_with_stac = dask.delayed(fetch_image_refs)(
        points_gdf, satellite_search_params
    )

    data_loader = dask.delayed(create_data_loader)(
        points_gdf_with_stac=points_gdf_with_stac,
        satellite_params=satellite_config,
        batch_size=featurization_config["model"]["batch_size"],
    )

    X_features = dask.delayed(create_features_from_image_array)(
        dataloader=data_loader,
        n_features=featurization_config["model"]["num_features"],
        model=model,
        device=featurization_config["model"]["device"],
        min_image_edge=satellite_config["min_image_edge"],
    )

    df = dask.delayed(utl.make_result_df)(
        features=X_features,
        index=points_gdf_with_stac.index,
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
    featurization_config: dict,
    satellite_config: dict,
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
    featurization_config : Dictionary of featurization parameters.
    satellite_config : Dictionary of satellite parameters.
    col_names : List of column names to be used for saving the features. Default is None, in which case the column names will be "mosaiks_0", "mosaiks_1", etc.
    save_folder_path : Path to folder where features will be saved. Default is None.
    save_filename : Name of file where features will be saved. Default is "".
    return_df : Whether to return the features as a DataFrame. Default is True.

    Returns
    --------
    None or DataFrame
    """

    try:
        satellite_search_params = featurization_config["satellite_search_params"]

        points_gdf_with_stac = fetch_image_refs(points_gdf, satellite_search_params)

        data_loader = create_data_loader(
            points_gdf_with_stac=points_gdf_with_stac,
            satellite_params=satellite_config,
            batch_size=featurization_config["model"]["batch_size"],
        )

        X_features = create_features_from_image_array(
            dataloader=data_loader,
            n_features=featurization_config["model"]["num_features"],
            model=model,
            device=featurization_config["model"]["device"],
            min_image_edge=satellite_config["min_image_edge"],
        )

        df = utl.make_result_df(
            features=X_features,
            index=points_gdf_with_stac.index,
            mosaiks_col_names=col_names,
        )

        if save_folder_path is not None:
            utl.save_dataframe(df=df, file_path=save_folder_path / save_filename)

    except Exception as e:
        logging.warn(e)

    if return_df:
        return df
