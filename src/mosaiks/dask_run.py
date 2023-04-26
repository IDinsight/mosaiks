import logging
from datetime import datetime
from time import sleep

import numpy as np
import pandas as pd
import torch.nn as nn
from dask import delayed
from dask.distributed import Client, LocalCluster, as_completed

import mosaiks.utils as utl
from mosaiks.featurize import create_data_loader, create_data_loader_GEE, create_features

__all__ = [
    "get_local_dask_client",
    "run_single_partition",
    "run_partitions",
]


def run_partitions(
    partitions: list,
    satellite_config: dict,
    featurization_config: dict,
    model: nn.Module,
    client: Client,
    mosaiks_folder_path: str = None,
    partition_ids: list = None,
) -> list:
    """Run partitions in batches of n_per_run and save the result for each partition
    to a parquet file. If a partition fails to be featurized, the partition ID is added
    to a list and returned at the end of the run.

    Parameters
    ----------
    partitions : List of dataframes.
    satellite_config : Dictionary containing the satellite configuration.
    featurization_config : Dictionary containing the featurization parameters.
    model : PyTorch random convolutional feature model.
    client : Dask client.
    mosaiks_folder_path : Path to the folder where the mosaiks features should be saved.
    partition_ids : List of partition IDs corresponding to each partition in `partitions`.
        If None, the partition IDs will be inferred from the order of the
        partitions in the list. Default is None.

    Returns
    -------
    failed_ids : List of partition IDs that failed to be featurized.
    """

    n_per_run = featurization_config["dask"]["n_per_run"]
    n_partitions = len(partitions)
    logging.info(f"Running {n_partitions} partitions...")

    if n_partitions < n_per_run:
        logging.info(
            f"n_partitions is smaller than n_per_run. Running all {n_partitions} partitions."
        )
        n_per_run = n_partitions

    if partition_ids is None:
        partition_ids = list(range(n_partitions))

    mosaiks_column_names = [
        f"mosaiks_{i}" for i in range(featurization_config["model"]["num_features"])
    ]

    failed_ids = []
    checkpoint_indices = np.arange(0, n_partitions + n_per_run, n_per_run)
    for p_start_id, p_end_id in zip(checkpoint_indices[:-1], checkpoint_indices[1:]):

        now = datetime.now().strftime("%d-%b %H:%M:%S")
        logging.info(f"{now} Running batch: {p_start_id} to {p_end_id - 1}")

        batch_indices = list(range(p_start_id, p_end_id))
        batch_p_ids = [partition_ids[i] for i in batch_indices]
        batch_partitions = [partitions[i] for i in batch_indices]

        failed_ids += run_batch(
            partitions=batch_partitions,
            partition_ids=batch_p_ids,
            satellite_config=satellite_config,
            featurization_config=featurization_config,
            mosaiks_column_names=mosaiks_column_names,
            model=model,
            client=client,
            mosaiks_folder_path=mosaiks_folder_path,
        )

    return failed_ids


def run_batch(
    partitions: list,
    partition_ids: list,
    satellite_config: dict,
    featurization_config: dict,
    mosaiks_column_names: list,
    model: nn.Module,
    client: Client,
    mosaiks_folder_path: str,
) -> list:
    """
    Run a batch of partitions and save the result for each partition to a parquet file.

    Parameters
    ----------
    partitions :List of dataframes to process.
    partition_ids : List containing IDs corresponding to the partitions passed (to be used
        for naming saved files and reference in case of failure).
    satellite_config : Dictionary containing the satellite configuration.
    featurization_config : Dictionary containing the featurization parameters.
    model : PyTorch random convolutional feature model.
    client : Dask client.
    mosaiks_folder_path : Path to the folder where the mosaiks features should be saved.

    Returns
    -------
    failed_ids : List of partition labels that failed to be featurized.
    """

    failed_ids = []
    delayed_dfs = []

    # collect futures
    for p_id, p in zip(partition_ids, partitions):
        str_id = str(p_id).zfill(3)  # makes 1 into '001'

        f = delayed_partition_run(
            df=p,
            satellite_config=satellite_config,
            featurization_config=featurization_config,
            mosaiks_column_names=mosaiks_column_names,
            model=model,
            dask_key_name=f"features_{str_id}",
        )
        delayed_dfs.append(f)

    # delayed -> futures -> collected results
    futures_dfs = client.compute(delayed_dfs)
    failed_ids = collect_results(
        futures_dfs=futures_dfs, mosaiks_folder_path=mosaiks_folder_path
    )

    # prep for next run
    client.restart()
    sleep(5)

    return failed_ids


def collect_results(futures_dfs: list, mosaiks_folder_path: str) -> list:
    """
    Save computed dataframes to parquet files. If a partition fails to be featurized,
    the partition ID is added to a list.

    Parameters
    ----------
    futures_dfs : List of futures containing the computed dataframes.
    mosaiks_folder_path : Path to the folder where the mosaiks features should be saved.

    Returns
    -------
    failed_ids : List of partition IDs that failed to be featurized.
    """

    failed_ids = []
    for f in as_completed(futures_dfs):
        try:
            df = f.result()
            utl.save_dataframe(
                df=df, file_path=f"{mosaiks_folder_path}/df_{f.key}.parquet.gzip"
            )
        except Exception as e:
            f_key = f.key
            partition_id = int(f_key.split("features_")[1])
            logging.info(f"Partition {partition_id} failed. Error:", e)
            failed_ids.append(partition_id)

    return failed_ids


def run_single_partition(
    partition: pd.DataFrame,
    satellite_config: dict,
    featurization_config: dict,
    model: nn.Module,
    client: Client,
) -> pd.DataFrame:
    """Run featurization for a single partition. For testing.

    Parameters
    ----------
    partition : Dataframe containing the data to featurize.
    satellite_config : Dictionary containing the satellite configuration.
    featurization_config : Dictionary containing the featurization parameters.
    model : PyTorch random convolutional feature model.
    client : Dask client.

    Returns
    -------
    df : Dataframe containing the featurized data.
    """

    mosaiks_column_names = [
        f"mosaiks_{i}" for i in range(featurization_config["model"]["num_features"])
    ]

    f = delayed_partition_run(
        df=partition,
        satellite_config=satellite_config,
        featurization_config=featurization_config,
        mosaiks_column_names=mosaiks_column_names,
        model=model,
        dask_key_name="single_run",
    )

    df_future = client.compute(f)
    for f in as_completed([df_future]):
        df = f.result()

    return df


@delayed
def delayed_partition_run(
    df: pd.DataFrame,
    satellite_config: dict,
    featurization_config: dict,
    mosaiks_column_names: list,
    model: nn.Module,
) -> pd.DataFrame:
    """Run featurization for a single partition."""

    data_loader = create_data_loader(
        points_gdf_with_stac=df,
        satellite_params=satellite_config,
        batch_size=featurization_config["model"]["batch_size"],
    )

    X_features = create_features(
        dataloader=data_loader,
        n_features=featurization_config["model"]["num_features"],
        n_points=len(df),
        model=model,
        device=featurization_config["model"]["device"],
        min_image_edge=satellite_config["min_image_edge"],
    )

    df = pd.DataFrame(
        data=X_features, index=df.index.copy(), columns=mosaiks_column_names
    )

    return df


def get_local_dask_client(n_workers: int = 4, threads_per_worker: int = 4) -> Client:
    """
    Get a local dask client.

    Parameters:
    -----------
    n_workers : Number of workers to use.
    threads_per_worker : Number of threads per worker.
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
