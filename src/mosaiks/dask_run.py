import logging
import math
from datetime import datetime
from typing import Generator

import dask.delayed
import dask_geopandas as dask_gpd
import geopandas as gpd
import pandas as pd
import torch.nn
from dask.distributed import Client, LocalCluster, as_completed

import mosaiks.utils as utl
from mosaiks.featurize import create_data_loader, create_features, fetch_image_refs

__all__ = [
    "get_local_dask_client",
    "get_sorted_partitions_generator",
    "run_pipeline",
    "run_partitions_as_queue",
    "get_sorted_dask_gdf",
    "make_all_delayed_pipelines",
    "make_delayed_pipeline",
]


def get_local_dask_client(n_workers: int = 4, threads_per_worker: int = 4) -> Client:
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


def get_sorted_partitions_generator(
    points_gdf: gpd.GeoDataFrame, chunksize: int
) -> Generator[gpd.GeoDataFrame, None, None]:
    """
    Given a GeoDataFrame, this function sorts the points by hilbert distance and
    creates a generator that returns chunksize number of rows per iteration.

    To be used for submitting Dask Futures.

    Parameters
    -----------
    points_gdf : GeoDataFrame of points to be featurized.
    chunksize : Number of points to be featurized per iteration.

    Returns
    --------
    Generator
    """

    points_gdf = _sort_by_hilbert_distance(points_gdf)
    num_chunks = math.ceil(len(points_gdf) / chunksize)

    logging.info(
        f"Distributing {len(points_gdf)} points across {chunksize}-point partitions results in {num_chunks} partitions."
    )

    chunked_df_list = []
    for i in range(num_chunks):
        yield points_gdf.iloc[i * chunksize : (i + 1) * chunksize]


def _sort_by_hilbert_distance(points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sort the points in the GeoDataFrame by their Hilbert distance."""

    ddf = dask_gpd.from_geopandas(points_gdf, npartitions=1)
    hilbert_distance = ddf.hilbert_distance().compute()
    points_gdf["hilbert_distance"] = hilbert_distance
    points_gdf = points_gdf.sort_values("hilbert_distance")

    return points_gdf


def run_pipeline(
    points_gdf: gpd.GeoDataFrame,
    model: torch.nn.Module,
    featurization_config: dict,
    satellite_config: dict,
    column_names: list,
    save_folder_path: str,
    save_filename: str,
    return_df: bool = False,
) -> None:
    """
    For a given GeoDataFrame of coordinate points, this function runs the necessary functions
    and saves resulting mosaiks features to file.

    Parameters
    -----------
    points_gdf : GeoDataFrame of points to be featurized.
    model : PyTorch model to be used for featurization.
    featurization_config : Dictionary of featurization parameters.
    satellite_config : Dictionary of satellite parameters.
    column_names : List of column names to be used for saving the features.
    save_folder_path : Path to folder where features will be saved.
    save_filename : Name of file where features will be saved.

    Returns
    --------
    None
    """

    satellite_search_params = featurization_config["satellite_search_params"]

    points_gdf_with_stac = fetch_image_refs(points_gdf, satellite_search_params)

    data_loader = create_data_loader(
        points_gdf_with_stac=points_gdf_with_stac,
        satellite_params=satellite_config,
        batch_size=featurization_config["model"]["batch_size"],
    )

    X_features = create_features(
        dataloader=data_loader,
        n_features=featurization_config["model"]["num_features"],
        model=model,
        device=featurization_config["model"]["device"],
        min_image_edge=satellite_config["min_image_edge"],
    )

    df = pd.DataFrame(data=X_features, index=points_gdf.index, columns=column_names)

    if save_folder_path is not None:
        utl.save_dataframe(df=df, file_path=save_folder_path / save_filename)

    if return_df:
        return df


def run_partitions_as_queue(
    points_gdf: gpd.GeoDataFrame,
    client: Client,
    model: torch.nn.Module,
    featurization_config: dict,
    satellite_config: dict,
    column_names: list,
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
    column_names : List of column names to be used for saving the features.
    save_folder_path : Path to folder where features will be saved.

    Returns
    --------
    None
    """

    # make generator for spliting up the data into partitions
    partitions = get_sorted_partitions_generator(
        points_gdf, featurization_config["dask"]["chunksize"]
    )

    # get total number of threads available
    n_threads = (
        featurization_config["dask"]["n_workers"]
        * featurization_config["dask"]["threads_per_worker"]
    )

    # kickoff "n_threads" number of tasks. Each of these will be replaced by a new task
    # upon completion.
    now = datetime.now().strftime("%d-%b %H:%M:%S")
    logging.info(f"{now} Trying to kick off initial {n_threads} partitions...")
    initial_futures_list = []
    for i in range(n_threads):

        try:
            partition = next(partitions)
        except StopIteration:
            logging.info(
                f"There are less partitions than processors. All {i} partitions running."
            )
            break

        future = client.submit(
            run_pipeline,
            points_gdf=partition,
            model=model,
            featurization_config=featurization_config,
            satellite_config=satellite_config,
            column_names=column_names,
            save_folder_path=save_folder_path,
            save_filename=f"df_{str(i).zfill(3)}.parquet.gzip",
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
            featurization_config=featurization_config,
            satellite_config=satellite_config,
            column_names=column_names,
            save_folder_path=save_folder_path,
            save_filename=f"df_{str(i).zfill(3)}.parquet.gzip",
        )
        as_completed_generator.add(new_future)

    now = datetime.now().strftime("%d-%b %H:%M:%S")
    logging.info(f"{now} Added last partition.")


# OLD - USING DASK DELAYED ################
def get_sorted_dask_gdf(
    points_gdf: gpd.GeoDataFrame, chunksize: int
) -> dask_gpd.GeoDataFrame:
    """
    Spatially sort and split the gdf up by the given chunksize. To be used to create
    Dask Delayed jobs.

    Parameters
    ----------
    points_dgdf : A GeoDataFrame with a column named "geometry" containing shapely Point objects.
    chunksize : The number of points per partition to use creating the Dask GeoDataFrame.

    Returns
    -------
    points_dgdf: Dask GeoDataFrame split into partitions of size `chunksize`.
    """

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


def make_all_delayed_pipelines(
    points_gdf: gpd.GeoDataFrame,
    model: torch.nn.Module,
    featurization_config: dict,
    satellite_config: dict,
    save_folder_path: str,
) -> list[dask.delayed]:
    """
    Given a partitioned Dask GeoDataFrame of coordinate points, returns a list of each partition's
    respective Dask Delayed pipeline.

    Parameters
    ----------
    points_gdf : A GeoDataFrame with a column named "geometry" containing shapely Point objects.
    model : PyTorch model to be used for featurization.
    featurization_config : Dictionary of featurization parameters.
    satellite_config : Dictionary of satellite parameters.
    save_folder_path : Path to folder where features will be saved.

    Returns
    -------
    delayed_pipelines : List of Dask Delayed pipelines.
    """

    dask_gdf = get_sorted_dask_gdf(
        points_gdf, featurization_config["dask"]["chunksize"]
    )
    partitions = dask_gdf.to_delayed()

    mosaiks_column_names = [
        f"mosaiks_{i}" for i in range(featurization_config["model"]["num_features"])
    ]

    delayed_pipelines = []

    for i, partition in enumerate(partitions):

        str_i = str(i).zfill(3)
        delayed_pipeline = make_delayed_pipeline(
            points_gdf=partition,
            model=model,
            featurization_config=featurization_config,
            satellite_config=satellite_config,
            column_names=mosaiks_column_names,
            save_folder_path=save_folder_path,
            save_filename=f"df_{str_i}.parquet.gzip",
        )
        delayed_pipelines.append(delayed_pipeline)

    return delayed_pipelines


def make_delayed_pipeline(
    points_gdf: gpd.GeoDataFrame,
    model: torch.nn.Module,
    featurization_config: dict,
    satellite_config: dict,
    column_names: list[str],
    save_folder_path: str,
    save_filename: str,
) -> dask.delayed:
    """
    For a given GeoDataFrame of coordinate points, this function creates the necesary chain
    of delayed dask operations for image fetching and feaurisation.

    Parameters
    ----------
    points_gdf : A GeoDataFrame with a column named "geometry" containing shapely Point objects.
    model : PyTorch model to be used for featurization.
    featurization_config : Dictionary of featurization parameters.
    satellite_config : Dictionary of satellite parameters.
    column_names : List of column names to be used for the output GeoDataFrame.
    save_folder_path : Path to folder where features will be saved.
    save_filename : Filename of the output GeoDataFrame.

    Returns
    --------
    delayed_pipeline : Dask Delayed pipeline.
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

    X_features = dask.delayed(create_features)(
        dataloader=data_loader,
        n_features=featurization_config["model"]["num_features"],
        model=model,
        device=featurization_config["model"]["device"],
        min_image_edge=satellite_config["min_image_edge"],
    )

    df = dask.delayed(pd.DataFrame)(
        data=X_features, index=points_gdf.index, columns=column_names
    )

    delayed_pipeline = dask.delayed(utl.save_dataframe)(
        df=df, file_path=save_folder_path / save_filename
    )

    return delayed_pipeline
