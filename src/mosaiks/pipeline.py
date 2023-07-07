import logging
import os
from typing import List

import pandas as pd

import mosaiks.utils as utl
from mosaiks.dask import (
    get_dask_client,
    get_features_without_parallelization,
    run_batched_delayed_pipeline,
)
from mosaiks.featurize import RCF


def get_features(
    latitudes: List[float],
    longitudes: List[float],
    parallelize: bool = True,
    satellite_name: str = "landsat-8-c2-l2",
    image_resolution: int = 30,
    image_dtype: str = "int16",
    image_bands: List[str] = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    buffer_distance: int = 1200,
    min_image_edge: int = 30,
    sort_points_by_hilbert_distance: bool = True,
    seasonal: bool = False,
    year: int = None,
    search_start: str = "2013-01-01",
    search_end: str = "2013-12-31",
    mosaic_composite: str = "least_cloudy",
    stac_api: str = "planetary-compute",
    n_mosaiks_features: int = 4000,
    mosaiks_kernel_size: int = 3,
    mosaiks_batch_size: int = 10,
    model_device: str = "cpu",
    dask_client_type: str = "local",
    dask_n_concurrent_tasks: int = 8,
    dask_chunksize: int = 500,
    dask_n_workers: int = 4,
    dask_threads_per_worker: int = 4,
    dask_worker_cores: int = 4,
    dask_worker_memory: int = 2,
    dask_pip_install: bool = False,
    mosaiks_col_names: list = None,
) -> pd.DataFrame:  # or None
    """
    For a given DataFrame of coordinate points, this function runs the necessary
    functions, optionally with Dask parallel processing, and optionally save results to
        file.

    Parameters:
    -----------
    latitudes: list of latitudes
    longitudes: list of longitudes
    parallelize: whether to use Dask parallel processing
    satellite_name: name of the satellite to use. Options are "landsat-8-c2-l2" or "sentinel-2-l2a". Defaults to "landsat-8-c2-l2".
    image_resolution: resolution of the satellite images in meters. Defaults to 30.
    image_dtype: data type of the satellite images. Defaults to "int16".
    image_bands: list of bands to use for the satellite images. Defaults to ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"].
    buffer_distance: buffer distance in meters. Defaults to 1000.
    min_image_edge: minimum image edge in meters. Defaults to 1000.
    sort_points_by_hilbert_distance: whether to sort points by Hilbert distance before fetching images. Defaults to True.
    seasonal: whether to get seasonal images. Defaults to False.
    year: year to get seasonal images for in format YYYY. Only needed if seasonal = True. Defaults to None.
    search_start: start date for image search in format YYYY-MM-DD. Defaults to "2013-01-01".
    search_end: end date for image search in format YYYY-MM-DD. Defaults to "2013-12-31".
    mosaic_composite: how to composite multiple images for same GPS location. Options are "least_cloudy" (pick least cloudy image) or "all" (get all images and average across them). Defaults to "least_cloudy".
    stac_api: which STAC API to use. Options are "planetary-compute" or "earth-search". Defaults to "planetary-compute".
    n_mosaiks_features: number of mosaiks features to generate. Defaults to 4000.
    mosaiks_kernel_size: kernel size for mosaiks filters. Defaults to 3.
    mosaiks_batch_size: batch size for mosaiks filters. Defaults to 10.
    model_device: compute device for mosaiks model. Options are "cpu" or "cuda". Defaults to "cpu".
    dask_client_type: type of Dask client to use. Options are "local" or "gateway". Defaults to "local".
    dask_n_concurrent_tasks: number of concurrent tasks to run in Dask. Defaults to 8.
    dask_chunksize: number of datapoints per data partition in Dask. Defaults to 500.
    dask_n_workers: number of Dask workers to use. Only needed if dask_client_type = "local". Defaults to 4.
    dask_threads_per_worker: number of threads per Dask worker to use. Defaults to 4.
    dask_worker_cores: number of cores per Dask worker to use. Defaults to 4.
    dask_worker_memory: amount of memory per Dask worker to use in GB. Defaults to 2.
    dask_pip_install: whether to install mosaiks in Dask workers. Defaults to False.
    mosaiks_col_names: column names for the mosaiks features. Defaults to None.

    Returns
    --------
    None or DataFrame

    """
    # Make points df
    points_df = pd.DataFrame({"Lat": latitudes, "Lon": longitudes})

    # Convert points to gdf
    points_gdf = utl.df_w_latlons_to_gdf(points_df)

    if mosaiks_col_names is None:
        mosaiks_col_names = [f"mosaiks_{i}" for i in range(n_mosaiks_features)]

    # Create model
    model = RCF(
        num_features=n_mosaiks_features,
        kernel_size=mosaiks_kernel_size,
        num_input_channels=len(image_bands),
    )

    # If using parallelization, run the featurization without Dask
    if parallelize:
        # Make folder for temporary checkpoints
        save_folder_path_temp = utl.make_output_folder_path(
            satellite_name=satellite_name,
            year=search_start.split("-")[0],
            n_mosaiks_features=n_mosaiks_features,
        )
        os.makedirs(save_folder_path_temp, exist_ok=True)

        # Create dask client
        _, client = get_dask_client(
            client_type=dask_client_type,
            n_workers=dask_n_workers,
            threads_per_worker=dask_threads_per_worker,
            n_concurrent=dask_n_concurrent_tasks,
            chunksize=dask_chunksize,
            worker_cores=dask_worker_cores,
            worker_memory=dask_worker_memory,
            pip_install=dask_pip_install,
        )
        logging.info("Dask client created.")

        # Run in parallel
        run_batched_delayed_pipeline(
            points_gdf=points_gdf,
            client=client,
            model=model,
            satellite_name=satellite_name,
            image_resolution=image_resolution,
            image_dtype=image_dtype,
            image_bands=image_bands,
            buffer_distance=buffer_distance,
            min_image_edge=min_image_edge,
            sort_points=sort_points_by_hilbert_distance,
            seasonal=seasonal,
            year=year,
            search_start=search_start,
            search_end=search_end,
            mosaic_composite=mosaic_composite,
            stac_api_name=stac_api,
            num_features=n_mosaiks_features,
            batch_size=mosaiks_batch_size,
            device=model_device,
            n_concurrent=dask_n_concurrent_tasks,
            chunksize=dask_chunksize,
            col_names=mosaiks_col_names,
            save_folder_path=save_folder_path_temp,
        )

        # Load checkpoint files and combine
        logging.info("Loading and combining checkpoint files...")
        checkpoint_filenames = utl.get_filtered_filenames(
            folder_path=save_folder_path_temp, prefix="df_"
        )
        combined_df = utl.load_and_combine_dataframes(
            folder_path=save_folder_path_temp, filenames=checkpoint_filenames
        )
        logging.info(
            f"Dataset size in memory (MB): {combined_df.memory_usage().sum() / 1000000}"
        )

        # Return combined df
        return combined_df
    else:
        return get_features_without_parallelization(
            points_gdf=points_gdf,
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
            mosaic_composite=mosaic_composite,
            stac_api_name=stac_api,
            num_features=n_mosaiks_features,
            batch_size=mosaiks_batch_size,
            device=model_device,
            col_names=mosaiks_col_names,
            save_folder_path=None,
        )
