import logging
import os
from typing import List

import pandas as pd

import mosaiks.checks as checks
import mosaiks.utils as utl
from mosaiks.dask import (
    get_features_without_parallelization,
    get_local_dask_cluster_and_client,
    run_batched_delayed_pipeline,
)
from mosaiks.featurize import RCF

logging.basicConfig(level=logging.INFO)

# Rasterio variables
# See https://github.com/pangeo-data/cog-best-practices
RASTERIO_CONFIG = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
    "GDAL_SWATH_SIZE": "200000000",
    "VSI_CURL_CACHE_SIZE": "200000000",
    "AWS_REQUEST_PAYER": "requester",
}


def get_features(
    latitudes: List[float],
    longitudes: List[float],
    satellite_name: str = "landsat-8-c2-l2",
    image_resolution: int = 30,
    image_dtype: str = "int16",
    image_bands: List[str] = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    image_width: int = 3000,
    min_image_edge: int = 30,
    sort_points_by_hilbert_distance: bool = True,
    seasonal: bool = False,
    year: int = None,
    search_start: str = "2013-01-01",
    search_end: str = "2013-12-31",
    image_composite_method: str = "least_cloudy",
    stac_api: str = "planetary-compute",
    n_mosaiks_features: int = 4000,
    mosaiks_kernel_size: int = 3,
    mosaiks_batch_size: int = 10,  # TODO - What is this?
    mosaiks_random_seed_for_filters: int = 768,
    model_device: str = "cpu",
    parallelize: bool = False,
    dask_n_concurrent_tasks: int = 8,
    dask_chunksize: int = 500,
    dask_n_workers: int = 4,
    dask_threads_per_worker: int = 4,
    mosaiks_col_names: list = None,
    setup_rasterio_env: bool = True,
) -> pd.DataFrame:  # or None
    """
    For a given DataFrame of coordinate points, this function runs the necessary
    functions, optionally with Dask parallel processing, and optionally save results to
        file.

    Parameters:
    -----------
    latitudes: list of latitudes
    longitudes: list of longitudes
    satellite_name: name of the satellite to use. Options are "landsat-8-c2-l2" or "sentinel-2-l2a". Defaults to "landsat-8-c2-l2".
    image_resolution: resolution of the satellite images in meters. Defaults to 30.
    image_dtype: data type of the satellite images. Defaults to "int16". All options - "int16", "int32", and "float"
    image_bands: list of bands to use for the satellite images. Defaults to ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]. For options, read the satellite docs
    image_width: Desired width of the image to be fetched (in meters). Default 3000m.
    min_image_edge: minimum image edge in meters. Defaults to 1000.
    sort_points_by_hilbert_distance: whether to sort points by Hilbert distance before fetching images. Defaults to True.
    seasonal: whether to get seasonal images. Defaults to False.
    year: year to get seasonal images for in format YYYY. Only needed if seasonal = True. Defaults to None.
    search_start: start date for image search in format YYYY-MM-DD. Defaults to "2013-01-01".
    search_end: end date for image search in format YYYY-MM-DD. Defaults to "2013-12-31".
    image_composite_method: how to composite multiple images for same GPS location. Options are "least_cloudy" (pick least cloudy image) or "all" (get all images and average across them). Defaults to "least_cloudy".
    stac_api: which STAC API to use. Options are "planetary-compute" or "earth-search". Defaults to "planetary-compute".
    n_mosaiks_features: number of mosaiks features to generate. Defaults to 4000.
    mosaiks_kernel_size: kernel size for mosaiks filters. Defaults to 3.
    mosaiks_batch_size: batch size for mosaiks filters. Defaults to 10.
    mosaiks_random_seed_for_filters: random seed for mosaiks filters. Defaults to 768.
    model_device: compute device for mosaiks model. Options are "cpu" or "cuda". Defaults to "cpu".
    parallelize: whether to use Dask parallel processing. Defaults to False.
    dask_n_concurrent_tasks: number of concurrent tasks to run in Dask. Defaults to 8.
    dask_chunksize: number of datapoints per data partition in Dask. Defaults to 500.
    dask_n_workers: number of Dask workers to use. Defaults to 4.
    dask_threads_per_worker: number of threads per Dask worker to use. Defaults to 4.
    mosaiks_col_names: column names for the mosaiks features. Defaults to None.
    setup_rasterio_env: whether to set up rasterio environment variables. Defaults to True.

    Returns
    --------
    None or DataFrame

    """
    # Set up Rasterio
    if setup_rasterio_env:
        os.environ.update(RASTERIO_CONFIG)

    # Check inputs
    logging.info("Checking inputs...")
    checks.check_latitudes_and_longitudes(latitudes, longitudes)
    checks.check_satellite_name(satellite_name)
    checks.check_stac_api_name(stac_api)
    checks.check_search_dates(search_start, search_end)

    # Make points df
    logging.info("Formatting data and creating model...")
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
        random_seed_for_filters=mosaiks_random_seed_for_filters,
    )

    logging.info("Getting MOSAIKS features...")
    if not parallelize:
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
            image_composite_method=image_composite_method,
            stac_api_name=stac_api,
            num_features=n_mosaiks_features,
            batch_size=mosaiks_batch_size,
            device=model_device,
            col_names=mosaiks_col_names,
            save_folder_path=None,  # TODO - pass this up to the get_features() function?
        )
    else:  # Run the featurization with Dask
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
            n_workers=dask_n_workers, threads_per_worker=dask_threads_per_worker
        )

        logging.info(
            f"Dask client created. Dashboard link: {client.dashboard_link}\n"
            "Running featurization in parallel with:\n"
            f"{dask_n_concurrent_tasks} concurrent tasks running on\n"
            f"{dask_n_workers} workers\n"
            f"{dask_threads_per_worker} threads per worker\n"
        )

        # Run in batches in parallel
        run_batched_delayed_pipeline(
            points_gdf=points_gdf,
            client=client,
            model=model,
            satellite_name=satellite_name,
            image_resolution=image_resolution,
            image_dtype=image_dtype,
            image_bands=image_bands,
            image_width=image_width,
            min_image_edge=min_image_edge,
            sort_points=sort_points_by_hilbert_distance,
            seasonal=seasonal,
            year=year,
            search_start=search_start,
            search_end=search_end,
            image_composite_method=image_composite_method,
            stac_api_name=stac_api,
            num_features=n_mosaiks_features,
            batch_size=mosaiks_batch_size,
            device=model_device,
            n_concurrent=dask_n_concurrent_tasks,
            chunksize=dask_chunksize,
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
        return combined_df
