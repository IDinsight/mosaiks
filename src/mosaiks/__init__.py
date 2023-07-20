# from .extras import load_and_save_features

import logging
import os
from typing import List, Optional

import pandas as pd

import mosaiks.checks as checks
import mosaiks.utils as utl
from mosaiks.dask import run_pipeline_with_parallelization
from mosaiks.featurize import RCF
from mosaiks.pipeline import run_pipeline

__version__ = "0.0.1rc"

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
    seasonal: bool = False,
    year: int = None,
    search_start: str = "2013-01-01",
    search_end: str = "2013-12-31",
    image_composite_method: str = "least_cloudy",
    stac_api_name: str = "planetary-compute",
    n_mosaiks_features: int = 4000,
    mosaiks_kernel_size: int = 3,
    mosaiks_random_seed_for_filters: int = 768,
    model_device: str = "cpu",
    parallelize: bool = False,
    dask_n_concurrent_tasks: Optional[int] = None,
    dask_chunksize: int = 500,
    dask_n_workers: Optional[int] = None,
    dask_threads_per_worker: Optional[int] = None,
    dask_sort_points_by_hilbert_distance: bool = True,
    mosaiks_col_names: list = None,
    setup_rasterio_env: bool = True,
) -> pd.DataFrame:  # or None
    """
    For a given set of coordinate points, this function runs the necessary functions,
    optionally with Dask parallel processing, and returns a dataframe of MOSAIKS features.

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
    seasonal: whether to get seasonal images. Defaults to False.
    year: year to get seasonal images for in format YYYY. Only needed if seasonal = True. Defaults to None.
    search_start: start date for image search in format YYYY-MM-DD. Defaults to "2013-01-01".
    search_end: end date for image search in format YYYY-MM-DD. Defaults to "2013-12-31".
    image_composite_method: how to composite multiple images for same GPS location. Options are "least_cloudy" (pick least cloudy image) or "all" (get all images and average across them). Defaults to "least_cloudy".
    stac_api_name: which STAC API to use. Options are "planetary-compute" or "earth-search". Defaults to "planetary-compute".
    n_mosaiks_features: number of mosaiks features to generate. Defaults to 4000.
    mosaiks_kernel_size: kernel size for mosaiks filters. Defaults to 3.
    mosaiks_random_seed_for_filters: random seed for mosaiks filters. Defaults to 768.
    model_device: compute device for mosaiks model. Options are "cpu" or "cuda". Defaults to "cpu".
    parallelize: whether to use Dask parallel processing. Defaults to False.
    dask_n_concurrent_tasks: number of concurrent tasks to run in Dask. Defaults to None, which sets the total number of tasks to number of threads.
    dask_chunksize: number of datapoints per data partition in Dask. Defaults to 500.
    n_workers : Number of workers to use. If None, let Dask decide (uses all available cores).
    threads_per_worker : Number of threads per worker. If None, let Dask decide (uses all available threads per core).
    dask_sort_points_by_hilbert_distance: Whether to sort points by Hilbert distance before partitioning them. Defaults to True.
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
    checks.check_stac_api_name(stac_api_name)
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
        return run_pipeline(
            points_gdf=points_gdf,
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
            num_features=n_mosaiks_features,
            device=model_device,
            col_names=mosaiks_col_names,
            save_folder_path=None,  # TODO - pass this up to the get_features() function?
        )
    else:
        return run_pipeline_with_parallelization(
            points_gdf=points_gdf,
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
            n_mosaiks_features=n_mosaiks_features,
            model_device=model_device,
            n_concurrent_tasks=dask_n_concurrent_tasks,
            chunksize=dask_chunksize,
            n_workers=dask_n_workers,
            threads_per_worker=dask_threads_per_worker,
            sort_points_by_hilbert_distance=dask_sort_points_by_hilbert_distance,
            mosaiks_col_names=mosaiks_col_names,
            save_folder_path=None,  # TODO - pass this up to the get_features() function?
        )
