import os

os.environ["USE_PYGEOS"] = "0"  # must happen before geopandas import

import logging

import mosaiks.utils as utl
from mosaiks.pipeline import get_features

from .checks import check_satellite_name, check_search_dates, check_stac_api_name
from .utils import combine_results_df_with_context_df, get_dataset_path


def load_data_and_save_created_features(
    dataset: dict,
    config_dictionary: dict = {},
    rasterio_config: dict = None,
    parallelize: bool = True,
) -> None:
    """
    Load data and save created features.

    Parameters
    ----------
    dataset: dictionary with inputs related to the dataset with the following structure:
        {"dataset_name": "dataset_name",
         "context_col_names_to_keep": list of column names to keep in final dataframe,
         "input": {"folder": folder name,
                   "file": file name (csv .or .parquet),
                   "relative_path_is_root_folder": True or False},
        "output": {"folder": folder name,
                   "file": file name (csv .or .parquet),
                   "relative_path_is_root_folder": True or False}
        }
    config_dictionary: configuration dictionary for MOSAIKS pipeline with the following structure:
        {"satellite_name": satellite_name,
         "image_resolution": resolution of the satellite images in meters,
         "image_dtype": data type of the satellite images,
         "image_bands": list of bands to use for the satellite images,
         "buffer_distance": buffer distance in meters,
         "min_image_edge": minimum image edge in meters,
         "sort_points_by_hilbert_distance": whether to sort points by Hilbert distance before fetching images,
         "seasonal": whether to get seasonal images,
         "year": year to get seasonal images for in format YYYY,
         "search_start": start date for image search in format YYYY-MM-DD,
         "search_end": end date for image search in format YYYY-MM-DD,
         "mosaic_composite": how to composite multiple images for same GPS location,
         "stac_api": which STAC API to use,
         "n_mosaiks_features": number of mosaiks features to generate,
         "mosaiks_kernel_size": kernel size for mosaiks filters,
         "mosaiks_batch_size": batch size for mosaiks filters,
         "model_device": compute device for mosaiks model,
         "dask_client_type": type of Dask client to use,
         "dask_n_concurrent_tasks": number of concurrent tasks for Dask client,
         "dask_chunksize": number of datapoints per data partition in Dask,
         "dask_n_workers": number of Dask workers to use,
         "dask_threads_per_worker": number of threads per Dask worker to use,
         "dask_worker_cores": number of cores per Dask worker to use,
         "dask_worker_memory": amount of memory per Dask worker to use in GB,
         "dask_pip_install": whether to install mosaiks in Dask workers,
         "mosaiks_col_names": list of column names to use for saving the features,
        }
        For defaults see `get_features` docstrings.
    rasterio_config: dictionary for Rasterio setup with the following structure:
        {"GDAL_DISABLE_READDIR_ON_OPEN": str,
         "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": str,
         "GDAL_SWATH_SIZE": str,
         "AWS_REQUEST_PAYER": str,
         VSI_CURL_CACHE_SIZE : str
         }
        Default config is config/rasterio_config.yaml.
    parallelize: whether to parallelize the featurization process.
    """
    # Setup Rasterio
    if rasterio_config is None:
        rasterio_config = utl.load_yaml_config("rasterioc_config.yaml")
    os.environ.update(rasterio_config)

    # Check params
    if len(config_dictionary) > 0:
        if "satellite_name" in config_dictionary.keys():
            check_satellite_name(config_dictionary["satellite_name"])
        if (
            "search_start" in config_dictionary.keys()
            and "search_end" in config_dictionary.keys()
        ):
            check_search_dates(
                config_dictionary["search_start"],
                config_dictionary["search_end"],
            )
        if "stac_api" in config_dictionary.keys():
            check_stac_api_name(config_dictionary["stac_api"])

    # Load data
    input_file_path = get_dataset_path(**dataset["input"])
    output_file_path = get_dataset_path(**dataset["output"])

    logging.info("Loading {} points...".format(dataset["dataset_name"]))
    points_df = utl.load_dataframe(input_file_path)

    # Get features
    logging.info("Getting MOSAIKS features...")
    features_df = get_features(
        latitudes=points_df["Lat"].values,
        longitudes=points_df["Lon"].values,
        parallelize=parallelize,
        **config_dictionary,
    )

    combined_df = combine_results_df_with_context_df(
        features_df=features_df,
        context_df=points_df,
        context_cols_to_keep=dataset["context_col_names_to_keep"],
    )

    # Save features
    logging.info("Saving features...")
    utl.save_dataframe(combined_df, output_file_path)
