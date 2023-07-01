import os

os.environ["USE_PYGEOS"] = "0"  # must happen before geopandas import

import logging

import mosaiks.utils as utl
from mosaiks.pipeline import get_features

from .checks import check_satellite_name, check_search_dates, check_stac_api_name
from .utils import combine_results_df_with_context_df, get_dataset_path


def load_data_and_save_created_features(
    dataset: dict,
    featurisation_config: dict = None,
    satellite_config: dict = None,
    rasterio_config: dict = None,
    featurize_with_parallelization: bool = True,
    col_names: list = None,
) -> None:
    """
    Load data and save created features.

    Parameters
    ----------
    dataset: dictionary with the following structure:
        {"dataset_name": "dataset_name",
         "context_col_names_to_keep": list of column names to keep in final dataframe,
         "input": {"folder": folder name,
                   "file": file name (csv .or .parquet),
                   "relative_path_is_root_folder": True or False},
        "output": {"folder": folder name,
                   "file": file name (csv .or .parquet),
                   "relative_path_is_root_folder": True or False}
        }
    featurisation_config: dictionary with the following structure:
        {"fetch": {"sort_points": True or False},
         "satellite_search_params": {"satellite_name": landsat-8-c2-l2 or sentinel-2-l2a,
                                     "seasonal": True or False,
                                     "year": YYYY (only needed if seasonal = True),
                                     "search_start": "YYYY-MM-DD",
                                     "search_end": "YYYY-MM-DD",
                                     "mosaic_composite": least_cloudy or all,
                                     "stac_api": planetary-compute or earth-search},
         "model": {"num_features": int,
                   "kernel_size": int,
                   "batch_size": int,
                   "device": cpu or cuda},
         "dask": {"client_type": local or gateway,
                  "n_workers": int (needed for local),
                  "threads_per_worker": int (needed for local),
                  "chunksize": int (needed for local),
                  "worker_memory": int (needed for gateway),
                  "worker_cores": int (needed for gateway),
                  "pip_install": True or False (needed for gateway)}
                  }
        }
        Default config is config/featurisation_config.yaml.
    satellite_config: dictionary with the following structure:
        {satellite_name: {"resolution": int,
                          "bands": list of strings,
                          "dtype": string,
                          "buffer_distance": int,
                          "min_image_edge": int
                          }
        }
        Default config is config/satellite_config.yaml.
    rasterio_config: dictionary with the following structure:
        {"GDAL_DISABLE_READDIR_ON_OPEN": str,
         "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": str,
         "GDAL_SWATH_SIZE": str,
         "AWS_REQUEST_PAYER": str,
         VSI_CURL_CACHE_SIZE : str
         }
        Default config is config/rasterio_config.yaml.
    featurize_with_parallelization: If True, use dask parallel processing to featurize.
        Default is True.
    col_names : List of column names to be used for saving the features. Default is
        None, in which case the column names will be "mosaiks_0", "mosaiks_1", etc.
    """
    # Setup Rasterio
    if rasterio_config is None:
        rasterio_config = utl.load_yaml_config("rasterioc_config.yaml")
    os.environ.update(rasterio_config)

    # Check params
    if featurisation_config is not None:
        check_satellite_name(
            featurisation_config["satellite_search_params"]["satellite_name"]
        )
        if satellite_config is not None:
            assert (
                featurisation_config["satellite_search_params"]["satellite_name"]
                in satellite_config.keys(),
                "satellite_config must contain a dictionary for the satellite name in {}".format(
                    featurisation_config["satellite_search_params"]["satellite_name"]
                ),
            )

        check_search_dates(
            featurisation_config["satellite_search_params"]["search_start"],
            featurisation_config["satellite_search_params"]["search_end"],
        )

        check_stac_api_name(featurisation_config["satellite_search_params"]["stac_api"])

    # Load data
    input_file_path = get_dataset_path(**dataset["input"])
    output_file_path = get_dataset_path(**dataset["output"])

    logging.info("Loading {} points...".format(dataset["dataset_name"]))
    points_df = utl.load_dataframe(input_file_path)

    # Get features
    logging.info("Getting MOSAIKS features...")
    features_df = get_features(
        points_df,
        featurisation_config=featurisation_config,
        satellite_config=satellite_config,
        featurize_with_parallelization=featurize_with_parallelization,
        col_names=col_names,
    )

    combined_df = combine_results_df_with_context_df(
        features_df=features_df,
        context_df=points_df,
        context_cols_to_keep=dataset["context_col_names_to_keep"],
    )

    # Save features
    logging.info("Saving features...")
    utl.save_dataframe(combined_df, output_file_path)
