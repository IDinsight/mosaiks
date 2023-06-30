import logging
import os

import pandas as pd

import mosaiks.utils as utl
from mosaiks.dask import (
    get_dask_client,
    get_features_without_parallelization,
    run_batched_delayed_pipeline,
)
from mosaiks.featurize import RCF


def get_features(
    points_df: pd.DataFrame,
    featurization_config: dict,
    satellite_config: dict,
    featurize_with_parallelization: bool = True,
    dask_cluster_config: dict = None,
    col_names: list = None,
) -> pd.DataFrame:  # or None
    """
    For a given DataFrame of coordinate points, this function runs the necessary functions, optionally with Dask parallel processing, and optionally save results to file.

    Parameters:
    -----------
    points_df : DataFrame of points to be featurized.
    featurization_config : Dictionary of featurization parameters.
    satellite_config : Dictionary of satellite parameters.
    featurize_with_parallelization: If True, use dask parallel processing to featurize. Default is True.
    dask_cluster_config : Dictionary of dask cluster parameters. Default is None.
    col_names : List of column names to be used for saving the features. Default is None, in which case the column names will be "mosaiks_0", "mosaiks_1", etc.
    save_folder_path : Path to folder where features will be saved. Default is None.
    save_filename : Name of file where features will be saved. Default is "".
    return_df : Whether to return the features as a DataFrame. Default is True.

    Returns
    --------
    None or DataFrame

    """
    # Convert points to gdf
    points_gdf = utl.df_w_latlons_to_gdf(points_df)

    if col_names is None:
        col_names = [
            f"mosaiks_{i}" for i in range(featurization_config["model"]["num_features"])
        ]

    # Create model
    model = RCF(
        num_features=featurization_config["model"]["num_features"],
        kernel_size=featurization_config["model"]["kernel_size"],
        num_input_channels=len(satellite_config["bands"]),
    )

    # If using parallelization, run the featurization without Dask
    if featurize_with_parallelization:
        # Make folder for temporary checkpoints

        save_folder_path_temp = utl.make_output_folder_path(featurization_config)
        os.makedirs(save_folder_path_temp, exist_ok=True)

        # Create dask client
        _, client = get_dask_client(**dask_cluster_config)
        logging.info("Dask client created.")

        # Run in parallel
        run_batched_delayed_pipeline(
            points_gdf=points_gdf,
            client=client,
            model=model,
            featurization_config=featurization_config,
            satellite_config=satellite_config,
            col_names=col_names,
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
            featurization_config=featurization_config,
            satellite_config=satellite_config,
            col_names=col_names,
        )
