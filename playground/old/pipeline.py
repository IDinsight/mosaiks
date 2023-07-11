import os

os.environ["USE_PYGEOS"] = "0"  # must happen before geopandas import

import logging
import sys
import warnings

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sys.path += ["../"]
warnings.filterwarnings("ignore")

import mosaiks.utils as utl
from mosaiks.checks import check_satellite_name, check_search_dates, check_stac_api_name
from mosaiks.dask import get_local_dask_client, run_queued_futures_pipeline
from mosaiks.featurize import RCF

if __name__ == "__main__":

    # Setup Rasterio
    rasterio_config = utl.load_yaml_config("rasterioc_config.yaml")
    os.environ.update(rasterio_config)

    # Load params
    featurization_config = utl.load_yaml_config("featurisation.yaml")
    satellite_config = utl.load_yaml_config("satellite_config.yaml")
    satellite_config = satellite_config[
        featurization_config["satellite_search_params"]["satellite_name"]
    ]

    # Check params
    check_satellite_name(
        featurization_config["satellite_search_params"]["satellite_name"]
    )
    check_search_dates(
        featurization_config["satellite_search_params"]["search_start"],
        featurization_config["satellite_search_params"]["search_end"],
    )
    check_stac_api_name(featurization_config["satellite_search_params"]["stac_api"])

    # Setup Dask Cluster and Client
    cluster, client = get_local_dask_client(
        featurization_config["dask"]["n_workers"],
        featurization_config["dask"]["threads_per_worker"],
    )

    # Load point coords
    coord_set_name = featurization_config["coord_set"]["coord_set_name"]
    logging.info(f"Loading {coord_set_name} points...")
    points_gdf = utl.load_df_w_latlons_to_gdf(dataset_name=coord_set_name)

    # Setup model
    model = RCF(
        num_features=featurization_config["model"]["num_features"],
        kernel_size=featurization_config["model"]["kernel_size"],
        num_input_channels=len(satellite_config["bands"]),
    )

    # Set output path
    mosaiks_folder_path = utl.make_output_folder_path(
        featurization_config
    )  # Path("DATA_TEST")
    os.makedirs(mosaiks_folder_path, exist_ok=True)

    # Run in parallel
    mosaiks_col_names = [
        f"mosaiks_{i}" for i in range(featurization_config["model"]["num_features"])
    ]
    run_queued_futures_pipeline(
        points_gdf=points_gdf,  # iloc[:10]
        client=client,
        model=model,
        featurization_config=featurization_config,
        satellite_config=satellite_config,
        col_names=mosaiks_col_names,
        save_folder_path=mosaiks_folder_path,
    )

    # Load checkpoint files and combine
    logging.info("Loading and combining checkpoint files...")
    checkpoint_filenames = utl.get_filtered_filenames(
        folder_path=mosaiks_folder_path, prefix="df_"
    )
    combined_df = utl.load_and_combine_dataframes(
        folder_path=mosaiks_folder_path, filenames=checkpoint_filenames
    )
    logging.info(
        f"Dataset size in memory (MB): {combined_df.memory_usage().sum() / 1000000}"
    )

    # Save to file
    combined_filename = "new_combined_features.parquet.gzip"
    combined_filepath = mosaiks_folder_path / combined_filename
    logging.info(f"Saving combined file to {str(combined_filepath)}...")
    utl.save_dataframe(df=combined_df, file_path=combined_filepath)
