import logging
import os
import sys
import warnings

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sys.path += ["../"]
warnings.filterwarnings("ignore")

import mosaiks.utils as utl
from mosaiks.dask_run import *
from mosaiks.featurize import *

if __name__ == "__main__":

    # Setup Rasterio
    rasterio_config = utl.load_yaml_config("rasterioc_config.yaml")
    os.environ.update(rasterio_config)

    # Setup Dask Cluster and Client
    client = get_dask_client(kind="local")

    # Load params
    featurization_config = utl.load_yaml_config("featurisation.yaml")
    satellite_config = utl.load_yaml_config("satellite_config.yaml")
    satellite_config = satellite_config[
        featurization_config["satellite_search_params"]["satellite_name"]
    ]

    # Load point coords
    logging.info(f"Loading {featurization_config.coord_set_name} points...")
    points_gdf = utl.load_df_w_latlons_to_gdf(
        dataset_name=featurization_config["coord_set_name"]
    )
    points_dgdf = get_dask_gdf(points_gdf, featurization_config["dask"]["chunksize"])

    # Fetch image stac refs
    points_gdf_with_stac = fetch_image_refs(
        points_dgdf=points_dgdf,
        satellite_search_params=featurization_config["satellite_search_params"],
    )
    partitions = points_gdf_with_stac.to_delayed()

    # setup model
    model = RCF(
        num_features=featurization_config["num_features"],
        kernel_size=featurization_config["kernel_size"],
        num_input_channels=len(satellite_config["bands"]),
    )

    # Run in parallel
    mosaiks_folder_path = utl.make_features_path_from_dict(
        featurization_config, featurization_config["coord_set_name"]
    )
    failed_partition_ids = run_partitions(
        partitions=partitions,
        satellite_config=satellite_config,
        featurization_config=featurization_config,
        model=model,
        client=client,
        mosaiks_folder_path=mosaiks_folder_path,
    )

    # Re-run failed partitions
    if len(failed_partition_ids) > 0:

        logging.info(f"Re-running {len(failed_partition_ids)} failed partitions...")
        failed_partitions = [partitions[i] for i in failed_partition_ids]

        failed_partition_ids_1 = run_partitions(
            partitions=failed_partitions,
            partition_ids=failed_partition_ids,
            satellite_config=satellite_config,
            featurization_config=featurization_config,
            model=model,
            client=client,
            mosaiks_folder_path=mosaiks_folder_path,
        )

    # Load checkpoint files and combine
    logging.info("Loading and combining checkpoint files...")
    checkpoint_filenames = utl.get_filtered_filenames(
        folder_path=mosaiks_folder_path, prefix="df_"
    )
    combined_df = utl.load_and_combine_dataframes(
        folder_path=mosaiks_folder_path, filenames=checkpoint_filenames
    )
    combined_df = combined_df.join(points_gdf[["Lat", "Lon", "shrid"]])
    print("Dataset size in memory (MB):", combined_df.memory_usage().sum() / 1000000)

    combined_filename = "features.parquet.gzip"
    utl.save_dataframe(
        df=combined_df, file_path=mosaiks_folder_path / combined_filename
    )
