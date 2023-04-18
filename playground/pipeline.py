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

# Setup Rasterio
rasterio_config = utl.load_yaml_config("rasterioc_config.yaml")
os.environ.update(rasterio_config)

# Setup Dask Cluster and Client
client = get_dask_client(kind="local")

# Load params
featurization_params = utl.load_yaml_config("featurisation.yaml")
satellite_config = utl.load_yaml_config("satellite_config.yaml")
satellite_config = satellite_config[
    featurization_params["satellite_search_params"]["satellite_name"]
]

# Load point coords
points_gdf = utl.load_df_w_latlons_to_gdf(
    dataset_name=featurization_params["coord_set_name"]
)
points_dgdf = get_dask_gdf(points_gdf, featurization_params["dask"]["chunksize"])

# Fetch image stac refs
points_gdf_with_stac = fetch_image_refs(
    points_dgdf, featurization_params["satellite_search_params"]
)
partitions = points_gdf_with_stac.to_delayed()

# setup model
model = RCF(
    featurization_params["num_features"],
    featurization_params["kernel_size"],
    len(satellite_config["bands"]),
)

# Run in parallel
mosaiks_folder_path = utl.make_features_path_from_dict(
    featurization_params, featurization_params["coord_set_name"]
)
failed_partition_ids = run_partitions(
    partitions,
    satellite_config,
    featurization_params,
    model,
    client,
    mosaiks_folder_path,
)

# Re-run failed partitions
failed_partitions = [partitions[i] for i in failed_partition_ids]
failed_partition_ids_1 = run_partitions(
    partitions=failed_partitions,
    partition_ids=failed_partition_ids,
    satellite_config=satellite_config,
    featurization_params=featurization_params,
    model=model,
    client=client,
    mosaiks_folder_path=mosaiks_folder_path,
)

# Load checkpoint files and combine
checkpoint_filenames = utl.get_filtered_filenames(mosaiks_folder_path, prefix="df_")
combined_df = utl.load_and_combine_dataframes(mosaiks_folder_path, checkpoint_filenames)
combined_df = combined_df.join(points_gdf[["Lat", "Lon", "shrid"]])
print("Dataset size in memory (MB):", combined_df.memory_usage().sum() / 1000000)

combined_filename = "features.parquet.gzip"
utl.save_dataframe(combined_df, file_path=mosaiks_folder_path / combined_filename)
