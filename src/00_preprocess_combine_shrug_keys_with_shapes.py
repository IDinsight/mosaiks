import logging
import pandas as pd
from custom.shrug_data import (
    load_shrug_shapefiles,
    load_shrug_urban_keys,
    load_shrug_rural_keys,
    clean_shrug_rural_keys,
    change_shapefile_IDs_to_int,
)
from custom.utils import save_gdf

logging.basicConfig(level=logging.INFO)


def main():
    """Preprocess SHRUG keys (rural and urban) by adding shapes and saving to file."""

    logging.info("Loading SHRUG shapes")
    # Note: "Village" shapefiles also include towns!
    shrug_shapes = load_shrug_shapefiles(level="village")
    shrug_shapes = change_shapefile_IDs_to_int(shrug_shapes)

    logging.info("Loading rural keys")
    shrug_r_keys = load_shrug_rural_keys()
    shrug_r_keys_clean = clean_shrug_rural_keys(shrug_r_keys)
    shrug_r_keys_clean = shrug_r_keys_clean.rename(
        columns={"pc11_village_id": "pc11_tv_id"}
    )
    shrug_r_keys_clean["is_urban"] = 0  # preserve urban/rural information
    shrug_r_keys_clean = shrug_r_keys_clean[["pc11_tv_id", "is_urban", "shrid"]]

    logging.info("Loading urban keys")
    shrug_u_keys = load_shrug_urban_keys()
    shrug_u_keys = shrug_u_keys.rename(columns={"pc11_town_id": "pc11_tv_id"})
    shrug_u_keys["is_urban"] = 1
    shrug_u_keys = shrug_u_keys[["pc11_tv_id", "is_urban", "shrid"]]

    logging.info("Combining rural and urban keys")
    shrug_all_keys = pd.concat([shrug_r_keys_clean, shrug_u_keys])
    shrug_all_keys = shrug_all_keys.sort_values(by="pc11_tv_id")

    logging.info("Combining keys with shapes")
    shrug_all_keys_with_shapes = pd.merge(
        shrug_shapes,
        shrug_all_keys,
        on="pc11_tv_id",
        how="inner",
    )

    logging.info(
        "Saving new keys to `01_preprocessed/SHRUG/shrug_all_keys_with_shapes/`"
    )
    save_gdf(
        gdf=shrug_all_keys_with_shapes,
        folder_name="01_preprocessed/SHRUG/shrug_all_keys_with_shapes",
        file_name="shrug_all_keys_with_shapes.shp",
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
