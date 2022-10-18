from custom.shrug_data import (
    merge_shapes_and_keys,
    change_shapefile_IDs_to_int,
    clean_shrug_rural_keys,
    load_shrug_rural_keys,
    load_shrug_shapefiles,
    shorten_keys_ID_names,
)
from custom.utils import save_gdf


def main():
    """Preprocess SHRUG rural keys by adding shapes and saving to file."""

    # Load the SHRUG shapes
    shrug_village_shapes = load_shrug_shapefiles(level="village")
    shrug_village_shapes = change_shapefile_IDs_to_int(shrug_village_shapes)

    # Load the SHRUG 2011 rural keys
    shrug_pc11r_keys = load_shrug_rural_keys()
    shrug_pc11r_keys_clean = clean_shrug_rural_keys(shrug_pc11r_keys)
    shrug_pc11r_keys_clean = shorten_keys_ID_names(shrug_pc11r_keys_clean)

    # Add the shapes to the keys
    shrug_pc11r_keys_geoms = merge_shapes_and_keys(
        shrug_village_shapes,
        shrug_pc11r_keys_clean,
    )

    # Save the keys with shapes to file
    save_gdf(
        gdf=shrug_pc11r_keys_geoms,
        folder_name="01_preprocessed/SHRUG/shrug_pc11r_key_with_shapes",
        file_name="shrug_pc11r_key_with_shapes.shp",
    )


if __name__ == "__main__":
    main()
