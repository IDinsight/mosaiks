# TO-DO

# Contents of main() should be a separate function

# Add
# 1. path_to_shapes
# 2. step
# 3. pre_calc_bounds
# 4. subset to 6 states var
# as script arguments

import logging
from custom.mosaiks_points import (
    create_gdf_of_enclosed_points,
    plot_selected_points,
    points_to_latlon_df,
)
from custom.utils import load_gdf

logging.basicConfig(level=logging.INFO)


def main():
    """Create a GeoDataFrame of points enclosed by the SHRUG shapes and save to file."""

    logging.info("Load preprocessed SHRUG keys with shapes")  # takes around 24s
    shrug_key_geoms = load_gdf(
        "01_preprocessed/SHRUG/shrug_all_keys_with_shapes",
        "shrug_all_keys_with_shapes.shp",
    )

    # optional - subset shapes to a smaller area. E.g. to 6 focus states:
    # shrug_key_geoms = shrug_key_geoms[shrug_key_geoms["pc11_s_id"].isin([8, 16, 18, 20, 22, 23, 28])]

    logging.info(
        "Create lattice of point coordinates, keeping those that sit within our shrug shapes"
    )  # takes around 50s
    selected_points_gdf = create_gdf_of_enclosed_points(
        shrug_key_geoms,
        step=0.05,
        pre_calc_bounds="india",
    )

    logging.info("Save plot and csv to `data/01_preprocessed/mosaiks_request_points`")
    file_name = "urb_rur_request_points"
    plot_selected_points(
        selected_points_gdf, color_column="pc11_s_id", file_name=file_name
    )
    points_to_latlon_df(selected_points_gdf.geometry, file_name)
    logging.info("Done.")


if __name__ == "__main__":
    main()
