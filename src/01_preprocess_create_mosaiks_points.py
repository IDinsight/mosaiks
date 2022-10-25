from custom.mosaiks_points import (
    create_gdf_of_enclosed_points,
    plot_selected_points,
    points_to_latlon_df,
)
from custom.shrug_data import lengthen_shapefile_ID_names
from custom.utils import load_gdf


def main():
    """Create a GeoDataFrame of points enclosed by the SHRUG rural shapes and save to file."""

    # load preprocessed SHRUG keys with shapes - takes around 24s
    shrug_key_geoms = load_gdf(
        "01_preprocessed/SHRUG/shrug_pc11r_key_with_shapes",
        "shrug_pc11r_key_with_shapes.shp",
    )
    shrug_key_geoms = lengthen_shapefile_ID_names(shrug_key_geoms)

    # optional - subset shapes to a smaller area. E.g. to 6 focus states:
    # shrug_r_matched = shrug_r_matched[shrug_r_matched["pc11_state_id"].isin([8, 16, 18, 20, 22, 23, 28])]

    # create lattice of point coordinates, keeping those that sit within our shrug shapes - takes around 50s
    selected_points_gdf = create_gdf_of_enclosed_points(
        shrug_key_geoms,
        step=0.05,
        pre_calc_bounds="india",  # remove this if selection is not all-india
    )

    # save plot and csv. Saves to "data/01_preprocessed/mosaiks_request_points"
    file_name = "INDIA_SHRUG_request_points"
    plot_selected_points(selected_points_gdf, file_name=file_name)
    points_to_latlon_df(selected_points_gdf.geometry, file_name)


if __name__ == "__main__":
    main()