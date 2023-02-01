# TO-DO

# change print() to logging.

# change convex-hull to just union of points? (~line 118)

## NOTE: this method of finding the least cloudy image in batches of points
## does no stitching or compositing and so can't handle points
## that are close to the edge of an image well!

# modularise out the fetching and cloud-cover selection function...

# limit on line 136 seems arbitrary

import dask_geopandas as dask_gpd
import geopandas as gpd
import planetary_computer
import pyproj
import pystac_client
import shapely.geometry
import stackstac
import torch
from torch.utils.data import Dataset


def sort_by_hilbert_distance(points_gdf):

    ddf = dask_gpd.from_geopandas(points_gdf, npartitions=1)
    hilbert_distance = ddf.hilbert_distance().compute()
    points_gdf["hilbert_distance"] = hilbert_distance
    points_gdf = points_gdf.sort_values("hilbert_distance")

    del ddf
    del hilbert_distance

    return points_gdf


class CustomDataset(Dataset):
    def __init__(self, points, items, buffer, bands, resolution):
        self.points = points
        self.items = items
        self.buffer = buffer
        self.bands = bands
        self.resolution = resolution

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):

        lon, lat = self.points[idx]
        stac_item = self.items[idx]

        if stac_item is None:
            return None
        else:
            try:
                # 1. Fetch image(s)
                xarray = stackstac.stack(
                    stac_item, assets=self.bands, resolution=self.resolution
                )

                # 2. Crop image(s) - WARNING: VERY SLOW if multiple images are stacked.
                x_utm, y_utm = pyproj.Proj(xarray.crs)(lon, lat)
                x_min, x_max = x_utm - self.buffer, x_utm + self.buffer
                y_min, y_max = y_utm - self.buffer, y_utm + self.buffer

                aoi = xarray.loc[..., y_max:y_min, x_min:x_max]
                cropped_xarray = aoi.compute()

                # 2.5 Composite if there are multiple images across time
                # 3. Convert to numpy
                if type(stac_item) == list:
                    cropped_xarray = cropped_xarray.median(dim="time").compute()
                    out_image = cropped_xarray.data
                else:
                    out_image = cropped_xarray.data.squeeze()

                # 4. Min-max normalize pixel values to [0,1]
                #    !! From MOSAIKS code... Check the effect of this !!
                out_image = (out_image - out_image.min()) / (
                    out_image.max() - out_image.min()
                )

            except BaseException as e:
                print(f"{idx} Error: {str(e)}")
                return None

            # 5. Finally, convert to pytorch tensor
            out_image = torch.from_numpy(out_image).float()

            return out_image


# TODO: Make this function not be awful. Modularise out the fetching and cloud-cover selection.
# TODO: This function's logic re. first getting the ID then getting the actual item seems repeated/not good. Fix.
def fetch_stac_items(
    points_gdf, satellite, search_start, search_end, stac_output="least_cloudy"
):
    """
    Find a STAC item for points in the `points_gdf` GeoDataFrame

    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        A GeoDataFrame
    satellite : string
        Name of MPC-hosted satellite
    search_start : string
        Date formatted as YYYY-MM-DD
    search_end : string
        Date formatted as YYYY-MM-DD
    stac_output : string
        Whether to store "all" images found or just the "least_cloudy"

    Returns
    -------
    geopandas.GeoDataFrame
        A new geopandas.GeoDataFrame with a `stac_item` column containing the STAC
        item that covers each point.
    """

    # Get the images that cover any of the given points
    mpc_stac_api = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # change to just union of points?
    bounding_poly = shapely.geometry.mapping(points_gdf.unary_union.convex_hull)

    search_results = mpc_stac_api.search(
        collections=[satellite],
        intersects=bounding_poly,
        datetime=[search_start, search_end],
        query={"eo:cloud_cover": {"lt": 10}},
        limit=500,  # this limit seems arbitrary
    )
    item_collection = search_results.item_collection()
    items = search_results.get_all_items_as_dict()["features"]

    # Select only images that cover each point and add the STAC item for the least cloudy
    # image to the df
    if stac_output == "least_cloudy":
        # Create GeoDataFrame of image shapes, ids, and cloud cover tags
        id_list = []
        cloud_cover_list = []
        image_geom_list = []
        for item in items:
            id_list.append(item["id"])
            cloud_cover_list.append(item["properties"]["eo:cloud_cover"])
            image_geom_list.append(shapely.geometry.shape(item["geometry"]))

        items_gdf = gpd.GeoDataFrame(
            {"eo:cloud_cover": cloud_cover_list},
            index=id_list,
            geometry=image_geom_list,
        )

        # sort by cloud cover so we can find the least cloudy image for each point
        items_gdf.sort_values("eo:cloud_cover", inplace=True)

        # associate each point with the least cloudy image that covers it
        # NOTE: this method does no stitching or compositing and so can't handle points
        # that are close to the edge of an image well!
        point_geom_list = points_gdf.geometry.tolist()

        least_cloudy_items = []
        for point in point_geom_list:
            items_covering_point = items_gdf[items_gdf.covers(point)]
            if len(items_covering_point) == 0:
                least_cloudy_item = None
            else:
                least_cloudy_item_id = items_covering_point.index[0]
                # FIX THIS JANK (picks out the correct item based on the ID):
                least_cloudy_item = [
                    item
                    for item in item_collection.items
                    if item.id == least_cloudy_item_id
                ][0]

            least_cloudy_items.append(least_cloudy_item)

        points_gdf = points_gdf.assign(stac_item=least_cloudy_items)

    # if all images requested that cover a point (not just that with the lowest cloud cover),
    # then return a list of STAC items that cover each point instead of just one
    if stac_output == "all":
        point_geom_list = points_gdf.geometry.tolist()

        id_list = []
        image_geom_list = []
        for item in items:
            id_list.append(item["id"])
            image_geom_list.append(shapely.geometry.shape(item["geometry"]))

        items_gdf = gpd.GeoDataFrame(
            index=id_list,
            geometry=image_geom_list,
        )

        list_of_items_covering_points = []

        for point in point_geom_list:
            items_covering_point_geoms = items_gdf[items_gdf.covers(point)]
            if len(items_covering_point_geoms) == 0:
                items_covering_point = None
            else:
                # RETURN THE ACTUAL STAC ITEM HERE
                IDs_items_covering_point = list(items_covering_point_geoms.index)
                # FIX THIS JANK (picks out the correct item based on the ID):
                items_covering_point = [
                    item
                    for item in item_collection.items
                    if item.id in IDs_items_covering_point
                ]
            list_of_items_covering_points.append(items_covering_point)

        points_gdf = points_gdf.assign(stac_item=list_of_items_covering_points)

    return points_gdf