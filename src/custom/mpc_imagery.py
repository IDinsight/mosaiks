# TO-DO

# change print() to logging.

# change image cropping to caleb's neater way? can that be parallelized?

# change convex-hull to just union of points? (~line 118)

## NOTE: this method of finding the least cloudy image in batches of points
## does no stitching or compositing and so can't handle points
## that are close to the edge of an image well!

# limit on line 125 seems arbitrary

import geopandas as gpd
import dask_geopandas as dask_gpd

import pystac_client
import shapely.geometry
import planetary_computer

import stackstac
import pyproj

import torch
from torch.utils.data import Dataset


def sort_by_hilbert_distance(points_gdf):

    ddf = dask_gpd.from_geopandas(points_gdf, npartitions=1)
    hd = ddf.hilbert_distance().compute()
    points_gdf["hd"] = hd
    points_gdf = points_gdf.sort_values("hd")

    del ddf
    del hd

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
            # print(f"{idx} Error: no stac item passed\n")
            return None
        else:
            try:
                stack = stackstac.stack(
                    stac_item,
                    assets=self.bands,
                    resolution=self.resolution,
                )
                
                # make composite if multiple images returned
                if type(stac_item) is list:
                    stack = stack.median(dim="time").compute()
                
                x_utm, y_utm = pyproj.Proj(stack.crs)(lon, lat)
                x_min, x_max = x_utm - self.buffer, x_utm + self.buffer
                y_min, y_max = y_utm - self.buffer, y_utm + self.buffer

                aoi = stack.loc[..., y_max:y_min, x_min:x_max]
                data = aoi.compute(
                    # scheduler="single-threaded"
                )
                
                # Min-max normalize pixel values to [0,1]?
                out_image = (out_image - out_image.min()) / (
                    out_image.max() - out_image.min()
                )

            except BaseException as e:
                print(f"{idx} Error: {str(e)}")
                return None

            out_image = torch.from_numpy(out_image).float()

            return out_image


def fetch_stac_items(points_gdf, satellite, search_start, search_end, least_cloudy_only=True):
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

    Returns
    -------
    geopandas.GeoDataFrame
        A new geopandas.GeoDataFrame with a `stac_item` column containing the STAC
        item that covers each point.
    """

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

    if least_cloudy_only:
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
        
    else: # return a list of STAC image IDs that cover each point 
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


