# TO-DO
# Change image cropping to caleb's neater way

# change convex-hull to just union of points? (~line 118)

### NOTE: this method of finding the least cloudy image in batches of points
### does no stitching or compositing and so can't handle points 
### that are close to the edge of an image well!

### limit on line 125 seems arbitrary
    
import geopandas
import dask_geopandas

import pystac_client
import shapely.geometry
import planetary_computer

from pystac import Item
import stackstac
import pyproj

import torch
from torch.utils.data import Dataset, DataLoader


def filter_points_with_buffer(points_gdf, shape, buffer_distance):
    # This buffer ensures that no points are take at the border
    # which would lead to duplication with neighboring countries

    return points_gdf[points_gdf.within(shape.unary_union.buffer(buffer_distance))]



def sort_by_hilbert_distance(points_gdf):

    ddf = dask_geopandas.from_geopandas(points_gdf, npartitions=1)
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
        fn = self.items[idx]

        if fn is None:
            return None
        else:
            try:
                stack = stackstac.stack(
                    fn,
                    assets=self.bands,
                    resolution=self.resolution,
                )
                x_min, y_min = pyproj.Proj(stack.crs)(
                    lon - self.buffer, lat - self.buffer
                )
                x_max, y_max = pyproj.Proj(stack.crs)(
                    lon + self.buffer, lat + self.buffer
                )
                aoi = stack.loc[..., y_max:y_min, x_min:x_max]
                data = aoi.compute(
                    # scheduler="single-threaded"
                )
                out_image = data.data
                out_image = ((out_image - out_image.min())) / (
                    out_image.max() - out_image.min()
                )
            except ValueError:
                pass
            
            out_image = torch.from_numpy(out_image).float()
            
            return out_image


def fetch_least_cloudy_stac_items(
    points_gdf,
    satellite,
    search_start,
    search_end
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

    Returns
    -------
    geopandas.GeoDataFrame
        A new geopandas.GeoDataFrame with a `stac_item` column containing the STAC
        item that covers each point.
    """

    mpc_stac_api = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    # change to just union of points?
    bounding_poly = shapely.geometry.mapping(points_gdf.unary_union.convex_hull)

    search_results = mpc_stac_api.search(
        collections=[satellite],
        intersects=bounding_poly,
        datetime=[search_start, search_end],
        query={"eo:cloud_cover": {"lt": 10}},
        limit=500, ### this limit seems arbitrary ###
    )
    item_collection = search_results.get_all_items()

    items = search_results.get_all_items_as_dict()["features"]
    items_id_dict = {item["id"]: item for item in items}

    # Create GeoDataFrame of image shapes, ids, and cloud cover tags
    id_list = []
    cloud_cover_list = []
    image_geom_list = []
    for item in items:
        id_list.append(item["id"])
        cloud_cover_list.append(item["properties"]["eo:cloud_cover"])
        image_geom_list.append(shapely.geometry.shape(item["geometry"]))
        
    items_gdf = geopandas.GeoDataFrame(
        {"eo:cloud_cover":cloud_cover_list}, 
        index=id_list, 
        geometry=image_geom_list
    )

    # sort by cloud cover so we can find the least cloudy image for each point
    items_gdf.sort_values("eo:cloud_cover", inplace=True)    

    # associate each point with the least cloudy image that covers it
    ### NOTE: this method does no stitching or compositing and so can't handle points 
    ### that are close to the edge of an image well!
    point_geom_list = points_gdf.geometry.tolist()
    
    least_cloudy_items = []
    for point in point_geom_list:
        items_covering_point = items_gdf[items_gdf.covers(point)]
        if len(items_covering_point)==0:
            least_cloudy_item = None
        else:
            least_cloudy_item_id = items_covering_point.index[0]
            # fix this jank (picks out the correct item based on the ID):
            least_cloudy_item = [item for item in item_collection.items if item.id==least_cloudy_item_id][0]
            # least_cloudy_item = items_id_dict[least_cloudy_item_id]
            
        least_cloudy_items.append(least_cloudy_item)

    return points_gdf.assign(stac_item=least_cloudy_items)

### OLD
# def items_dict_to_stac_items(items_dict):
#     items = []
#     for item_dict in items_dict:
#         items.append(Item.from_dict(item_dict))
#     return items
    

# def sign_stac_items(stac_item_list):
#     """Expects a list of stac item dicts"""
    
#     signed_stac_items = []
#     for stac_item in stac_item_list:
#         signed_stac_items.append(planetary_computer.sign(stac_item))
    
#     return signed_stac_items
