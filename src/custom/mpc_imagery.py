import geopandas

import pystac_client
import shapely.geometry
import planetary_computer as pc

from pystac import Item
import stackstac
import pyproj

import torch
from torch.utils.data import Dataset, DataLoader


## TEMP: Parameters for CustomDataset
satellite = "landsat-8-c2-l2"
bands = [
    # "SR_B1", # Coastal/Aerosol Band (B1)
    "SR_B2",  # Blue Band (B2)
    "SR_B3",  # Green Band (B3)
    "SR_B4",  # Red Band (B4)
    "SR_B5",  # Near Infrared Band 0.8 (B5)
    "SR_B6",  # Short-wave Infrared Band 1.6 (B6)
    "SR_B7",  # Short-wave Infrared Band 2.2 (B7)
]

resolution = 30
min_image_edge = 6


class CustomDataset(Dataset):
    def __init__(self, points, items, buffer=0.005):
        self.points = points
        self.items = items
        self.buffer = buffer

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
                    assets=bands,
                    resolution=resolution,
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


def query(
    points,
    satellite="landsat-8-c2-l2",
    search_start="2018-01-01",
    search_end="2018-12-31",
):
    """
    Find a STAC item for points in the `points` GeoDataFrame

    Parameters
    ----------
    points : geopandas.GeoDataFrame
        A GeoDataFrame

    Returns
    -------
    geopandas.GeoDataFrame
        A new geopandas.GeoDataFrame with a `stac_item` column containing the STAC
        item that covers each point.
    """

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    bounding_poly = shapely.geometry.mapping(points.unary_union.convex_hull)

    search = catalog.search(
        collections=[satellite],
        intersects=bounding_poly,  # change to just union of points?
        datetime=[search_start, search_end],
        query={"eo:cloud_cover": {"lt": 10}},
        # limit=500,
    )
    ic = search.get_all_items_as_dict()

    features = ic["features"]
    features_d = {item["id"]: item for item in features}

    ids = []
    data = {
        "eo:cloud_cover": [],
        "geometry": [],
    }

    for item in features:
        ids.append(item["id"])
        data["eo:cloud_cover"].append(item["properties"]["eo:cloud_cover"])
        data["geometry"].append(shapely.geometry.shape(item["geometry"]))

    items = geopandas.GeoDataFrame(data, index=ids, geometry="geometry").sort_values(
        "eo:cloud_cover"
    )
    point_list = points.geometry.tolist()

    point_items = []
    for point in point_list:
        covered_by = items[items.covers(point)]
        if len(covered_by):
            point_items.append(features_d[covered_by.index[0]])
        else:
            # No images match filter over this point
            point_items.append(None)

    return points.assign(stac_item=point_items)


def match_images_and_points(points_gdf_with_stac):

    points_gdf_with_stac_clean = points_gdf_with_stac.dropna(subset=["stac_item"])

    matched_image_items = [
        pc.sign(Item.from_dict(point))
        for point in points_gdf_with_stac_clean.stac_item.tolist()
    ]
    matched_points_list = points_gdf_with_stac_clean[["lon", "lat"]].to_numpy()

    return matched_image_items, matched_points_list
