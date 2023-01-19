import os

import dask_geopandas as dask_gpd
import geopandas as gpd
import pandas as pd
import planetary_computer
import pyproj
import pystac_client
import shapely.geometry
import stackstac
import torch
from torch.utils.data import DataLoader, Dataset


def fetch_image_refs(points_gdf, n_partitions, satellite_image_params):
    points_gdf = sort_by_hilbert_distance(points_gdf)
    points_dgdf = dask_gpd.from_geopandas(
        points_gdf, npartitions=n_partitions, sort=False
    )

    meta = points_dgdf._meta
    meta = meta.assign(stac_item=pd.Series([], dtype="object"))
    meta = meta.assign(cloud_cover=pd.Series([], dtype="object"))

    points_gdf_with_stac = points_dgdf.map_partitions(
        fetch_stac_items,
        **satellite_image_params,
        meta=meta,
    )

    return points_gdf_with_stac  # .compute()


def create_data_loader(points_gdf_with_stac, satellite_params, batch_size):

    stac_item_list = points_gdf_with_stac.stac_item.tolist()
    points_list = points_gdf_with_stac[["Lon", "Lat"]].to_numpy()
    dataset = CustomDataset(
        points_list,
        stac_item_list,
        buffer=satellite_params["buffer_distance"],
        bands=satellite_params["bands"],
        resolution=satellite_params["resolution"],
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
    )

    return data_loader


def sort_by_hilbert_distance(points_gdf):

    ddf = dask_gpd.from_geopandas(points_gdf, npartitions=1)
    hilbert_distance = ddf.hilbert_distance().compute()
    points_gdf["hilbert_distance"] = hilbert_distance
    points_gdf = points_gdf.sort_values("hilbert_distance")

    return points_gdf


def fetch_stac_items(
    points_gdf, satellite_name, search_start, search_end, stac_api, stac_output
):
    """
    Find a STAC item for points in the `points_gdf` GeoDataFrame.

    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        A GeoDataFrame
    satellite_name : string
        Name of MPC-hosted satellite
    search_start : string
        Date formatted as YYYY-MM-DD
    search_end : string
        Date formatted as YYYY-MM-DD
    stac_output : string
        Whether to store "all" images found or just the "least_cloudy"
    stac_api: string
        The stac api that pystac should connect to

    Returns
    -------
    geopandas.GeoDataFrame
        A new geopandas.GeoDataFrame with a `stac_item` column containing the STAC
        item that covers each point.
    """

    # Get the images that cover any of the given points
    # mpc_stac_api = pystac_client.Client.open(
    #    "https://planetarycomputer.microsoft.com/api/stac/v1",
    #    modifier=planetary_computer.sign_inplace,
    # )

    stac_api = get_stac_api(stac_api)

    # change to just union of points?
    bounding_poly = shapely.geometry.mapping(points_gdf.unary_union.convex_hull)

    search_results = stac_api.search(
        collections=[satellite_name],
        intersects=bounding_poly,
        datetime=[search_start, search_end],
        query={"eo:cloud_cover": {"lt": 10}},
        limit=500,  # this limit seems arbitrary
    )
    item_collection = search_results.item_collection()
    items = search_results.get_all_items_as_dict()["features"]

    # Select only images that cover each point and add the STAC item for the
    # cleast cloudy image to the df
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
        least_cloudy_item_cover = []
        for point in point_geom_list:
            items_covering_point = items_gdf[items_gdf.covers(point)]
            if len(items_covering_point) == 0:
                least_cloudy_item = None
            else:
                least_cloudy_item_id = items_covering_point.index[0]
                least_cloudy_item_cover = items_covering_point["eo:cloud_cover"].iloc[0]
                # FIX THIS JANK (picks out the correct item based on the ID):
                least_cloudy_item = [
                    item
                    for item in item_collection.items
                    if item.id == least_cloudy_item_id
                ][0]

            least_cloudy_items.append(least_cloudy_item)

        points_gdf = points_gdf.assign(stac_item=least_cloudy_items)
        points_gdf = points_gdf.assign(cloud_cover=least_cloudy_item_cover)

    # if all images requested that cover a point (not just that with the lowest
    # cloud cover), then return a list of STAC items that cover each point
    # instead of just one
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


def get_stac_api(api_name):
    if api_name == "planetary-compute":
        stac_api = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
    elif api_name == "earth":
        stac_api = pystac_client.Client.open(
            "https://earth-search.aws.element84.com/v0"
        )
    else:
        raise NotImplementedError(f"STAC api {api_name} is not implemented")

    return stac_api


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
            if isinstance(stac_item, list):
                cropped_xarray = cropped_xarray.median(dim="time").compute()
                out_image = cropped_xarray.data
            else:
                out_image = cropped_xarray.data.squeeze()

            # 4. Min-max normalize pixel values to [0,1]
            #    !! From MOSAIKS code... Check the effect of this !!
            out_image = (out_image - out_image.min()) / (
                out_image.max() - out_image.min()
            )

            # 5. Finally, convert to pytorch tensor
            out_image = torch.from_numpy(out_image).float()

            return out_image
