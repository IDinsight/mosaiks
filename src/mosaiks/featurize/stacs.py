import dask_geopandas as dask_gpd
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pyproj
import pystac_client
import shapely.geometry
import stackstac
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

__all__ = ["fetch_image_refs", "create_data_loader", "get_an_image"]


def fetch_image_refs(points_gdf, n_partitions, satellite_image_params):
    """
    Find a STAC item for points in the `points_gdf` GeoDataFrame.

    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        A GeoDataFrame with the points we want to fetch imagery for
    n_partitions : int
        Number of partitions to use for the dask geopandas dataframe
    satellite_image_params : dict
        A dictionary of parameters for the satellite imagery to fetch

    Returns
    -------
    geopandas.GeoDataFrame
    """
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

    return points_gdf_with_stac


def create_data_loader(points_gdf_with_stac, satellite_params, batch_size):
    """
    Create a PyTorch DataLoader for the satellite imagery.

    Parameters
    ----------
    points_gdf_with_stac : geopandas.GeoDataFrame
        A GeoDataFrame with the points we want to fetch imagery for + its STAC ref
    satellite_params : dict
        A dictionary of parameters for the satellite imagery to fetch
    batch_size : int
        The batch size to use for the DataLoader

    Returns
    -------
    torch.utils.data.DataLoader
    """

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
        collate_fn=lambda x: x,
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


def get_an_image(lon, lat, stac_item, idx, params):

    bands = params["bands"]
    resolution = params["resolution"]
    buffer = params["buffer_distance"]

    image_array = stackstac.stack(stac_item, assets=bands, resolution=resolution)

    x_utm, y_utm = pyproj.Proj(image_array.crs)(lon, lat)
    x_min, x_max = x_utm - buffer, x_utm + buffer
    y_min, y_max = y_utm - buffer, y_utm + buffer

    cropped_xarray = image_array.loc[..., y_max:y_min, x_min:x_max]

    if isinstance(stac_item, list):
        cropped_xarray = cropped_xarray.median(dim="time")
        out_image = cropped_xarray.data
    else:
        out_image = cropped_xarray.data.squeeze()

    out_image = (out_image - out_image.min()) / (out_image.max() - out_image.min())

    out_image = torch.from_numpy(out_image.compute()).float()

    return out_image


class CustomDataset(Dataset):
    def __init__(
        self,
        points,
        items,
        buffer,
        bands,
        resolution,
        transforms=T.Compose(
            [T.ToTensor(), T.Resize((120, 120)), T.ConvertImageDtype(torch.float32)]
        ),
    ):
        """
        Parameters
        ----------
        points : np.array
            Array of points to sample from
        items : list
            List of STAC items to sample from
        buffer : int
            Buffer in meters around each point to sample from
        bands : list
            List of bands to sample
        resolution : int
            Resolution of the image to sample
        transforms : torchvision.transforms
            Transforms to apply to the image

        Returns
        -------
        None
        """
        self.points = points
        self.items = items
        self.buffer = buffer
        self.bands = bands
        self.resolution = resolution
        self.transforms = transforms

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
                stac_item,
                assets=self.bands,
                resolution=self.resolution,
                rescale=False,
                dtype=np.uint8,
                fill_value=0,
                # chunksize=(-1, 1, 12000, 12000),
            )

            xarray = xarray.transpose("y", "x", "band", "time")

            # 2. Crop image(s) - WARNING: VERY SLOW if multiple images are stacked.
            x_utm, y_utm = pyproj.Proj(xarray.crs)(lon, lat)
            x_min, x_max = x_utm - self.buffer, x_utm + self.buffer
            y_min, y_max = y_utm - self.buffer, y_utm + self.buffer

            cropped_xarray = xarray.loc[y_max:y_min, x_min:x_max, ...].compute()

            # 2.5 Composite if there are multiple images across time
            # 3. Convert to numpy
            if isinstance(stac_item, list):
                out_image = cropped_xarray.median(dim="time")  # .compute()
            else:
                out_image = cropped_xarray.squeeze()  # .compute()

            out_image = self.transforms(out_image.values)

            # 5. Finally, convert to pytorch tensor
            # out_image = torch.from_numpy(out_image).float()

            return out_image
