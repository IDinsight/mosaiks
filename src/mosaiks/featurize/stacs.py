import logging
from typing import List


import ee
# ee.Authenticate()
# ee.Initialize()

import dask_geopandas as dask_gpd
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pyproj
import pystac_client
import shapely
import stackstac
import torch
import wxee
from pystac.item import Item
from pystac.item_collection import ItemCollection
from torch.utils.data import DataLoader, Dataset

__all__ = ["get_dask_gdf", "fetch_image_refs", "create_data_loader"]


def get_dask_gdf(points_gdf: gpd.GeoDataFrame, chunksize: int) -> dask_gpd.GeoDataFrame:
    """
    Spatially sort and split the gdf up by the given chunksize.

    Parameters
    ----------
    points_dgdf : A GeoDataFrame with a column named "geometry" containing shapely Point objects.
    chunksize : The number of points per partition to use creating the Dask GeoDataFrame.

    Returns
    -------
    points_dgdf: Dask GeoDataFrame split into partitions of size `chunksize`.
    """

    points_gdf = sort_by_hilbert_distance(points_gdf)
    points_dgdf = dask_gpd.from_geopandas(
        points_gdf,
        chunksize=chunksize,
        sort=False,
    )

    logging.info(
        f"{chunksize} points per partition results in {len(points_dgdf.divisions)} partitions."
    )

    logging.info(
        f"Distributing {len(points_gdf)} points across {chunksize}-point partitions results in {points_dgdf.npartitions} partitions."
    )

    return points_dgdf


def fetch_image_refs(
    points_dgdf: gpd.GeoDataFrame, satellite_search_params: dict
) -> dask_gpd.GeoDataFrame:
    """
    Find a STAC item for points in the `points_dgdf` Dask GeoDataFrame. Returns a
    Dask GeoDataFrame with STAC items as a new column.

    Parameters
    ----------
    points_dgdf : A DaskGeoDataFrame with a column named "geometry" containing shapely
        Point objects.
    satellite_search_params : A dictionary containing the parameters for the STAC
        search.

    Returns
    -------
    points_dgdf: A Dask GeoDataFrame with a column named "stac_item" containing STAC
        items.
    """

    # # meta not needed at the moment, speed is adequate
    # meta = points_dgdf._meta
    # meta = meta.assign(stac_item=pd.Series([], dtype="object"))
    # meta = meta.assign(cloud_cover=pd.Series([], dtype="object"))

    if satellite_search_params["seasonal"]:
        points_gdf_with_stac = points_dgdf.map_partitions(
            fetch_seasonal_stac_items,
            satellite_name=satellite_search_params["satellite_name"],
            year=satellite_search_params["year"],
            stac_output=satellite_search_params["stac_output"],
            stac_api=satellite_search_params["stac_api"],
            # meta=meta,
        )
    else:
        points_gdf_with_stac = points_dgdf.map_partitions(
            fetch_stac_items,
            satellite_name=satellite_search_params["satellite_name"],
            search_start=satellite_search_params["search_start"],
            search_end=satellite_search_params["search_end"],
            stac_api=satellite_search_params["stac_api"],
            stac_output=satellite_search_params["stac_output"],
            # meta=meta,
        )

    return points_gdf_with_stac


def create_data_loader(
    points_gdf_with_stac: gpd.GeoDataFrame, satellite_params: dict, batch_size: int
) -> DataLoader:
    """
    Creates a PyTorch DataLoader from a GeoDataFrame with STAC items.

    Parameters
    ----------
    points_gdf_with_stac : A GeoDataFrame with the points we want to fetch imagery for + its STAC ref
    satellite_params : A dictionary of parameters for the satellite imagery to fetch
    batch_size : The batch size to use for the DataLoader

    """

    stac_item_list = points_gdf_with_stac.stac_item.tolist()
    points_list = points_gdf_with_stac[["Lon", "Lat"]].to_numpy()
    dataset = CustomDataset(
        points_list,
        stac_item_list,
        buffer=satellite_params["buffer_distance"],
        bands=satellite_params["bands"],
        resolution=satellite_params["resolution"],
        dtype=satellite_params["dtype"],
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
        pin_memory=False,
    )

    return data_loader


def sort_by_hilbert_distance(points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sort the points in the GeoDataFrame by their Hilbert distance."""

    ddf = dask_gpd.from_geopandas(points_gdf, npartitions=1)
    hilbert_distance = ddf.hilbert_distance().compute()
    points_gdf["hilbert_distance"] = hilbert_distance
    points_gdf = points_gdf.sort_values("hilbert_distance")

    return points_gdf


def fetch_seasonal_stac_items(
    points_gdf: gpd.GeoDataFrame,
    satellite_name: str,
    year: int,
    stac_api: str,
    stac_output: str = "least_cloudy",
) -> gpd.GeoDataFrame:
    """
    Takes a year as input and creates date ranges for the four seasons, runs these
    through fetch_stac_items, and concatenates the results. Can be used where-ever
    `fetch_stac_items` is used.

    Note: Winter is the first month and includes December from the previous year.

    Months of the seasons taken from [here](https://delhitourism.gov.in/delhitourism/aboutus/seasons_of_delhi.jsp) for now.
    """
    season_dict = {
        "winter": (f"{year-1}-12-01", f"{year}-01-31"),
        "spring": (f"{year}-02-01", f"{year}-03-31"),
        "summer": (f"{year}-04-01", f"{year}-09-30"),
        "autumn": (f"{year}-09-01", f"{year}-11-30"),
    }
    seasonal_gdf_list = []
    for season, dates in season_dict.items():
        search_start, search_end = dates
        season_points_gdf = fetch_stac_items(
            points_gdf=points_gdf,
            satellite_name=satellite_name,
            search_start=search_start,
            search_end=search_end,
            stac_api=stac_api,
            stac_output=stac_output,
        )

        season_points_gdf["season"] = season
        seasonal_gdf_list.append(season_points_gdf)

    combined_gdf = pd.concat(seasonal_gdf_list, axis="index")
    combined_gdf.index.name = "point_id"

    return combined_gdf


def fetch_stac_items(
    points_gdf: gpd.GeoDataFrame,
    satellite_name: str,
    search_start: str,
    search_end: str,
    stac_api: str,
    stac_output: str = "least_cloudy",
) -> gpd.GeoDataFrame:
    """
    Find the STAC item(s) that overlap each point in the `points_gdf` GeoDataFrame.

    Parameters
    ----------
    points_gdf : A GeoDataFrame
    satellite_name : Name of MPC-hosted satellite
    search_start : Date formatted as YYYY-MM-DD
    search_end : Date formatted as YYYY-MM-DD
    stac_api: The stac api that pystac should connect to
    stac_output : Whether to store "all" images found or just the "least_cloudy"

    Returns
    -------
    points_gdf: A new geopandas.GeoDataFrame with a `stac_item` column containing the
        STAC item that covers each point.
    """

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

    if len(item_collection) == 0:
        return points_gdf.assign(stac_item=None)
    else:
        # Convert ItemCollection to GeoDataFrame
        if satellite_name == "landsat-8-c2-l2":
            # For landsat: trim the shapes to fit proj:bbox
            stac_gdf = _get_trimmed_stac_shapes_gdf(item_collection)
        else:
            # For Sentinel there is no need - there is no "proj:bbox" parameter
            # which could cause STACK issues
            stac_gdf = gpd.GeoDataFrame.from_features(item_collection.to_dict())

        # add items as an extra column
        stac_gdf["stac_item"] = item_collection.items

        if stac_output == "all":
            points_gdf["stac_item"] = points_gdf.apply(
                _items_covering_point, stac_gdf=stac_gdf, axis=1
            )

        if stac_output == "least_cloudy":
            stac_gdf.sort_values(by="eo:cloud_cover", inplace=True)
            points_gdf["stac_item"] = points_gdf.apply(
                _least_cloudy_item_covering_point, sorted_stac_gdf=stac_gdf, axis=1
            )

        return points_gdf


def _get_trimmed_stac_shapes_gdf(item_collection: ItemCollection) -> gpd.GeoDataFrame:
    """
    To prevent the edge case where a point sits inside the STAC geometry
    but outwith the STAC proj:bbox shape (resulting in a dud xarray to be
    returned later), trim the STAC shapes to within the proj:bbox borders.

    Returns
    -------
    GeoDataFrame where each row is an Item and columns include cloud cover percentage
    and item shape trimmed to within proj:bbox.
    """

    rows_list = []
    for item in item_collection:

        stac_crs = item.properties["proj:epsg"]

        # get STAC geometry
        stac_geom = shapely.geometry.shape(item.geometry)

        # convert proj:bbox to polygon
        x_min_p, y_min_p, x_max_p, y_max_p = item.properties["proj:bbox"]
        image_bbox = shapely.geometry.Polygon(
            [
                [x_min_p, y_min_p],
                [x_min_p, y_max_p],
                [x_max_p, y_max_p],
                [x_max_p, y_min_p],
            ]
        )
        # convert to EPSG:4326 (to match STAC geometry)
        image_bbox = (
            gpd.GeoSeries(image_bbox).set_crs(stac_crs).to_crs(4326).geometry[0]
        )

        # trim stac_geom to only what's inside bbox
        trimmed_geom = stac_geom.intersection(image_bbox)

        row_data = {
            "eo:cloud_cover": [item.properties["eo:cloud_cover"]],
            "geometry": [trimmed_geom],
        }

        row = gpd.GeoDataFrame(row_data, crs=4326)
        rows_list.append(row)

    return pd.concat(rows_list)


def _least_cloudy_item_covering_point(
    row: gpd.GeoDataFrame, sorted_stac_gdf: gpd.GeoDataFrame
) -> Item:
    """
    Takes in a sorted dataframe of stac items and returns the
    least cloudy item that covers the current row. For use in
    `fetch_stac_items`.

    TODO: Add cloud_cover column back
    """

    items_covering_point = sorted_stac_gdf[sorted_stac_gdf.covers(row.geometry)]
    if len(items_covering_point) == 0:
        return None
    else:
        least_cloudy_item = items_covering_point.iloc[0]["stac_item"]
        return least_cloudy_item  # , least_cloudy_item.properties["eo:cloud_cover"]


def _items_covering_point(
    row: gpd.GeoDataFrame, stac_gdf: gpd.GeoDataFrame
) -> List[Item]:
    """Takes in a sorted dataframe of stac items and returns all
    stac items that cover the current row as a list. For use in
    `fetch_stac_items`

    """
    items_covering_point = stac_gdf[stac_gdf.covers(row.geometry)]
    if len(items_covering_point) == 0:
        return None
    else:
        return items_covering_point["stac_item"].tolist()


def get_stac_api(api_name: str) -> pystac_client.Client:
    """Get a STAC API client for a given API name."""

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


def minmax_normalize_image(image: torch.tensor) -> torch.tensor:

    img_min, img_max = image.min(), image.max()
    return (image - img_min) / (img_max - img_min)


class CustomDataset(Dataset):
    def __init__(
        self,
        points: np.array,
        items: List[Item],
        buffer: int,
        bands: List[str],
        resolution: int,
        dtype: str = "int16",
    ) -> None:
        """
        Parameters
        ----------
        points : Array of points to sample from
        items : List of STAC items to sample from
        buffer : Buffer in meters around each point to sample from
        bands : List of bands to sample
        resolution : Resolution of the image to sample
        dtype : Data type of the image to sample. Defaults to "int16".
            NOTE - np.uint8 results in loss of signal in the features
            and np.uint16 is not supported by PyTorch.
        """

        self.points = points
        self.items = items
        self.buffer = buffer
        self.bands = bands
        self.resolution = resolution
        self.dtype = dtype

    def __len__(self):
        """Returns the number of points in the dataset"""

        return self.points.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Parameters
        ----------
        idx :Index of the point to get imagery for

        Returns
        -------
        out_image : Image tensor of shape (C, H, W)
        """

        lon, lat = self.points[idx]
        stac_item = self.items[idx]

        if stac_item is None:
            print(f"Skipping {idx}: No STAC item given.")
            return None
        else:
            # calculate crop bounds
            crs = stac_item.properties["proj:epsg"]
            proj_latlon_to_stac = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
            x_utm, y_utm = proj_latlon_to_stac.transform(lon, lat)
            x_min, x_max = x_utm - self.buffer, x_utm + self.buffer
            y_min, y_max = y_utm - self.buffer, y_utm + self.buffer

            # get image(s) as xarray
            xarray = stackstac.stack(
                stac_item,
                assets=self.bands,
                resolution=self.resolution,
                rescale=False,
                dtype=self.dtype,
                bounds=[x_min, y_min, x_max, y_max],
                fill_value=0,
            )

            # remove the time dimension either by compositing over it or with .squeeze()
            if isinstance(stac_item, list):
                image = xarray.median(dim="time")
            else:
                image = xarray.squeeze()

            # normalise and return image
            # Note: need to catch errors for images that are all 0s
            try:
                image = image.values
                torch_image = torch.from_numpy(image).float()
                torch_image = minmax_normalize_image(torch_image)
                return torch_image
            except Exception as e:
                print(f"Skipping {idx}:", e)
                return None


def create_data_loader_GEE(
    points_gdf, satellite_params, featurization_params, batch_size
):
    """
    Creates a PyTorch DataLoader from a GeoDataFrame with STAC items.

    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        A GeoDataFrame with the points we want to fetch imagery for
    satellite_params : dict
        A dictionary of parameters for the satellite imagery to fetch
    featurization_params : dict
        A dictionary of parameters for the featurization process.
    batch_size : int
        The batch size to use for the DataLoader

    Returns
    -------
    torch.utils.data.DataLoader
    """

    points_list = points_gdf[["Lon", "Lat"]].to_numpy()
    dataset = CustomDataset_GEE(
        points=points_list,
        buffer=satellite_params["buffer_distance"],
        bands=satellite_params["bands"],
        search_start=featurization_params["satellite_search_params"]["search_start"],
        search_end=featurization_params["satellite_search_params"]["search_end"],
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
        pin_memory=False,
    )

    return data_loader


class CustomDataset_GEE(Dataset):
    def __init__(self, points, buffer, bands, search_start, search_end):
        """
        Parameters
        ----------
        points : np.array
            Array of points to sample from
        buffer : int
            Buffer in meters around each point to sample from
        bands : list
            List of bands to sample
        search_start : str
            Start date for the search
        search_end : str
            End date for the search

        Returns
        -------
        None
        """

        self.points = points
        self.buffer = buffer
        self.bands = bands
        self.search_start = search_start
        self.search_end = search_end

    def __len__(self):
        """Returns the number of points in the dataset"""

        return self.points.shape[0]

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int
            Index of the point to get imagery for

        Returns
        -------
        out_image : torch.Tensor
            Image tensor of shape (C, H, W)
        """

        lon, lat = self.points[idx]

        point = ee.Geometry.Point(lon, lat)
        crop = point.buffer(self.buffer).bounds()

        collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(crop)
            .filterDate(self.search_start, self.search_end)
            .sort("CLOUD_COVER")
        )

        least_cloudy_image = collection.first()

        xarray = least_cloudy_image.wx.to_xarray(region=crop, scale=30)
        xarray = xarray[self.bands].to_array()
        final_xarray = xarray.transpose("time", "variable", "y", "x").squeeze()

        # normalise and return image
        try:
            image = final_xarray.values
            torch_image = torch.from_numpy(image).float()
            torch_image = minmax_normalize_image(torch_image)
            return torch_image
        except Exception as e:
            print(f"Skipping {idx}:", e)
            return None
