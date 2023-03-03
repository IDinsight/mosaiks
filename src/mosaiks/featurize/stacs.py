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
from torch.utils.data import DataLoader, Dataset

__all__ = ["fetch_image_refs", "create_data_loader"]


def fetch_image_refs(points_gdf, n_partitions, satellite_search_params):
    """
    Find a STAC item for points in the `points_gdf` GeoDataFrame.

    Takes a GeoDataFrame of points and returns a GeoDataFrame with STAC items.
    Uses dask to parallelize the STAC search.

    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        A GeoDataFrame with a column named "geometry" containing shapely Point objects.
    n_partitions : int
        The number of partitions to use when creating the Dask GeoDataFrame.
    satellite_search_params : dict
        A dictionary containing the parameters for the STAC search.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame with a column named "stac_item" containing STAC items.
    """

    points_gdf = sort_by_hilbert_distance(points_gdf)
    points_dgdf = dask_gpd.from_geopandas(
        points_gdf,
        npartitions=n_partitions,
        sort=False,
    )

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


def create_data_loader(points_gdf_with_stac, satellite_params, batch_size):
    """
    Creates a PyTorch DataLoader from a GeoDataFrame with STAC items.

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
        collate_fn=lambda x: x,
        pin_memory=False,
    )

    return data_loader


def sort_by_hilbert_distance(points_gdf):
    """Sort the points in the GeoDataFrame by their Hilbert distance."""

    ddf = dask_gpd.from_geopandas(points_gdf, npartitions=1)
    hilbert_distance = ddf.hilbert_distance().compute()
    points_gdf["hilbert_distance"] = hilbert_distance
    points_gdf = points_gdf.sort_values("hilbert_distance")

    return points_gdf


def fetch_seasonal_stac_items(
    points_gdf,
    satellite_name,
    year,
    stac_api,
    stac_output="least_cloudy",
):
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
    points_gdf,
    satellite_name,
    search_start,
    search_end,
    stac_api,
    stac_output="least_cloudy",
):
    """
    Find the STAC item(s) that overlap each point in the `points_gdf` GeoDataFrame.

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
    stac_api: string
        The stac api that pystac should connect to
    stac_output : string
        Whether to store "all" images found or just the "least_cloudy"

    Returns
    -------
    geopandas.GeoDataFrame
        A new geopandas.GeoDataFrame with a `stac_item` column containing the STAC
        item that covers each point.
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
        # # 1. using original STAC shapes:
        # stac_gdf = gpd.GeoDataFrame.from_features(item_collection.to_dict())
        # 2. trimming the shapes to fit bbox
        stac_gdf = _get_trimmed_stac_shapes_gdf(item_collection)

        # add items as an extra column
        stac_gdf["stac_item"] = item_collection.items

        if stac_output == "all":
            points_gdf["stac_item"] = points_gdf.apply(
                _items_covering_point,
                stac_gdf=stac_gdf,
                axis=1,
            )

        if stac_output == "least_cloudy":
            stac_gdf.sort_values(by="eo:cloud_cover", inplace=True)
            points_gdf["stac_item"] = points_gdf.apply(
                _least_cloudy_item_covering_point,
                sorted_stac_gdf=stac_gdf,
                axis=1,
            )

        return points_gdf


def _get_trimmed_stac_shapes_gdf(item_collection):
    """
    To prevent the edge case where a point sits inside the STAC geometry
    but outwith the STAC proj:bbox shape (resulting in a dud xarray to be
    returned later), trim the STAC shapes to within the proj:bbox borders.

    Parameters
    ----------
    item_collection : pystac.ItemCollection

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame where each row is an Item and columns include
        cloud cover percentage and item shape trimmed to within proj:bbox.
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


def _least_cloudy_item_covering_point(row, sorted_stac_gdf):
    """
    Takes in a sorted dataframe of stac items and returns the
    least cloudy item that covers the current row. For use in
    `fetch_stac_items`.

    TODO: Add cloud_cover column back
    """
    items_covering_point = stac_gdf[stac_gdf.covers(row.geometry)]
    if len(items_covering_point) == 0:
        return None
    else:
        return items_covering_point["stac_item"].tolist()


def get_stac_api(api_name):
    """Get a STAC API client for a given API name."""

    items_covering_point = sorted_stac_gdf[sorted_stac_gdf.covers(row.geometry)]
    if len(items_covering_point) == 0:
        return None
    else:
        least_cloudy_item = items_covering_point.iloc[0]["stac_item"]
        return least_cloudy_item  # , least_cloudy_item.properties["eo:cloud_cover"]


def _items_covering_point(row, stac_gdf):
    """Takes in a sorted dataframe of stac items and returns all
    stac items that cover the current row as a list. For use in
    `fetch_stac_items`

    """
    items_covering_point = stac_gdf[stac_gdf.covers(row.geometry)]
    if len(items_covering_point) == 0:
        return None
    else:
        return items_covering_point["stac_item"].tolist()


def get_stac_api(api_name):
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


class CustomDataset(Dataset):
    def __init__(
        self,
        points,
        items,
        buffer,
        bands,
        resolution,
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

        Returns
        -------
        None
        """
        self.points = points
        self.items = items
        self.buffer = buffer
        self.bands = bands
        self.resolution = resolution

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
        stac_item = self.items[idx]

        if stac_item is None:
            return None
        else:
            crs = stac_item.properties["proj:epsg"]
            proj_latlon_to_stac = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
            x_utm, y_utm = proj_latlon_to_stac.transform(lon, lat)
            x_min, x_max = x_utm - self.buffer, x_utm + self.buffer
            y_min, y_max = y_utm - self.buffer, y_utm + self.buffer

            xarray = stackstac.stack(
                stac_item,
                assets=self.bands,
                resolution=self.resolution,
                rescale=False,
                dtype=np.uint8,
                bounds=[x_min, y_min, x_max, y_max],
                fill_value=0,
            )

            if isinstance(stac_item, list):
                out_image = xarray.median(dim="time")
            else:
                out_image = xarray.squeeze()

            out_image = torch.from_numpy(out_image.values).float()

            return out_image
