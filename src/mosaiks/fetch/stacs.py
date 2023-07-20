import copy
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import shapely
from pystac.item import Item
from pystac.item_collection import ItemCollection

__all__ = ["fetch_image_refs", "fetch_stac_item_from_id"]


def fetch_image_refs(
    points_gdf: gpd.GeoDataFrame,
    satellite_name: str,
    seasonal: bool,
    year: int,
    datetime: str,
    image_composite_method: str,
    stac_api_name: str,
) -> gpd.GeoDataFrame:
    """
    Find a STAC item for points in the `points_gdf` GeoDataFrame. Returns a
    GeoDataFrame with STAC items as a new column.

    Parameters
    ----------
    points_gdf : A GeoDataFrame with a column named "geometry" containing shapely
        Point objects.
    satellite_name : Name of MPC-hosted satellite.
    seasonal: Whether to fetch seasonal STAC items.
    year: The year to fetch seasonal STAC items for.
    datetime : date/times for fetching satellite images. See STAC API docs for `pystac.Client.search`'s `datetime` parameter for more details
    image_composite_method : how to composite multiple images for same GPS location.
    stac_api_name: The name of the STAC API to use.

    Returns
    -------
    points_gdf: A GeoDataFrame with a column named "stac_item" containing STAC
        items.
    """
    stac_api = get_stac_api(stac_api_name)

    if seasonal:
        points_gdf_with_stac = fetch_seasonal_stac_items(
            points_gdf=points_gdf,
            satellite_name=satellite_name,
            year=year,
            stac_api=stac_api,
            image_composite_method=image_composite_method,
        )
    else:
        points_gdf_with_stac = fetch_stac_items(
            points_gdf=points_gdf,
            satellite_name=satellite_name,
            datetime=datetime,
            stac_api=stac_api,
            image_composite_method=image_composite_method,
        )

    return points_gdf_with_stac


def fetch_seasonal_stac_items(
    points_gdf: gpd.GeoDataFrame,
    satellite_name: str,
    year: int,
    stac_api: pystac_client.Client,
    image_composite_method: str,
) -> gpd.GeoDataFrame:
    """
    Takes a year as input and creates date ranges for the four seasons, runs these
    through fetch_stac_items, and concatenates the results. Can be used where-ever
    `fetch_stac_items` is used.

    Note: Winter is the first month and includes December from the previous year.

    Months of the seasons taken from [here](https://delhitourism.gov.in/delhitourism/aboutus/seasons_of_delhi.jsp) for now.
    """

    # Make a copy of the points_gdf so we don't modify the original
    points_gdf_copy = copy.deepcopy(points_gdf)
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
            points_gdf=points_gdf_copy,
            satellite_name=satellite_name,
            datetime=[search_start, search_end],
            stac_api=stac_api,
            image_composite_method=image_composite_method,
        )
        season_points_gdf["season"] = season

        # Save copy of the seasonal gdf to list, since pointer is overwritten in the
        # next iteration
        seasonal_gdf_list.append(season_points_gdf.copy())

    combined_gdf = pd.concat(seasonal_gdf_list, axis="index")
    combined_gdf.index.name = "point_id"

    return combined_gdf


def fetch_stac_items(
    points_gdf: gpd.GeoDataFrame,
    satellite_name: str,
    datetime: str or list[str] or callable,
    stac_api: pystac_client.Client,
    image_composite_method: str,
) -> gpd.GeoDataFrame:
    """
    Find the STAC item(s) that overlap each point in the `points_gdf` GeoDataFrame.

    Parameters
    ----------
    points_gdf : A GeoDataFrame
    satellite_name : Name of MPC-hosted satellite
    datetime : date/times for fetching satellite images. See STAC API docs for `pystac.Client.search`'s `datetime` parameter for more details
    stac_api: The pystac_client.Client object to use for searching for image refs. Output of `get_stac_api`
    image_composite_method : Whether to store "all" images found or just the "least_cloudy"

    Returns
    -------
    points_gdf: A new geopandas.GeoDataFrame with a `stac_item` column containing the
        STAC item that covers each point.
    """
    points_gdf = points_gdf.copy()

    # Check for NaNs in lat lons
    nan_mask = points_gdf["Lat"].isna() + points_gdf["Lon"].isna()
    points_gdf["stac_item"] = None

    if not nan_mask.all():
        points_gdf_not_nan = points_gdf.loc[~nan_mask].copy()
        points_union = shapely.geometry.mapping(points_gdf_not_nan.unary_union)

        search_results = stac_api.search(
            collections=[satellite_name],
            intersects=points_union,
            datetime=datetime,
            query={"eo:cloud_cover": {"lt": 10}},
            limit=500,  # this limit seems arbitrary
        )
        item_collection = search_results.item_collection()

        if len(item_collection) == 0:
            return points_gdf

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

            points_gdf.loc[~nan_mask] = _add_overlapping_stac_items(
                gdf=points_gdf_not_nan,
                stac_gdf=stac_gdf,
                image_composite_method=image_composite_method,
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
    for i, item in enumerate(item_collection):

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


def _add_overlapping_stac_items(
    gdf: gpd.GeoDataFrame,
    stac_gdf: gpd.GeoDataFrame,
    image_composite_method: str,
) -> List[Item]:
    """
    Takes in geodataframe of points and a sorted dataframe of stac items and returns
    the point geodataframe with an additional column which lists the items that covers
    each point. For use in `fetch_stac_items`.
    """

    gdf = gdf.copy()

    if image_composite_method == "least_cloudy":
        stac_gdf = stac_gdf.sort_values(by="eo:cloud_cover")

    for index, row in gdf.iterrows():

        items_covering_point = stac_gdf[stac_gdf.covers(row.geometry)]
        if len(items_covering_point) == 0:
            gdf.at[index, "stac_item"] = None
        else:
            if image_composite_method in ["all", "least_cloudy"]:
                all_items = items_covering_point["stac_item"].tolist()
                gdf.at[index, "stac_item"] = all_items
            else:
                raise ValueError(
                    f"image_composite_method must be 'least_cloudy' or 'all', not {image_composite_method}"
                )

    return gdf


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


# for debugging


def fetch_stac_item_from_id(
    ids: list[str], stac_api_name: str = "planetary-compute"
) -> list[Item]:
    """
    For debugging.

    Fetches STAC items from a list of ids.

    Parameters
    ----------
    ids: List of STAC item ids.
    stac_api_name: Name of STAC API to use. Has to be supported by `get_stac_api`.

    Returns
    -------
    List of STAC items.
    """
    stac_api = get_stac_api(stac_api_name)
    nan_mask = np.array([x is None for x in ids])
    if np.all(nan_mask):
        return [None] * len(ids)
    elif np.any(~nan_mask):
        search_results = [None] * len(ids)
        for i, id in enumerate(ids):
            if id is not None:
                stac_item_list = list(stac_api.search(ids=id).items())
                if len(stac_item_list) > 0:
                    search_results[i] = stac_item_list[0]

    return search_results
