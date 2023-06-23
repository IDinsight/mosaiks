import copy
from typing import List

import geopandas as gpd
import pandas as pd
import planetary_computer
import pystac_client
import shapely
from pystac.item import Item
from pystac.item_collection import ItemCollection

__all__ = ["fetch_image_refs", "fetch_stac_item_from_id"]


def fetch_image_refs(
    points_gdf: gpd.GeoDataFrame, satellite_search_params: dict
) -> gpd.GeoDataFrame:
    """
    Find a STAC item for points in the `points_gdf` GeoDataFrame. Returns a
    GeoDataFrame with STAC items as a new column.

    Parameters
    ----------
    points_gdf : A GeoDataFrame with a column named "geometry" containing shapely
        Point objects.
    satellite_search_params : A dictionary containing the parameters for the STAC
        search.

    Returns
    -------
    points_gdf: A GeoDataFrame with a column named "stac_item" containing STAC
        items.
    """
    if satellite_search_params["seasonal"]:
        points_gdf_with_stac = fetch_seasonal_stac_items(
            points_gdf=points_gdf,
            satellite_name=satellite_search_params["satellite_name"],
            year=satellite_search_params["year"],
            stac_output=satellite_search_params["stac_output"],
            stac_api=satellite_search_params["stac_api"],
        )
    else:
        points_gdf_with_stac = fetch_stac_items(
            points_gdf=points_gdf,
            satellite_name=satellite_search_params["satellite_name"],
            search_start=satellite_search_params["search_start"],
            search_end=satellite_search_params["search_end"],
            stac_api=satellite_search_params["stac_api"],
            stac_output=satellite_search_params["stac_output"],
        )

    return points_gdf_with_stac


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
            search_start=search_start,
            search_end=search_end,
            stac_api=stac_api,
            stac_output=stac_output,
        )

        season_points_gdf["season"] = season
        # Save copy of the seasonal gdf to list, since pointer is overwritten in the
        #  next iteration
        seasonal_gdf_list.append(season_points_gdf.copy())

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

    points_union = shapely.geometry.mapping(points_gdf.unary_union)

    search_results = stac_api.search(
        collections=[satellite_name],
        intersects=points_union,
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

        points_gdf["stac_item"] = _get_overlapping_stac_items(
            gdf=points_gdf, stac_gdf=stac_gdf, stac_output=stac_output
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


def _get_overlapping_stac_items(
    gdf: gpd.GeoDataFrame,
    stac_gdf: gpd.GeoDataFrame,
    stac_output: str = "least_cloudy",
) -> Item:  # or List[Item]
    """
    Takes in a sorted dataframe of stac items and returns the item(s) that covers each
    row. For use in `fetch_stac_items`.
    """

    if stac_output == "least_cloudy":
        stac_gdf = stac_gdf.sort_values(by="eo:cloud_cover")

    col_value_list = []
    for index, row in gdf.iterrows():

        items_covering_point = stac_gdf[stac_gdf.covers(row.geometry)]
        if len(items_covering_point) == 0:
            col_value_list.append(None)
        else:
            if stac_output == "all":
                all_items = items_covering_point["stac_item"].tolist()
                col_value_list.append(all_items)
            elif stac_output == "least_cloudy":
                least_cloudy_item = items_covering_point.iloc[0]["stac_item"]
                col_value_list.append(least_cloudy_item)
            else:
                raise ValueError(
                    f"stac_output must be 'least_cloudy' or 'all', not {stac_output}"
                )

    return col_value_list


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
    search_results = stac_api.search(ids=ids)
    return list(search_results.items())
