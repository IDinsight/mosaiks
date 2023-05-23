from typing import List

import geopandas as gpd
import pandas as pd
import planetary_computer
import pystac_client
import shapely
from pystac.item import Item
from pystac.item_collection import ItemCollection

__all__ = ["fetch_image_refs", "fetch_stac_item_from_id"]


def fetch_stac_item_from_id(
    id: str, 
    stac_api_name: str = "planetary-compute"
):
    """For debugging."""
    
    stac_api = get_stac_api(stac_api_name)
    search_results = stac_api.search(ids=[id])
    return next(search_results.items())


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
    CAUTION - THIS IS BROKEN, FIX.

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
