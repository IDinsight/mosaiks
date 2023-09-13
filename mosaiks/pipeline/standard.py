import logging
from pathlib import Path
from typing import List

import geopandas as gpd
import pandas as pd
import torch.nn

import mosaiks.utils as utl
from mosaiks.featurize import create_features_from_image_array
from mosaiks.fetch import create_data_loader, fetch_image_refs


def run_pipeline(
    points_gdf: gpd.GeoDataFrame,
    model: torch.nn.Module,
    satellite_name: str,
    image_resolution: int,
    image_dtype: str,
    image_bands: list,
    image_width: int,
    min_image_edge: int,
    datetime: str or List[str] or callable,
    image_composite_method: str,
    stac_api_name: str,
    num_features: int,
    device: str,
    col_names: list,
    output_filepath: Path = None,
    return_df: bool = True,
) -> pd.DataFrame:  # or None
    """
    For a given DataFrame of coordinate points, this function runs the necessary
    functions and optionally saves resulting mosaiks features to file.

    Parameters
    -----------
    points_gdf : GeoDataFrame of points to be featurized.
    model: PyTorch model to be used for featurization.
    satellite_name: name of the satellite to use. Options are "landsat-8-c2-l2" or "sentinel-2-l2a".
    image_resolution: resolution of the satellite images in meters. Set depending on satellite.
    image_dtype: data type of the satellite images. Suggested "int16". All options - "int16", "int32", and "float"
    image_bands: list of bands to use for the satellite images. Suggested ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]. For options, read the satellite docs
    image_width: Desired width of the image to be fetched (in meters). Suggested 3000m for landsat.
    min_image_edge: minimum image edge in meters. Suggested 1000.
    datetime : date/times for fetching satellite images. See STAC API docs for `pystac.Client.search`'s `datetime` parameter for more details
    image_composite_method: how to composite multiple images for same GPS location. Options are "least_cloudy" (pick least cloudy image) or "all" (get all images and average across them). Suggested "least_cloudy" for speed.
    stac_api_name: which STAC API to use. Options are "planetary-compute" or "earth-search". Suggested "planetary-compute".
    num_features: number of mosaiks features to generate. Suggested 1000-4000.
    device: compute device for mosaiks model. Options are "cpu" or "cuda".
    col_names: column names for the mosaiks features.
    output_filepath : Path to file where features will be saved. Must have .csv or .parquet or .parquet.gzip format. Default is None.
    return_df : Whether to return the features as a DataFrame. Default is True.

    Returns
    --------
    None or DataFrame depending on the "return_df" parameter.
    """

    points_gdf_with_stac = fetch_image_refs(
        points_gdf=points_gdf,
        satellite_name=satellite_name,
        datetime=datetime,
        image_composite_method=image_composite_method,
        stac_api_name=stac_api_name,
    )

    data_loader = create_data_loader(
        points_gdf_with_stac=points_gdf_with_stac,
        image_bands=image_bands,
        image_resolution=image_resolution,
        image_dtype=image_dtype,
        image_width=image_width,
        image_composite_method=image_composite_method,
    )

    X_features = create_features_from_image_array(
        dataloader=data_loader,
        n_features=num_features,
        model=model,
        device=device,
        min_image_edge=min_image_edge,
    )

    df = utl.make_result_df(
        features=X_features,
        context_gdf=points_gdf_with_stac,
        mosaiks_col_names=col_names,
    )

    if output_filepath is not None:
        try:
            utl.save_dataframe(df=df, file_path=output_filepath)
        except Exception as e:
            logging.error(f"Failed to save dataframe to {output_filepath}.")
            logging.error(e)

    if return_df:
        return df
