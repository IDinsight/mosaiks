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
    seasonal: bool,
    year: int,
    search_start: str,
    search_end: str,
    image_composite_method: str,
    stac_api_name: str,
    num_features: int,
    device: str,
    col_names: list,
    output_folderpath: str = None,
    save_filename: str = "features.csv",
    return_df: bool = True,
) -> pd.DataFrame:  # or None
    """
    For a given DataFrame of coordinate points, this function runs the necessary
    functions and optionally saves resulting mosaiks features to file.

    Parameters
    -----------
    points_gdf : GeoDataFrame of points to be featurized.
    model: PyTorch model to be used for featurization.
    satellite_name : Name of satellite to be used for featurization.
    image_resolution : Resolution of satellite images to be used for featurization.
    image_dtype : Data type of satellite images to be used for featurization.
    image_bands : List of satellite image bands to be used for featurization.
    image_width : Desired width of the image to be fetched (in meters).
    min_image_edge : Minimum image edge size.
    seasonal : Whether to use seasonal satellite images for featurization.
    year : Year to be used for featurization.
    search_start : Start date for satellite image search.
    search_end : End date for satellite image search.
    image_composite_method : Mosaic composite to be used for featurization.
    stac_api_name : Name of STAC API to be used for satellite image search.
    num_features : number of mosaiks features.
    device : Device to be used for featurization.
    col_names : List of column names to be used for saving the features. Default is None, in which case the column names will be "mosaiks_0", "mosaiks_1", etc.
    output_folderpath : Path to folder where features will be saved. Default is None.
    save_filename : Name of file where features will be saved. Default is "features.csv".
    return_df : Whether to return the features as a DataFrame. Default is True.

    Returns
    --------
    None or DataFrame
    """

    points_gdf_with_stac = fetch_image_refs(
        points_gdf=points_gdf,
        satellite_name=satellite_name,
        seasonal=seasonal,
        year=year,
        search_start=search_start,
        search_end=search_end,
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

    if output_folderpath is not None:
        utl.save_dataframe(df=df, file_path=output_folderpath / save_filename)

    if return_df:
        return df
