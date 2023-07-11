import logging
import math
from typing import List

import geopandas as gpd
import numpy as np
import pyproj
import stackstac
import torch
from pystac.item import Item
from torch.utils.data import DataLoader, Dataset

from mosaiks.fetch.stacs import fetch_stac_item_from_id

__all__ = [
    "create_data_loader",
    "fetch_image_crop_from_stac_id",
    "fetch_image_crop",
]


def fetch_image_crop(
    lon: float,
    lat: float,
    stac_item: Item,  # or list[Items]
    buffer_distance: int,
    bands: List[str],
    resolution: int,
    dtype: str = "int16",
    mosaic_composite: str = "least_cloudy",
    normalise: bool = True,
) -> np.array:
    """
    Fetches a crop of satellite imagery referenced by the given STAC item(s),
    centered around the given point and with the given buffer_distance.

    If multiple STAC items are given, the median composite of the images is returned.

    Parameters
    ----------
    lon : Longitude of the centerpoint to fetch imagery for
    lat : Latitude of the centerpoint to fetch imagery for
    stac_item : STAC Item to fetch imagery for. Can be a list of STAC Items.
    buffer_distance : buffer_distance in meters around the centerpoint to fetch imagery for
    bands : List of bands to fetch
    resolution : Resolution of the image to fetch
    dtype : Data type of the image to fetch. Defaults to "int16".
        NOTE - np.uint8 results in loss of signal in the features
        and np.uint16 is not supported by PyTorch.
    mosaic_composite : The type of composite to use for multiple images. If
        "least_cloudy"", take the least cloudy non-0 image. If "all", take a median
        composite of all images. Defaults to "least_cloudy".
    normalise : Whether to normalise the image. Defaults to True.

    Returns
    -------
    image : numpy array of shape (C, H, W)
    """
    if stac_item is None or all(x is None for x in stac_item):
        size = (
            len(bands),
            math.ceil(2 * buffer_distance / resolution + 1),
            math.ceil(2 * buffer_distance / resolution + 1),
        )
        return np.ones(size) * np.nan

    # Stac item must always be a list
    assert isinstance(stac_item, list)

    # calculate crop bounds
    # use the projection of the first non-None stac item
    (idx,) = np.nonzero([x is not None for x in stac_item])
    crs = stac_item[idx[0]].properties["proj:epsg"]

    proj_latlon_to_stac = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
    x_utm, y_utm = proj_latlon_to_stac.transform(lon, lat)
    x_min, x_max = x_utm - buffer_distance, x_utm + buffer_distance
    y_min, y_max = y_utm - buffer_distance, y_utm + buffer_distance

    # get image(s) as xarray
    xarray = stackstac.stack(
        stac_item,
        assets=bands,
        resolution=resolution,
        rescale=True,
        dtype=dtype,
        bounds=[x_min, y_min, x_max, y_max],
        fill_value=0,
    )

    # remove the time dimension
    if mosaic_composite == "all":
        # for a composite over all images, take median pixel over time
        image = xarray.median(dim="time").values
    elif mosaic_composite == "least_cloudy":
        # for least cloudy, take the first non zero image
        for i in range(len(stac_item)):
            image = xarray[i].values
            if ~np.all(image == 0.0):
                break

    if normalise:
        image = _minmax_normalize_image(image)

    return image


def _minmax_normalize_image(image: np.array) -> np.array:
    img_min, img_max = image.min(), image.max()
    return (image - img_min) / (img_max - img_min)


def create_data_loader(
    points_gdf_with_stac: gpd.GeoDataFrame,
    image_bands: List[str],
    image_resolution: int,
    image_dtype: str,
    buffer_distance: int,
    batch_size: int,
    mosaic_composite: str,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader which returns cropped images based on the given
    coordinate points and associated STAC image references.

    Parameters
    ----------
    points_gdf_with_stac : A GeoDataFrame with the points we want to fetch imagery for
        alongside the STAC item references to pictures for each point
    image_bands : The bands to use for the image crops
    image_resolution : The resolution to use for the image crops
    image_dtype : The data type to use for the image crops
    buffer_distance : The buffer distance in meters to use for the image crops
    batch_size : The batch size to use for the DataLoader
    mosaic_composite : The type of composite used for multiple images

    Returns
    -------
    data_loader : A PyTorch DataLoader which returns cropped images as tensors
    """

    stac_item_list = points_gdf_with_stac.stac_item.tolist()
    points_list = points_gdf_with_stac[["Lon", "Lat"]].to_numpy()
    dataset = CustomDataset(
        points_list,
        stac_item_list,
        buffer_distance=buffer_distance,
        bands=image_bands,
        resolution=image_resolution,
        dtype=image_dtype,
        mosaic_composite=mosaic_composite,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
        pin_memory=False,
    )

    return data_loader


class CustomDataset(Dataset):
    def __init__(
        self,
        points: np.array,
        items: List[Item],
        buffer_distance: int,
        bands: List[str],
        resolution: int,
        dtype: str = "int16",
        mosaic_composite: str = "least_cloudy",
    ) -> None:
        """
        Parameters
        ----------
        points : Array of points to sample from
        items : List of STAC items to sample from
        buffer_distance : Buffer distance in meters around each point to sample from
        bands : List of bands to sample
        resolution : Resolution of the image to sample
        dtype : Data type of the image to sample. Defaults to "int16".
            NOTE - np.uint8 results in loss of signal in the features
            and np.uint16 is not supported by PyTorch.
        mosaic_composite: The type of composite to used for multiple images
        """

        self.points = points
        self.items = items
        self.buffer = buffer_distance
        self.bands = bands
        self.resolution = resolution
        self.dtype = dtype
        self.mosaic_composite = mosaic_composite

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

        try:
            image = fetch_image_crop(
                lon=lon,
                lat=lat,
                stac_item=stac_item,
                buffer_distance=self.buffer,
                bands=self.bands,
                resolution=self.resolution,
                dtype=self.dtype,
                mosaic_composite=self.mosaic_composite,
                normalise=True,
            )
            torch_image = torch.from_numpy(image).float()

            # if all 0s, replace with NaNs
            if torch.all(torch_image == 0):
                torch_image = np.ones_like(torch_image) * np.nan

            return torch_image
        except Exception as e:
            print(f"Skipping {idx}: {e}")
            return None


# for debugging


def fetch_image_crop_from_stac_id(
    stac_id: str,
    lon: float,
    lat: float,
    buffer_distance: int,
    bands: List[str],
    resolution: int,
    dtype: str = "int16",
    mosaic_composite: str = "least_cloudy",
    normalise: bool = True,
    stac_api_name: str = "planetary-compute",
) -> np.array:
    """
    Note: This function is necessary since STAC Items cannot be directly saved to file
    alongside the features in `.parquet` format, and so only their ID can be saved.
    This function uses the ID to first fetch the correct STAC Items, runs these
    through fetch_image_crop(), and optionally displays the image.

    Takes a stac_id (or list of stac_ids), lat, lon, and satellite
    parameters and returns and displays a cropped image. This image is fetched using the
    same process as when fetching for featurization.

    If multiple STAC items are given, the median composite of the images is returned.

    Parameters
    ----------
    stac_id : The STAC ID of the image to fetch
    lon : Longitude of the centerpoint to fetch imagery for
    lat : Latitude of the centerpoint to fetch imagery for
    buffer_distance : Buffer in meters around the centerpoint for fetching imagery
    bands : The satellite image bands to fetch
    resolution : The resolution of the image to fetch
    dtype : The data type of the image to fetch. Defaults to "int16".
    mosaic_composite : The type of composite to use for multiple images.
    normalise : Whether to normalise the image. Defaults to True.
    stac_api_name : The name of the STAC API to use. Defaults to "planetary-compute".

    Returns
    -------
    image_crop : A numpy array of the cropped image
    """

    if not isinstance(stac_id, list):
        stac_id = [stac_id]

    stac_items = fetch_stac_item_from_id(stac_id, stac_api_name)
    if len(stac_items) == 0:
        logging.warn("No STAC items found.")
        return None

    else:
        image_crop = fetch_image_crop(
            lon=lon,
            lat=lat,
            stac_item=stac_items,
            buffer_distance=buffer_distance,
            bands=bands,
            resolution=resolution,
            dtype=dtype,
            mosaic_composite=mosaic_composite,
            normalise=normalise,
        )

        return image_crop
