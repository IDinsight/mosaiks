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
    buffer: int,
    bands: List[str],
    resolution: int,
    dtype: str = "int16",
    normalise: bool = True,
) -> np.array:
    """
    Fetches a crop of satellite imagery referenced by the given STAC item(s),
    centered around the given point and with the given buffer.

    If multiple STAC items are given, the median composite of the images is returned.

    Parameters
    ----------
    lon : Longitude of the centerpoint to fetch imagery for
    lat : Latitude of the centerpoint to fetch imagery for
    stac_item : STAC Item to fetch imagery for. Can be a list of STAC Items.
    buffer : Buffer in meters around the centerpoint to fetch imagery for
    bands : List of bands to fetch
    resolution : Resolution of the image to fetch
    dtype : Data type of the image to fetch. Defaults to "int16".
        NOTE - np.uint8 results in loss of signal in the features
        and np.uint16 is not supported by PyTorch.
    normalise : Whether to normalise the image. Defaults to True.

    Returns
    -------
    image : numpy array of shape (C, H, W)
    """
    if stac_item is None:
        size = (
            len(bands),
            math.ceil(2 * buffer / resolution + 1),
            math.ceil(2 * buffer / resolution + 1),
        )
        return np.ones(size) * np.nan

    # calculate crop bounds
    if isinstance(stac_item, list):
        # if multiple items, use the projection of the first one
        crs = stac_item[0].properties["proj:epsg"]
    else:
        crs = stac_item.properties["proj:epsg"]
    proj_latlon_to_stac = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
    x_utm, y_utm = proj_latlon_to_stac.transform(lon, lat)
    x_min, x_max = x_utm - buffer, x_utm + buffer
    y_min, y_max = y_utm - buffer, y_utm + buffer

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
    if isinstance(stac_item, list):
        # if there are multiple image, make composite over time
        image = xarray.median(dim="time")
    else:
        # if there is only one image, just remove the redundant time dimension
        image = xarray.squeeze()

    # turn xarray to np.array
    image = image.values

    if normalise:
        image = _minmax_normalize_image(image)

    return image


def _minmax_normalize_image(image: np.array) -> np.array:
    img_min, img_max = image.min(), image.max()
    return (image - img_min) / (img_max - img_min)


def create_data_loader(
    points_gdf_with_stac: gpd.GeoDataFrame, satellite_params: dict, batch_size: int
) -> DataLoader:
    """
    Creates a PyTorch DataLoader which returns cropped images based on the given
    coordinate points and associated STAC image references.

    Parameters
    ----------
    points_gdf_with_stac : A GeoDataFrame with the points we want to fetch imagery for
        alongside the STAC item references to pictures for each point
    satellite_params : A dictionary of parameters for the satellite imagery to fetch
    batch_size : The batch size to use for the DataLoader

    Returns
    -------
    data_loader : A PyTorch DataLoader which returns cropped images as tensors
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

        try:
            image = fetch_image_crop(
                lon=lon,
                lat=lat,
                stac_item=stac_item,
                buffer=self.buffer,
                bands=self.bands,
                resolution=self.resolution,
                dtype=self.dtype,
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
    satellite_config: dict,
    normalise=True,
    stac_api_name: str = "planetary-compute",
    plot: bool = False,
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
    satellite_config : A dictionary of parameters for the satellite imagery to fetch
    normalise : Whether to normalise the image. Defaults to True.
    stac_api_name : The name of the STAC API to use. Defaults to "planetary-compute".
    plot : Whether to plot the image. Defaults to False.

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
            buffer=satellite_config["buffer_distance"],
            bands=satellite_config["bands"],
            resolution=satellite_config["resolution"],
            dtype=satellite_config["dtype"],
            normalise=normalise,
        )

        if plot:
            display_image(image_crop)

        return image_crop
