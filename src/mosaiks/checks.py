"""Internal functions for checking inputs to pipeline functions."""

from datetime import datetime

valid_satellite_names = ["sentinel-2-l2a", "landsat-8-c2-l2"]
valid_stac_api_names = ["planetary-compute", "earth"]


def check_latitudes_and_longitudes(latitudes: list, longitudes: list) -> None:
    """Check that latitudes and longitudes are of equal length."""
    if len(latitudes) != len(longitudes):
        raise ValueError(
            f"Length of latitudes ({len(latitudes)}) must be equal to length of\
            longitudes ({len(longitudes)})"
        )


def check_satellite_name(satellite_name: str) -> None:
    """Check that satellite name is valid."""
    if satellite_name not in valid_satellite_names:
        raise ValueError(
            f"Satellite name must be one of {list(valid_satellite_names)},\
            not {satellite_name}"
        )


def check_stac_api_name(stac_api: str) -> None:
    """Check that stac api name is valid."""
    if stac_api not in valid_stac_api_names:
        raise ValueError(
            f"STAC api must be one of {list(valid_stac_api_names)},\
            not {stac_api}"
        )
