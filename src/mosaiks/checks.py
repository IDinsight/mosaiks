"""Internal functions for checking inputs to pipeline functions."""

import datetime

valid_satellite_names = ["sentinel-2-l2a", "landsat-8-l1"]
valid_stac_api_names = ["planetary-compute", "earth"]


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


def check_search_dates(search_start_date: str, search_end_date: str) -> None:
    """Check that search dates are valid."""
    # Check date formats: datetime.strptime will raise a ValueError if the format is
    # wrong
    datetime.strptime(search_start_date, "%Y-%m-%d")
    datetime.strptime(search_end_date, "%Y-%m-%d")

    # Check that search start date is before search end date
    if datetime.strptime(search_start_date, "%Y-%m-%d") > datetime.strptime(
        search_end_date, "%Y-%m-%d"
    ):
        raise ValueError(
            f"Search start date {search_start_date} must be before search end date\
            {search_end_date}"
        )
