import importlib.util
import re

import requests


def main():
    # Get the latest version number from PyPI
    url = "https://pypi.python.org/pypi/mosaiks/json"
    response = requests.get(url)
    latest_version = response.json()["info"]["version"]

    # Get the version number from file
    init_file_path = "./mosaiks/__init__.py"
    version_pattern = r"^__version__ = ['\"]([^'\"]*)['\"]"

    installed_version = None

    with open(init_file_path, "r") as file:
        for line in file:
            # Search each line for the version pattern
            match = re.search(version_pattern, line, re.M)
            if match:
                # If a match is found, extract the version number
                installed_version = match.group(1)
                break
    # Compare the two version numbers
    if latest_version != installed_version:
        print(
            "You have updated the version of mosaiks to ({})".format(installed_version)
        )
    else:
        print(
            "You have not updated the version of mosaiks from ({})".format(
                installed_version
            )
        )
        raise AssertionError


if __name__ == "__main__":
    main()
