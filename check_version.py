import importlib.util

import requests


def main():
    # Get the latest version number from PyPI
    url = "https://pypi.python.org/pypi/mosaiks/json"
    response = requests.get(url)
    latest_version = response.json()["info"]["version"]

    # Get the installed version number
    module = importlib.util.spec_from_file_location("mosaiks", "./mosaiks/__init__.py")
    mosaiks = importlib.util.module_from_spec(module)
    module.loader.exec_module(mosaiks)
    installed_version = mosaiks.__version__

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
