# DSEM The/Nudge Ultrapoor Project

This repository holds the Data Science modelling work for the Ultrapoor project.

## Setup

### Environment

1. Run `make setup-env`

### Data

1. Run `make data-directories` to create data folder structure

2. Download the geometries and secc [SHRUG data](https://www.devdatalab.org/shrug_download/shrug_select) and place them into the root data folder as below:

    ```console
    📦 data
    ┗ 📂 00_raw
      ┗ 📂 SHRUG
        ┣ 📂 geometries_shrug-v1.5.samosa-open-polygons-shp
        ┣ 📂 shrug-v1.5.samosa-keys-csv
        ┗ 📂 shrug-v1.5.samosa-secc-csv
    ```

    > Note: The Keys folder is inside the SECC folder when downloaded from SHRUG but we separate them here. Keys are included with every dataset download from SHRUG.

### Processed Data

1. If never done before, run

    ```console
    make shrug-keys-with-shapes
    ```

2. If MOSAIKS data has never been acquired, run

    ```console
    make mosaiks-request-points
    ```

    to make the grid of points and request features for these points on the [MOSAIKS website](https://siml.berkeley.edu/portal/file_query/) or through the Featurization notebook.

### DEPRECATED:
3. Download and extract the resulting data and place it inside the folder below as a `.csv` file:

    ```console
    📦 data
    ┗ 📂 00_raw
      ┣ 📂 SHRUG
      ┃ ┗ ...
      ┗ 📂 MOSAIKS
        ┗ 📜 [filename].csv
    ```

    On EC2, you can use `cURL` with the pre-loaded `login_cookie.txt` as a cookie to login to the MOSAIKS website and download the data:

    ```console
    curl -b .mosaiks/login_cookie.txt -o ds_nudge_up/data/00_raw/MOSAIKS/[filename].zip [File download URL]
    ```

    Then extract the data with `unzip`:

    ```console
    unzip ds_nudge_up/data/00_raw/MOSAIKS/[filename].zip -d ds_nudge_up/data/00_raw/MOSAIKS/[filename]
    ```

## Running the modelling

Currently the POC modelling is done through a Jupyter notebook in the `src/` folder. Run this notebook.
