# MOSAIKS Satellite Imagery Featuri ation

MOSAIKS is a Python package that performs parallelized encoding of satellite imagery into easy-to-use features using the MOSAIKS algorithm and Dask for parallel processing. This package enables users to generate feature vectors based on satellite images by providing a list of latitudes and longitudes and Microsoft's Planetary Computer API key. It supports various satellites, image sizes, time periods, and parallelization options.

We implement the MOSAIKS algorithm based on work by [Rolf et al., 2021](https://www.nature.com/articles/s41467-021-24638-z). The authors of this paper implement MOSAIKS using Google Maps images (mostly 2018), and also provide some pre-computed features based on Planet imagery from 2019 [here](https://www.mosaiks.org/).

This package extends the functionality of the original MOSAIKS implementation in the following ways:

- Flexibility in choice of satellite (tested for Landsat-8 and Sentinel-2)
- Ability to select the timeframe from which to pull imagery
- Flexibility in choosing the size of the images centered on points of interest, image bands used, etc.
- Flexible upload and download for data
- Parallel processing with Dask to speed up fetching images and creating features.
- Once installed, the package can be run on any machine (with the API key).

The package has been tested via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) on Landsat-8 or Sentinel-2 imagery. **Please note this package has only been tested on Python 3.10 and 3.11**. Using other versions of Python are expected to raise errors due to dependency conflicts.

For more detailed information on this package and how to use it, please see [this blog post](link to technical blog post). For information on preview and potential use cases for this package, please see [this blog post](link to non-technical blog post)

## Quick Start
This section highlights a demo to help you get features ASAP.

### Step 1: Set-up

Ensure you have all requirements set up:

1. Install Python 3.10 or 3.11.
2. Install the MOSAIKS package -

    ```sh
    pip install git+https://github.com/IDinsight/mosaiks@main
    ```

3. Acquire the Planetary Computer API key from [Microsoft Planetary Computer (MPC)](https://planetarycomputer.microsoft.com/). We provide detailed instructions for getting an API key in the FAQs section of this README.

    In your terminal, run the following and fill the API key prompt -

    ```sh
    planetarycomputer configure
    ```

### Step 2: Test run in a Notebook

The quickest way to test the package is to run it in a notebook. Open up a notebook in the relevant environment (where Step 1 was executed) and run the following (the code is present in the "README_DEMO.ipynb" notebook):

1. **Import dependencies**

    ```python
    import os

    # Resolves a conflict in Geopandas. Improves speed.
    os.environ["USE_PYGEOS"] = "0"
    ```

2. **Import test data. In this case, we are creating random GPS coordinates**

    ```python
    # Example: Select 5 coordinates in Uttar Pradesh, India
    lats = [26.51268717, 26.55187804, 26.54949092, 26.54105597, 26.54843896]
    lons = [80.51489844, 80.54864309, 80.57813289, 80.51412136, 80.52254959]
    ```

3. **Execute a run of the `get_features` function:**

    ```python
    from mosaiks import get_features

    df_featurised = get_features(
        lats,
        lons,
        image_width=1000,
        search_start="2013-01-01",
        search_end="2013-12-31",
    )

    df_featurised
    ```

    The above code executes a default run of the get_features function which executes the featurisation in parallel using Dask.

4. **Run get_features without dask**

    It is possible that you want to implement your own parallelisation without dask. For that, you could do a non-parallelised run of the function across your own paralllelisation logic (through code or cloud):

    ```python
    df_featurised = get_features(
        lats,
        lons,
        image_width=1000,
        search_start="2013-01-01",
        search_end="2013-12-31",
        parallelize=False,
    )

    df_featurised
    ```

5. **Run Utility function to load data and save features**

    In situations where you want to load data, run featurisation, and save features on disk, quietly, you can use the `load_and_save_features`:

        ```python
    # Save test data to file to load later
    import pandas as pd

    df = pd.DataFrame({"lat": lats, "lon": lons})
    df.to_csv("test_data.csv")

    # Loading points, featurise images, and save features to file.
    from mosaiks.extras import load_and_save_features

    load_and_save_features(
        input_file_path="test_data.csv",
        lat_col="lat",
        lon_col="lon",
        path_to_save_data="test_features.csv",
        image_width=1000,
        search_start="2013-01-01",
        search_end="2013-12-31",
        context_cols_to_keep_from_input=["lat", "lon"],
    )
    ```

## Core functionality of the system

The high-level flow of our featurisation pipeline is the following:

1. The User feeds 'lat' and 'lon' lists for points they want to featurise
    - The user also adds relevant parameters to the function (see FAQs)
2. For each GPS coordinate, the function fetches [STAC](https://stacspec.org/en) references to satellite images
3. Once found, the function fetches the images
4. Function converts each image into features using the MOSAIKS algorithm
5. Lastly, the function returns a dataframe with the features, 'lat' and 'lon' columns, and any other columns if specified by the user

## Repository structure

```
├── src -- source code
│   ├── mosaiks
│   │   ├── pipeline.py -- pipeline code: takes in GPS coordinates and processing
│   │   ├── featurize/ -- code for featurisation from images
│   └   └── fetch/ -- code for fetching images
├── tests/ -- unit tests
├── project_config.cfg -- repository configuration
├── pyproject.toml -- repository install configuration
├── pytest.ini -- unit test configuration
├── requirements_test.txt -- unit test package install requirements
├── requirements_dev.txt -- dev install requirements
└── requirements.txt -- package install requirements
```

## FAQs

### - How do I get access to the Planetary Computer API key?
If you are running mosaiks locally or on a non-MPC server, then you need an access token for the satellite image database.

1. If you do not have an MPC account, go [here](https://planetarycomputer.microsoft.com/explore). You should see a “Request Access” button in the top right corner.

    It opens up a form which you should fill in. NB: Use your personal email ID, rather than an institutional one. If you already have a Microsoft account, use the email ID (non-institutional) associated with it: otherwise, you also have the additional step of creating a Microsoft account for the email ID you want to use for the MPC Hub.

    Once you submit the form, you should receive an email within a week granting you access to the hub.

2. To get the token, go [here](https://pccompute.westeurope.cloudapp.azure.com/compute/hub/token). Request and copy the new token, and save it.

3. On your local / virtual machine. Run `pip install planetary-compute` and `planetarycomputer configure` from the console, and paste in the API key you generated.

4. You only need to do this once, and should be able to access the database smoothly every time you run the pipeline after this.

5. More information available [here](https://planetarycomputer.microsoft.com/docs/reference/sas/).

### - Can you tell me about all the parameters that I can use in the `get_features` and `load_and_save_features`?

Here are all the parameters and defaults that `get_features` uses (`load_and_save_features` also accepts these):

```python
def get_features(
    latitudes: List[float],
    longitudes: List[float],
    parallelize: bool = True,
    satellite_name: str = "landsat-8-c2-l2", # or "sentinel-2-l2a"
    image_resolution: int = 30,
    image_dtype: str = "int16", # or "int32" or "float"
    image_bands: List[str] = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"], # For options, see FAQs below
    image_width: int = 3000,
    min_image_edge: int = 30,
    sort_points_by_hilbert_distance: bool = True,
    seasonal: bool = False,
    year: int = None,
    search_start: str = "2013-01-01",
    search_end: str = "2013-12-31",
    mosaic_composite: str = "least_cloudy", # or all
    stac_api: str = "planetary-compute", # or "earth-search"
    n_mosaiks_features: int = 4000,
    mosaiks_kernel_size: int = 3,
    mosaiks_batch_size: int = 10,
    model_device: str = "cpu", # or "cuda"
    dask_n_concurrent_tasks: int = 8,
    dask_chunksize: int = 500,
    dask_n_workers: int = 4,
    dask_threads_per_worker: int = 4,
    dask_worker_cores: int = 4,
    dask_worker_memory: int = 2,
    dask_pip_install: bool = False,
    mosaiks_col_names: list = None,
    setup_rasterio_env: bool = True,
) -> pd.DataFrame
```

You can also feed all of these parameters through a .yml file, read the file, and then input the parameters as **kwargs. Here is an example .yml file for the parameters that can be re-used:

```yml
parallelize: true
satellite_name: "landsat-8-c2-l2"  # or "sentinel-2-l2a"
image_resolution: 30
image_dtype: "int16"  # or "int32" or "float"
image_bands:
  - "SR_B2"
  - "SR_B3"
  - "SR_B4"
  - "SR_B5"
  - "SR_B6"
  - "SR_B7"
image_width: 3000
min_image_edge: 30
sort_points_by_hilbert_distance: true
seasonal: false
year: null
search_start: "2013-01-01"
search_end: "2013-12-31"
mosaic_composite: "least_cloudy"  # or "all"
stac_api: "planetary-compute"  # or "earth-search"
n_mosaiks_features: 4000
mosaiks_kernel_size: 3
mosaiks_batch_size: 10
model_device: "cpu"  # or "cuda"
dask_n_concurrent_tasks: 8
dask_chunksize: 500
dask_n_workers: 4
dask_threads_per_worker: 4
dask_worker_cores: 4
dask_worker_memory: 2
dask_pip_install: false
mosaiks_col_names: null
setup_rasterio_env: true

```

### - How do I save intermediate data to S3?

To save data to S3, you can simply set the output data path to an S3 bucket, as we simply pass this onto Pandas. Example code:

```python
load_and_save_features(
        input_file_path="test_data.csv",
        path_to_save_data="s3://gs-test-then-delete/test_pr.parquet.gzip",
        context_cols_to_keep_from_input=["lat", "lon"]
    )
```

One requirement for this is to setup the following AWS environment variables:

```yaml
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
```

### - Where can I learn more about the satellite images?

You can explore Microsoft Planetary Computer's [data catalog]([here](https://planetarycomputer.microsoft.com/explore)). It includes information about the satellites and links for further reading. You can also find information on the best image bands to use for images from the [Landsat](https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites) and [Sentinel](https://gisgeography.com/sentinel-2-bands-combinations/) satellites.

### - How do I contribute to this repo as a developer?

To contribute to this repo you can make a feature branch and raise a PR (after making sure that the code works and relevant tests pass).

To set up your dev environment, you can go through the following steps:

1. Clone the mosaiks repository.
2. Run `pip install -e .` in the repo's root folder to install a live local copy of the repository. This can be used in python as import mosaiks.
3. pip install the two requirements files `requirements_dev.txt` and `requirements_test.txt`.
4. Start contributing!

### - What if something isn't working for me?

We are happy to receive feedback on the package. Please do submit an issue, or if you know how to fix it, make a feature branch and raise a PR!
