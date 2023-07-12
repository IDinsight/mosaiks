# 📸🛰️ MOSAIKS Satellite Imagery Featurization 🛰️📸

MOSAIKS is a Python package that performs parallelized encoding of satellite imagery into easy-to-use features using the MOSAIKS algorithm and Dask for parallel processing. This package enables users to generate feature vectors based on satellite images by providing a list of latitudes and longitudes and Microsoft's Planetary Computer API key. It supports various satellites, resolutions, time periods, and parallelization options.

We implement the MOSAIKS algorithm based on work by [Rolf et al., 2021](https://www.nature.com/articles/s41467-021-24638-z). The authors of this paper implement MOSAIKS for the Landsat-8 satellite returning features for images from 2019, and also provide some pre-computed features [here](https://www.mosaiks.org/).

This packages extends the functionality of the original MOSAIKS implementation in the following ways:
- Extended options for satellites to pull images from
- Added flexibility in choosing resolution of the images, time period for fetching satellite images, etc.
- Flexible upload and download for data
- Parallel processing with Dask to speed up fetching images and creating features.
- Once installed, the package can be run on any machine (with the API key).

The package has been tested via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) and on Landsat-8 or Sentinel-2 imagery. **Please note this package has only been tested on Python 3.10**. Using other versions of Python are expected to raise errors due to dependency conflicts.

For more detailed information on this package and how to use it, please see [this blog post](link to technical blog post). For information on preview and potential use cases for this package, please see [this blog post](link to non-technical blog post)

## Quick Start
This section highlights a demo to help you get features ASAP.

### Step 1: Set-up

Ensure you have all requirements set up: 
1. Install Python 3.10.
2. Install the MOSAIKS package -
    ```sh
    pip install git+https://github.com/IDinsight/mosaiks.git@main
    ```
3. Acquire the Planetary Computer API key from [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
    - In your terminal run the following and fill the API key prompt -
    ```
    planetarycomputer configure
    ```

### Step 2: Test run in a Notebook
The quickest way to test the package is to run it in a notebook. Open up a notebook in the relevant environment (where Step 1 was executed) and run the following (the code is present in the "README_DEMO.ipynb" notebook):

1. **Import dependencies**
```python
import os

# Resolves a conflict in Geopandas. Improves speed. Slower otherwise
os.environ["USE_PYGEOS"] = "0"
```


2. **Import test data. In this case, we are creating random GPS coordinates**
```python
import pandas as pd
import numpy as np

# Create a dataframe with 10 rows of random lats and longs in Uttar Pradesh, India
df = pd.DataFrame(np.random.rand(10, 2)/10 + [26.5, 80.5], columns=['lat', 'lon'])
```


3. **Execute a default run of the `get_features` function:**
```python
from mosaiks import get_features

df_featurised = get_features(
    df['lat'].values,
    df['lon'].values
)

df_featurised
```
The above code executes a default run of the get_features function which executes the featurisation in parallel using Dask.


4. **Run get_features without dask**

It is possible that you want to implement your own parallelisation without dask. For that, you could do a non-parallelised run of the function across your own paralllelisation logic (through code or cloud):
```python
df_featurised = get_features(
    df["lat"].values,
    df["lon"].values,
    parallelize=False
)

df_featurised
```

5. **Run Utility function to load data and save features**

In situations where you want to load data, run featurisation, and save features on disk, quietly, you can use the `load_and_save_features`:
```python
# Create and Save test data
df = pd.DataFrame(np.random.rand(10, 2)/10 + [26.5, 80.5], columns=["lat", "lon"])
df.to_csv("test_data.csv")

# Run the load and save function
from mosaiks.extras import load_and_save_features

load_and_save_features(input_file_path="test_data.csv",
                       path_to_save_data="test_features.csv",
                       context_cols_to_keep_from_input=["lat", "lon"])
```

## Core functionality of the system

The high-level flow of our featurisation pipeline is the following:

1. The User feeds 'lat' and 'lon' lists for points they want to featurise
    - The user also adds relevant parameters to the function (refer to FAQs)
3. For each GPS coordinate, the function fetches [STAC](https://stacspec.org/en) references to satellite images
4. Once found, the function fetches the images
5. Function converts each image into features using the MOSAIKS algorithm
6. Lastly, the function returns a dataframe with the features, 'lat' and 'lon' columns, and any other columns if specified by the user


## Repository structure

```
.
├── playground
│   └── test_big_pipeline_function.ipynb -- demo code for running pipeline function
│
├── src -- source code
│   ├── mosaiks
│   │   ├── pipeline.py -- pipeline code: takes in GPS coordinates and config dictionaries; returns features
│   │   ├── utils.py -- utilities for pipeline code
│   │   ├── dask.py -- wrapper functions and utilities for Dask parallel processing
│   │   ├── extras/ -- wrapper functions for pipeline code; includes file I/O operations and checks for configuration files
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

# FAQs

### - How do I get access to the Planetary Computer API key?
Update

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
    image_bands: List[str] = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"], # For options, read the satellite docs
    buffer_distance: int = 1200,
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
    dask_client_type: str = "local", # or gateway
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
buffer_distance: 1200
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
dask_client_type: "local"  # or "gateway"
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

To save data to S3, you can use the standard save_dataframe command from the utils file. Example code:
```python
save_dataframe(df, "s3://gs-test-then-delete/test_pr.parquet.gzip", compression="gzip")
```
One requirement for this is to setup the following AWS environment variables:
```yaml
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
```


### - Can you share an example of a project where we have used this?

Yes! See the [mosaiks_ml](https://github.com/IDinsight/mosaiks_ml) repository for example ML models built using these features

### - How do I contribute to this repo as a developer?

To contribute to this repo you can make a feature branch and raise a PR (after making sure that the code works and relevant tests pass).

To set up your dev environment, you can go through the following steps:
1. Clone the mosaiks repository.
2. Run `pip install -e .` in the repo's root folder to install a live local copy of the repository. This can be used in python as import mosaiks.
3. pip install the two requirements files "requirements_dev.txt" and "requirements_test.txt".
4. Start contributing!