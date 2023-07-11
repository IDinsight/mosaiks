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
The quickest way to test the package is to run it in a notebook. Open up a notebook in the relevant environment (where Step 1 was executed) and go through the following:

1. **Import packages**
```python
import pandas as pd
import os

# Resolves a conflict in Geopandas. Improves speed. Slower otherwise
os.environ["USE_PYGEOS"] = "0"

from mosaiks import get_features
```


2. **Import test data. In this case, we are creating random GPS coordinates**
```python
import pandas as pd
import numpy as np

# Create a dataframe with 10 rows of random lats and longs in Uttar Pradesh, India
df = pd.DataFrame(np.random.rand(10, 2)/10 + [26.5, 80.5], columns=["lat", "lon"])
```


3. **Execute a default run of the `get_features` function:**
```python
df_featurised = get_features(
    df["lat"].values,
    df["lon"].values
)

df_featurised
```
The above code executes a default run of the get_features function which executes the featurisation in parallel using Dask.


4. **Test out a non-dask run**

It is possible that you want to implement your own parallelisation without dask. For that, you could do a non-parallelised run of the function across your own paralllelisation logic (through code or cloud):
```python
df_featurised = get_features(
    df["lat"].values,
    df["lon"].values,
    parallelize=False
)

df_featurised
```

5. **Test Utility function to load data and save features**
In situations where you want to also want to load data, run featurisation, and save features on disk you can use the `load_and_save_features`:
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

## Pipeline

1. Load dataset containing lat-lon coordinates for which to process images
2. Read config parameters (e.g. image size to be processed (buffer), year, satellite, number of features to produce, etc.). See `config/*.yaml` files in this repsitory for configurable parameters.
3. Fetch STAC references to images that overlap each point
4. Fetch the images
5. Convert each image into features using the MOSAIKS algorithm
6. Save features to file to be used for ML modelling (see the [mosaiks_ml](https://github.com/IDinsight/mosaiks_ml) repository for example ML models built using these features)

## How to Run

0. Clone this repository
1. Run `make setup-env` to make an environment called "mosaiks" and install the required libraries (or do so manually)

2. Run `pip install -e .` to install a live local copy of the repository. This can be used in python as `import mosaiks`.

3. Data request authentication
    - If running on the Microsoft Planetary Computer virtual machine, all data requests are automatically authenticated.
    - If running elsewhere, make sure that the `planetary-computer` python package is installed and run the command `planetarycomputer configure` in the console, entering the API key when prompted. See [here](https://planetarycomputer.microsoft.com/docs/concepts/sas/#:~:text=data%20catalog.-,planetary%2Dcomputer%20Python%20package,-The%20planetary%2Dcomputer) for more information.

4. Update configuration files in `config/`, or make your own config dictionaries.

5. Place a `csv` file containing the latitude and longitude coordinates of the points in a specific folder or load a file with latitude and longitude coordinates.

6. Run the wrapper pipeline function, or the pipeline function (see details below). Within the notebook:
    - Choose the Dask cluster/gateway as desired
    - Make sure to read the correct entry in the data catalog for the point coordinates file

## Repository structure
```
.
├── config
│   ├── featurisation_config.yaml -- configuration for image fetching function, MOSAIKS model and Dask
│   ├── rasterioc_config.yaml -- configuration for Rasterio
│   └── satellite_config.yaml -- configuration for satellite images (satellite name, year, resolution, etc.)
|
├── playground
│   ├── test_big_pipeline_function.ipynb -- demo code for running pipeline function
|
├── src -- source code
│   ├── mosaiks
│   │   ├── pipeline.py -- pipeline code: takes in GPS coordinates and config dictionaries; returns features
│   │   └── utils.py -- utilities for pipeline code
│   │   ├── dask.py -- wrapper functions and utilities for Dask parallel processing
│   │   ├── extras/ -- wrapper functions for pipeline code; includes file I/O operations and checks for configuration files
│   │   ├── featurize/ -- code for featurisation from images
│   │   ├── fetch/ -- code for fetching images
└── tests/ -- unit tests
├── Makefile -- set up file for installing package code code
├── project_config.cfg -- repository configuration
├── pyproject.toml -- repository install configuration
├── pytest.ini -- unit test configuration
├── requirements_test.txt -- unit test package install requirements
├── requirements.txt -- package install requirements

```

# FAQs

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

### - How do I get access to Planetary Computer API key?


