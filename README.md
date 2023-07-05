# MOSAIKS Satellite Imagery Featurization

This repository holds the code to perform parallelized encoding of satellite imagery into easy-to-use features using the MOSAIKS algorithm and Dask for parallel processing.
The package simply requires users to upload a file of GPS coordinates for which they need satellite image features, and optionally set some configurable parameters. The package returns a vector of features corresponding to each coordinate, which users can then use for to train models for (linear) prediction or classification problems.


The MOSAIKS algorithm for deriving features from satellite imagery implemented here is based on work by [Rolf et al., 2021](https://www.nature.com/articles/s41467-021-24638-z). The authors of this paper implement MOSAIKS for the Landsat-8 satellite returning features for images from 2019, and also provide some pre-computed features [here](https://www.mosaiks.org/).

This packages extends the functionality of the original MOSAIKS implementation in the following ways:
- Extended options for satellites to pull images from
- Added flexibility in choosing resolution of the images, time period for fetching satellite images, etc.
- Flexible upload and download for data
- Parallel processing with Dask to speed up fetching images and creating features.
- Once installed, the package can in principle be run on any machine.
The package has been tested via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) and on Landsat-8 or Sentinel-2 imagery.

For more detailed information on this package and how to use it, please see [this blog post](link to technical blog post). For information on previou and potential use cases for this package, please see [this blog post](link to non-technical blog post)


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

---

## FAQs

- How do I save intermediate data to S3?

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
