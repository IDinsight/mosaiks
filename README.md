# MOSAIKS Satellite Imagery Featurization

MOSAIKS is a Python package that performs parallelized encoding of satellite imagery into easy-to-use features using the MOSAIKS algorithm and Dask for parallel processing. This package enables users to generate feature vectors based on satellite images by providing a list of latitudes and longitudes and Microsoft's Planetary Computer API key. It supports various satellites, image sizes, time periods, and parallelization options.

We implement the MOSAIKS algorithm based on work by [Rolf et al., 2021](https://www.nature.com/articles/s41467-021-24638-z) based on random convolutional features. The authors of this paper make a global cross-section of pre-computed features using Planet imagery from 2019 available at [mosaiks.org](https://www.mosaiks.org/), along with tutorials and related research.

This package extends the functionality of the original MOSAIKS implementation in the following ways:

- Flexibility in choice of satellite (tested for Landsat-8 and Sentinel-2)
- Ability to select the timeframe from which to pull imagery
- Flexibility in choosing the size of the images centred on points of interest, image bands used, etc.
- Flexible upload and download of data
- Parallel processing with Dask to speed up fetching images and creating features.
- Once installed, the package can be run on any machine (with the API key).

The package has been tested via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) on Landsat-8 or Sentinel-2 imagery. **Please note this package has only been tested on Python 3.10 and 3.11**. Using other versions of Python are expected to raise errors due to dependency conflicts.

For more detailed information on this package and how to use it, please see [this blog post](https://idinsight.github.io/tech-blog/blog/mosaiks_part_1/). For information on preview and potential use cases for this package, please see [this blog post](https://www.idinsight.org/?post_type=article&p=20518&preview=true).  For more information on MOSAIKS and previous use cases, see the MOSAIKS website [here](https://www.mosaiks.org/).

Users of this package should acknowledge *IDinsight* and reference the MOSAIKS RCF algorithm as Rolf, Esther, et al. "A generalizable and accessible approach to machine learning with global satellite imagery." *Nature communications* 12.1 (2021): 4392.

## Quick Start

This section highlights a demo to help you get features ASAP.

### Step 1: Set-up

Ensure you have all requirements set up:

1. Install Python 3.10 or 3.11.
2. Install the MOSAIKS package -

    ```sh
    pip install git+https://github.com/IDinsight/mosaiks
    ```

    or

    ```sh
    pip install mosaiks
    ```

4. Acquire the Planetary Computer API key from [Microsoft Planetary Computer (MPC)](https://planetarycomputer.microsoft.com/). We provide detailed instructions for getting an API key in the FAQs section of this README.

    In your terminal, run the following and fill in the API key prompt -

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
        datetime="2017", # or ["2013-01-01", "2013-12-31"] or ...
        image_width=1000,
        )

    df_featurised
    ```

    The above code executes a default run of the get_features function which executes the featurisation.

4. **Run get_features with Dask parallelization**

    To run the code with the built-in Dask parallelization, set `parallelize` to `True` and `dask_chunksize` to a suitable integer given the size of your dataset.

    ```python
    df_featurised = get_features(
        lats,
        lons,
        datetime="2017", # or ["2013-01-01", "2013-12-31"] or ...
        image_width=1000,
        parallelize=True,
        dask_chunksize=2, # set this to 200+ to see benefits from parallization
    )

    df_featurised
    ```

    Check out `get_features`' docs for parameters to control the in-built parallelization scheme.

## Core functionality of the system

The high-level flow of our featurisation pipeline is the following:

1. The User feeds 'lat' and 'lon' lists for points they want to featurize
    - The user also adds relevant parameters to the function (see docstrings and FAQs)
2. For each GPS coordinate, the function fetches [STAC](https://stacspec.org/en) references to satellite images
3. Once found, the function fetches the images (either all or only the least cloudy depending on the `image_composite_method` parameter)
4. Function converts each image into features using the MOSAIKS algorithm
5. Lastly, the function returns a dataframe with the features alongside the STAC references to the image(s) used to create the features from.

## Repository structure

```
 ├── mosaiks
 │   ├── fetch -- fetching images
 │   ├── featurize -- converting images to MOSAIKS features
 │   └── pipeline -- get_features() is here.
 ├── tests -- pytests (need to install requirements_test to run)
 ├── README.md -- No but actually, read this.
 ├── README_DEMO.ipynb
 ├── requirements.txt
 ├── requirements_dev.txt
 ├── requirements_test.txt
 └── LICENSE
```

## FAQs

### • How do I get access to the Planetary Computer API key?

If you are running mosaiks locally or on a non-MPC server, then you need an access token for the satellite image database.

1. If you do not have an MPC account, go [here](https://planetarycomputer.microsoft.com/explore). You should see a “Request Access” button in the top right corner.

    It opens up a form which you should fill in. NB: Use your personal email ID rather than an institutional one. If you already have a Microsoft account, use the email ID (non-institutional) associated with it. Otherwise, you also have the additional step of creating a Microsoft account for the email ID you want to use for the MPC Hub.

    Once you submit the form, you should receive an email within a week granting you access to the hub.

2. To get the token, go [here](https://pccompute.westeurope.cloudapp.azure.com/compute/hub/token). Request and copy the new token, and save it.

3. On your local/virtual machine. Run `pip install planetary-compute` and `planetarycomputer configure` from the console, and paste in the API key you generated.

4. You only need to do this once and should be able to access the database smoothly every time you run the pipeline after this.

5. More information is available [here](https://planetarycomputer.microsoft.com/docs/reference/sas/).

### • Can you tell me about all the parameters that I can use in the `get_features`?

Below are all the parameters and defaults that `get_features` uses. 

**NOTE:** You'll probably want to leave most of these as defaults - see the `.yaml` file example that follows for the subset of parameters you most likely want to change.

```python
def get_features(
    latitudes: List[float],
    longitudes: List[float],
    datetime: str or List[str] or callable,
    satellite_name: str = "landsat-8-c2-l2", # or "sentinel-2-l2a"
    image_resolution: int = 30, # or 10 for Sentinel
    image_bands: List[str] = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"], # For options, see FAQs below
    image_width: int = 3000,
    min_image_edge: int = 30,
    image_composite_method: str = "least_cloudy",   # or "all" to create a multi-image median composite before featurisation
    image_dtype: str = "int16", # or "int32" or "float". "int8" not supported.
    stac_api_name: str = "planetary-compute", # or "earth-search"
    n_mosaiks_features: int = 4000,
    mosaiks_kernel_size: int = 3,
    mosaiks_random_seed_for_filters: int = 768,
    model_device: str = "cpu",  # or "cuda" if NVIDIA GPU available
    parallelize: bool = False,
    dask_chunksize: int = 500,
    dask_client: Optional[Client] = None, # Provide to override the default per-run LocalCluster creation
    dask_n_workers: Optional[int] = None, # Set to None to auto-select maximum
    dask_threads_per_worker: Optional[int] = None, # Set to None to auto-select maximum
    dask_n_concurrent_tasks: Optional[int] = None, # Set to None to set equal to number of threads
    dask_sort_points_by_hilbert_distance: bool = True,
    setup_rasterio_env: bool = True,
) -> pd.DataFrame
```

You can also set these parameters in a `.yml` file, read the file, and then input the parameters as **kwargs. Here is an example of a `.yml` file with some common project-specific parameters set, leaving everything else as default:

```yml
datetime: "2017",
satellite_name: "landsat-8-c2-l2",
image_resolution: 30,
image_bands:
  - "SR_B2"
  - "SR_B3"
  - "SR_B4"
  - "SR_B5"
  - "SR_B6"
  - "SR_B7"
image_width: 3000,
image_composite_method: "least_cloudy",
n_mosaiks_features: 4000,
model_device: "cpu", # or "gpu" if NVIDIA GPU available
parallelize: True,
dask_chunksize: 500,
```

### • How do I choose satellite parameters?

We have tested this package for 2 satellites: Sentinel-2 and Landsat-8.
Sentinel-2 images are available starting from 23. June 2015 (relevant for `datetime`) at 10m resolution (`image_resolution`) for 13 spectral bands (`image_bands`).
Landsat-8 images are available starting 11th February 2013, at 30m resolution and for 11 spectral bands.

You can explore Microsoft Planetary Computer's [data catalog]([here](https://planetarycomputer.microsoft.com/explore)) to learn more -- it includes information about the satellites and links for further reading. You can also find information on the best image bands to use for images from the [Landsat](https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites) and [Sentinel](https://gisgeography.com/sentinel-2-bands-combinations/) satellites.

### • How do I contribute to this repo as a developer?

To contribute to this repository, you can make a feature branch and raise a PR (after making sure that the code works and relevant tests pass).

To set up your dev environment, you can go through the following steps:

1. Clone the mosaiks repository.
2. Run `pip install -e .` in the repo's root folder to install a live local copy of the repository. This can be used in Python as import mosaiks.
3. pip install the two requirements files `requirements_dev.txt` and `requirements_test.txt`.
4. Start contributing!

### • What if something isn't working for me?

We are happy to receive feedback on the package. Please do submit an issue, or if you know how to fix it, make a feature branch and raise a PR!
