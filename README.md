# MOSAIKS Satellite Imagery Featurization

This repository holds the code to perform parallelized encoding of satellite imagery into easy-to-use features using the MOSAIKS algorithm and Dask.

It can currently run on [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) and has been tested on Landsat-8 or Sentinel-2 imagery.

## Pipeline

1. Load dataset containing lat-lon coordinates for which to process images
2. Read config parameters (e.g. image size to be process (buffer), year, satellite, number of features to produce, etc.)
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

4. Place a `csv` file containing the latitude and longitude coordinates of the points that are to be requested under `data/` and add or update its entry in `config/data_catalog.yaml`.

5. Run the `playground/featurize_pipeline.ipynb` notebook. Within the notebook:
    - Choose the Dask cluster/gateway as desired
    - Make sure to read the correct entry in the data catalog for the point coordinates file

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