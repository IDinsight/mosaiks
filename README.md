# MOSAIKS Satellite Imagery Featurization

This repository holds the code to perform parallelized encoding of satellite imagery into easy-to-use features using the MOSAIKS algorithm and Dask.

It can currently run on [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) and has been tested on Landsat-8 or Sentinel-2 imagery.

Pipeline:
1. Load dataset containing lat-lon coordinates for which to process images
2. Read config parameters (e.g. image size to be process (buffer), year, satellite, number of features to produce, etc.)
3. Fetch STAC references to images that overlap each point
4. Fetch the images
5. Convert each image into features using the MOSAIKS algorithm
6. Save features to file to be used for ML modelling (see the [mosaiks_ml](https://github.com/IDinsight/mosaiks_ml) repository for example ML models built using these features)
