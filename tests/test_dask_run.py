"""Test dask code."""
import numpy as np
import pandas as pd
import pytest
from dask.delayed import Delayed

from mosaiks.dask import delayed_partition_run
from mosaiks.featurize import RCF, fetch_image_refs, get_dask_gdf
from mosaiks.utils import df_w_latlons_to_gdf


@pytest.mark.slow
def test_delayed_partition_run(
    featurization_params: dict, satellite_config: dict, sample_test_data: pd.DataFrame
):
    """Test delayed partition run."""
    points_gdf = df_w_latlons_to_gdf(sample_test_data)
    points_dgdf = get_dask_gdf(points_gdf, 10)
    points_gdf_w_stac = fetch_image_refs(
        points_dgdf, featurization_params["satellite_search_params"]
    )

    partitions = points_gdf_w_stac.to_delayed()
    model = RCF(
        featurization_params["model"]["num_features"],
        featurization_params["model"]["kernel_size"],
        len(satellite_config["bands"]),
    )
    columns = [
        "feature_%d" % i for i in range(featurization_params["model"]["num_features"])
    ]
    run = delayed_partition_run(
        partitions[0], satellite_config, featurization_params, columns, model
    )
    df = run.compute()
    assert isinstance(run, Delayed)
    assert df.shape == (len(points_gdf), featurization_params["model"]["num_features"])
    assert np.all(df.columns.tolist() == columns)
