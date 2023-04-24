"""Tests for featurization functions."""
import pytest
import torch

from mosaiks.featurize import RCF


def test_random_conv_features():
    """Test random convolutional features."""
    model = RCF(num_features=666, kernel_size=5, num_input_channels=3)

    x1 = torch.empty(3, 10, 10)
    x2 = torch.ones(3, 10, 10) * torch.nan
    out1 = model(x1)
    out2 = model(x2)

    assert hasattr(model, "conv1")
    assert (
        model.conv1.weight.requires_grad == False
        and model.conv1.bias.requires_grad == False
    )
    assert out1.dim() == 1 and out2.dim() == 1
    assert out2.shape[0] == 666 and out2.shape[0] == 666


# @delayed
# def partition_run(
#     df, satellite_config, featurization_params, mosaiks_column_names, model, device
# ):

#     data_loader = create_data_loader(
#         df, satellite_config, featurization_params["model"]["batch_size"]
#     )
#     X_features = create_features(
#         data_loader,
#         featurization_params["model"]["num_features"],
#         len(df),
#         model,
#         device,
#         satellite_config["min_image_edge"],
#     )

#     df = pd.DataFrame(X_features, index=df.index.copy(), columns=mosaiks_column_names)

#     return df


# def test_image_fetch(
#     sample_test_data, featurization_params, satellite_config, local_cluster_client
# ):
#     """
#     Check images can be fetched, features generated and basic summary stats are as expected
#     """
#     points_gdf_with_stac = fetch_image_refs(
#         sample_test_data,
#         featurization_params["dask"]["n_partitions"],
#         featurization_params["satellite_search_params"],
#     )

#     mosaiks_column_names = [
#         f"mosaiks_{i}" for i in range(featurization_params["model"]["num_features"])
#     ]

#     partitions = points_gdf_with_stac.to_delayed()

#     model = RCF(
#         featurization_params["model"]["num_features"],
#         featurization_params["model"]["kernel_size"],
#         len(satellite_config["bands"]),
#     )

#     i = 0
#     p = partitions[i]
#     f = partition_run(
#         p,
#         satellite_config,
#         featurization_params,
#         mosaiks_column_names,
#         model,
#         featurization_params["model"]["device"],
#         dask_key_name=f"run_{i}",
#     )
#     df_future = local_cluster_client.compute(f)
#     for f in as_completed([df_future]):
#         df = f.result()

#     # n features
#     assert len(df.columns) == 4000
#     # non negative
#     assert df.min().min() >= 0
