import os

import dask.array as da
import numpy as np


from .utils import compute_weights, FeatureNormalizer, FEATURE_MINIMAP_PLAYER_RELATIVE, \
    FEATURE_SCREEN_UNIT_CATEGORY, FEATURE_SCREEN_PLAYER_RELATIVE, GT_MAPPING, LabelNormalizer, \
    RACE_MAPPING, StarcraftDataset, starcraft_labels, find_val_split


def starcraft_dataset(input_path, batch_size=1024, train_percentage=1., val_split=False):
    minimap_data = da.from_zarr(os.path.join(input_path, 'feature_minimap'))
    minimap_data = FeatureNormalizer(FEATURE_MINIMAP_PLAYER_RELATIVE)(minimap_data)

    screen_data = da.from_zarr(os.path.join(input_path, 'feature_screen'))
    screen_data = FeatureNormalizer([FEATURE_SCREEN_PLAYER_RELATIVE, FEATURE_SCREEN_UNIT_CATEGORY])(screen_data)

    race_data = da.from_zarr(os.path.join(input_path, 'race')).compute()
    race_data = LabelNormalizer(RACE_MAPPING)(race_data)

    y_data = da.from_zarr(os.path.join(input_path, 'name')).compute()
    y_data = LabelNormalizer(GT_MAPPING)(y_data)

    weights = compute_weights(y_data, len(GT_MAPPING))

    start, end = find_val_split(y_data, train_percentage, val_split)

    dataset = StarcraftDataset(
        minimap_data[start:end],
        screen_data[start:end],
        race_data[start:end],
        y_data[start:end],
        weights[start:end],
        batch_size=batch_size)
    return dataset