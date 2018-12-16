from functools import partial
import os
import gzip

import h5py_cache
import keras as k
from keras.utils import HDF5Matrix
import numpy as np

from pysc2.lib import features

from .utils import StarcraftDataset, RACE_MAPPING, PLAYER_MAPPING, GT_MAPPING, \
    FEATURE_SCREEN_PLAYER_RELATIVE, FEATURE_SCREEN_UNIT_CATEGORY, FEATURE_MINIMAP_PLAYER_RELATIVE, \
    FeatureNormalizer, LabelNormalizer

SAMPLING_RATE = 5


class CachingHDF5Matrix(HDF5Matrix):
    def __init__(self, datapath, dataset, *nargs, train_percentage=100., val_split=False, **kwargs):
        if datapath not in self.refs:
            f = h5py_cache.File(datapath, chunk_cache_mem_size=1024**3 // 10)
            self.refs[datapath] = f

        length = len(self.refs[datapath][dataset])
        split = int(np.ceil(train_percentage * length))
        if val_split:
            start = split
            end = length
        else:
            start = 0
            end = split

        super(CachingHDF5Matrix, self).__init__(
            datapath, dataset, *nargs, start=start, end=end, **kwargs)

        # bugfix: ndim is not changed to reflect the normalizer's output
        if self.normalizer is not None:
            self._base_ndim = self.normalizer(self.data[0:1]).ndim
        else:
            self._base_ndim = self.data.ndim

    @property
    def ndim(self):
        return self._base_ndim


class CachingStarcraftDataset(StarcraftDataset, k.utils.Sequence):
    def __init__(self, feature_minimap_proto, feature_screen_proto, race_proto, label_proto, batch_size):
        self.feature_minimap_proto = feature_minimap_proto
        self.feature_screen_proto = feature_screen_proto
        self.race_proto = race_proto
        self.y_proto = label_proto

        super(CachingStarcraftDataset, self).__init__(
            feature_minimap_proto(), feature_screen_proto(), race_proto(), label_proto(), batch_size)

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.feature_minimap = self.feature_minimap_proto()
        self.feature_screen = self.feature_screen_proto()
        self.race = self.race_proto()
        self.y = self.y_proto()

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['feature_minimap']
        del odict['feature_screen']
        del odict['race']
        del odict['y']
        return odict


def starcraft_labels():
    label_names = {v: k for k, v in GT_MAPPING.items()}
    label_names[len(label_names)] = 'UNKNOWN'
    return np.asarray([label_names[i] for i in range(len(label_names))])


def starcraft_dataset(input_path, batch_size=1024, train_percentage=1., val_split=False):
    #label_names = {name: idx for idx, name in enumerate(np.unique(h5py.File(input_path)['name']))}
    label_names = GT_MAPPING
    minimap_data = partial(
        CachingHDF5Matrix,
        input_path,
        'feature_minimap',
        train_percentage=train_percentage,
        val_split=val_split,
        normalizer=FeatureNormalizer(FEATURE_MINIMAP_PLAYER_RELATIVE)
    )
    screen_data = partial(
        CachingHDF5Matrix,
        input_path,
        'feature_screen',
        train_percentage=train_percentage,
        val_split=val_split,
        normalizer=FeatureNormalizer([FEATURE_SCREEN_PLAYER_RELATIVE, FEATURE_SCREEN_UNIT_CATEGORY])
    )
    race_data = partial(
        CachingHDF5Matrix,
        input_path,
        'race',
        train_percentage=train_percentage,
        val_split=val_split,
        normalizer=LabelNormalizer(RACE_MAPPING)
    )
    y_data = partial(
        CachingHDF5Matrix,
        input_path,
        'name',
        train_percentage=train_percentage,
        val_split=val_split,
        normalizer=LabelNormalizer(label_names)
    )

    dataset = CachingStarcraftDataset(minimap_data, screen_data, race_data, y_data, batch_size=batch_size)
    return dataset
