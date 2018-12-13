from functools import partial
import gzip

import h5py_cache
import keras as k
from keras.utils import HDF5Matrix
import numpy as np
import tqdm

from pysc2.lib import features

SAMPLING_RATE = 5

RACE_MAPPING = {b'Terran': 0, b'Zerg': 1, b'Protoss': 2}
PLAYER_MAPPING = {
    # Shared:
    'Has': 0,
    'HeRoMaRinE': 1, 'HeroMarine': 1,
    'Lambo': 2,
    'Neeb': 3,
    'Nerchio': 4,
    'Serral': 5,
    'ShoWTimE': 6,
    'SpeCial': 7,
    # WCS Global only:
    'Classic': 8,
    'Dark': 9,
    'Maru': 10,
    'Rogue': 11,
    'Stats': 12,
    'TYTY': 13,
    'Zest': 14,
    'sOs': 15,
    # WCS Montreal only:
    'Reynor': 16,
    'TIME': 17,
    'JonSnow': 18,
    'Denver': 19,
    'Semper': 20,
    'Zanster': 21,
    'Clem': 22,
    'DnS': 23,
    'Stephano': 24
}
GT_MAPPING = ['AeiS', 'Beez', 'Bioice', 'Bly', 'Buster', 'CalebAracous', 'Cham',
       'Clem', 'Crow', 'Cuddlebear', 'Cyan', 'Daydreamer', 'Denver',
       'DisK', 'DnS', 'Dolan', 'Elazer', 'ElegancE', 'Ethereal', 'ExpecT',
       'Frontstab', 'Future', 'GoGojOey', 'GogojOey', 'Harstem', 'Has',
       'HeroMarine', 'Hjax', 'HuT', 'JackO', 'JadedShard', 'Jason',
       'JonSnow', 'Kelazhur', 'Kozan', 'Lambo', 'Lighting', 'LiquidTLO',
       'MDStephano', 'MaSa', 'Mackintac', 'Mage', 'Mana', 'Namshar',
       'NeXa', 'Neeb', 'Nerchio', 'Nice', 'Ninja', 'NoRegreT', 'PengWin',
       'Poizon', 'Probe', 'PtitDrogo', 'Pure', 'Raze', 'Reynor', 'Rhizer',
       'Scarlett', 'Semper', 'Serral', 'ShaDoWn', 'ShoWTimE', 'Silky',
       'Snute', 'SpaceJam', 'SpeCial', 'Stephano', 'StevenBonnel',
       'Sugar', 'THERIDDLER', 'TIME', 'TRUE', 'TheWagon', 'TooDming',
       'Wilo', 'Zanster', 'cloudy', 'eonblu', 'harstem', 'jheffe', 'mLty',
       'soul', 'thermy', 'uThermaL']
GT_MAPPING = {name: idx for idx, name in enumerate(GT_MAPPING)}
GT_MAPPING['HeRoMaRinE'] = GT_MAPPING['HeroMarine']

FEATURE_MINIMAP_PLAYER_RELATIVE = 5
FEATURE_SCREEN_PLAYER_RELATIVE = 5
FEATURE_SCREEN_UNIT_CATEGORY = 6


class FeatureNormalizer(object):
    def __init__(self, layers):
        self._layers = layers

    def __call__(self, data):
        result = data[:, self._layers, :, :]
        return result


class LabelNormalizer(object):
    def __init__(self, mapping):
        self._mapping = mapping
        self._num_categories = len(set(mapping.values())) + 1
        self._sentinel = self._num_categories - 1

    def __call__(self, data):
        result = np.asarray([self._mapping.get(x, self._sentinel) for x in data])
        result = k.utils.to_categorical(result, self._num_categories)
        return result


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


class StarcraftDataset(k.utils.Sequence):
    def __init__(self, feature_minimap_proto, feature_screen_proto, race_proto, label_proto, batch_size):
        self.batch_size = batch_size
        self.feature_minimap_proto = feature_minimap_proto
        self.feature_screen_proto = feature_screen_proto
        self.race_proto = race_proto
        self.y_proto = label_proto

        self.__setstate__({})
        assert len(self.feature_minimap) == len(self.feature_screen) == len(self.race) == len(self.y)

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

    def __len__(self):
        return int(np.ceil(len(self.y) / float(self.batch_size)))

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min((idx + 1) * self.batch_size, len(self.y))
        batch_minimap = self.feature_minimap[low:high]
        batch_screen = self.feature_screen[low:high]
        batch_race = self.race[low:high]
        batch_y = self.y[low:high]

        return (
            {
                'feature_minimap': batch_minimap,
                'feature_screen': batch_screen
            },
            {
                'out_name': batch_y,
                'out_race': batch_race
            }
        )


def rolling_window(a, window, stride=1):
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0] * stride,) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def sample_input_pairs(game_trace, sampling_rate=SAMPLING_RATE):
    sampled_input = np.array(game_trace[::sampling_rate], copy=True)
    return rolling_window(sampled_input, 2)


def preprocess_input(data_x):
    shape = data_x.shape
    data_x = data_x.reshape(shape[0], shape[1] * shape[2], shape[3], shape[4])
    data_x = np.transpose(data_x, [0, 2, 3, 1])  # channels last
    data_x = data_x[:, :, :, [5, 7+5]]  # use only the "player_id" tensor because of memory issues
    return data_x


def build_dataset(games_per_player, max_num_games=100):
    data_x = []
    data_y = []
    # extract games in this weird way so the validation set has an equal distribution of games per
    #  player and so the train set has no overlap with validation set episodes
    num_games = len(next(iter(games_per_player.values())))
    num_games = min(num_games, max_num_games)
    for i_game in tqdm.tqdm(range(num_games)):
        for i_player, player in enumerate(games_per_player.keys()):
            game_trace = np.load(gzip.open(games_per_player[player][i_game]))
            trace_pairs = sample_input_pairs(game_trace)
            del game_trace
            data_x.append(trace_pairs)
            data_y.append([i_player] * len(trace_pairs))

    data_x = np.concatenate(data_x)
    data_x = preprocess_input(data_x)

    data_y = np.concatenate(data_y)
    data_y = k.utils.to_categorical(data_y, len(games_per_player))

    return data_x, data_y


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
        #normalizer=FeatureNormalizer(FEATURE_MINIMAP_PLAYER_RELATIVE)
    )
    screen_data = partial(
        CachingHDF5Matrix,
        input_path,
        'feature_screen',
        train_percentage=train_percentage,
        val_split=val_split,
        #normalizer=FeatureNormalizer([FEATURE_SCREEN_PLAYER_RELATIVE, FEATURE_SCREEN_UNIT_CATEGORY])
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

    dataset = StarcraftDataset(minimap_data, screen_data, race_data, y_data, batch_size=batch_size)
    return dataset
