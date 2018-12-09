from functools import partial
import gzip

import h5py
import keras as k
import numpy as np
import tqdm


SAMPLING_RATE = 5


class DataNormalizer(object):
    def __init__(self, channels):
        self._channels = list(channels)

    def __call__(self, data):
        result = np.transpose(data, [0, 2, 3, 1])  # channels last
        result = result[:, :, :, self._channels]
        return result


class LabelNormalizer(object):
    def __init__(self, num_classes):
        self._num_classes = num_classes

    def __call__(self, data):
        result = k.utils.to_categorical(data, self._num_classes)
        return result


class StarcraftDataset(k.utils.Sequence):
    def __init__(self, data_normalizer, label_normalizer, batch_size):
        self.x_proto = data_normalizer
        self.y_proto = label_normalizer

        self.x = self.x_proto()
        self.y = self.y_proto()
        assert len(self.x) == len(self.y)

        self.batch_size = batch_size

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['x']
        del odict['y']
        return odict

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.x = self.x_proto()
        self.y = self.y_proto()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min((idx + 1) * self.batch_size, len(self.y))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]

        return batch_x, batch_y


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


def shuffle_dataset(input_path):
    pass


def starcraft_dataset(input_path, batch_size=2048, num_classes=2):
    x_data = partial(k.utils.HDF5Matrix, input_path, 'feature_minimap', normalizer=DataNormalizer([5]))
    y_data = partial(k.utils.HDF5Matrix, input_path, 'player', normalizer=LabelNormalizer(num_classes))

    dataset = StarcraftDataset(x_data, y_data, batch_size=batch_size)
    return dataset
