import numpy as np

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


def find_val_split(data, train_percentage, validation=False):
    length = len(data)
    split = int(np.ceil(train_percentage * length))
    if validation:
        start = split
        end = length
    else:
        start = 0
        end = split

    return start, end


def compute_weights(labels, num_labels):
    argmax_labels = np.argmax(labels, axis=1)
    frequencies = np.bincount(argmax_labels, minlength=num_labels)

    frequencies = frequencies.astype(np.float32) / len(labels)
    target_frequency = 1. / num_labels

    inverse_frequency = np.divide(
        target_frequency,
        frequencies,
        where=frequencies > 0,
        out=np.ones_like(frequencies))
    return inverse_frequency[argmax_labels]


def dask_to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class LabelNormalizer(object):
    def __init__(self, mapping):
        self._mapping = mapping
        self._num_categories = len(set(mapping.values())) + 1
        self._sentinel = self._num_categories - 1

    def __call__(self, data):
        result = np.asarray([self._mapping.get(x, self._sentinel) for x in data])
        result = dask_to_categorical(result, self._num_categories)
        return result


def starcraft_labels():
    label_names = {v: k for k, v in GT_MAPPING.items()}
    label_names[len(label_names)] = 'UNKNOWN'
    return np.asarray([label_names[i] for i in range(len(label_names))])


class StarcraftDataset(object):
    def __init__(self, feature_minimap, feature_screen, race, label, weights, batch_size):
        self.batch_size = batch_size
        self.feature_minimap = feature_minimap
        self.feature_screen = feature_screen
        self.race = race
        self.y = label
        self.weights = weights

        assert len(self.feature_minimap) == len(self.feature_screen) == len(self.race) == \
               len(self.y) == len(self.weights)
        super(StarcraftDataset, self).__init__()

    def __len__(self):
        return int(np.ceil(len(self.y) / float(self.batch_size)))

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min((idx + 1) * self.batch_size, len(self.y))
        batch_minimap = self.feature_minimap[low:high]
        batch_screen = self.feature_screen[low:high]
        batch_race = self.race[low:high]
        batch_y = self.y[low:high]
        batch_weight = self.weights[low:high]

        return (
            {
                'feature_minimap': batch_minimap,
                'feature_screen': batch_screen
            },
            {
                'out_name': batch_y,
                'out_race': batch_race
            },
            {
                'out_name': batch_weight,
                'out_race': batch_weight,
            }

        )

    def all(self):
        return (
            {
                'feature_minimap': self.feature_minimap,
                'feature_screen': self.feature_screen
            },
            {
                'out_name': self.y,
                'out_race': self.race
            },
            {
                'out_name': self.weights,
                'out_race': self.weights,
            }
        )