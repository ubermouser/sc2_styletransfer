import keras as k
import numpy as np
import gzip
import tqdm
from glob import glob
import sys

SAMPLING_RATE = 5

def build_model(input_shape, output_shape):
    input = k.layers.Input(shape=input_shape)
    regularizer = k.regularizers.l2(l=1e-4)
    optimizer = k.optimizers.Adam(lr=1e-3)
    metrics = [k.metrics.categorical_accuracy]
    # normalize the input per channel
    out = k.layers.BatchNormalization(
        axis=3,
        momentum=0.975
    )(input)
    out = k.layers.Conv2D(
        filters=12,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation='relu',
        kernel_regularizer=regularizer,
    )(out)
    out = k.layers.Conv2D(
        filters=18,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation='relu',
        kernel_regularizer=regularizer,
    )(out)
    out = k.layers.Flatten()(out)
    out = k.layers.Dense(
        output_shape[0],
        activation='softmax',
        kernel_regularizer=regularizer
    )(out)

    model = k.models.Model(input=input, output=out)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model


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


def discriminate(games_per_player, validation_split=0.10):
    train_x, train_y = build_dataset(games_per_player)

    model = build_model(input_shape=train_x.shape[1:], output_shape=train_y.shape[1:])
    print(model.summary())
    try:
        model.fit(x=train_x, y=train_y, batch_size=512, shuffle=True, epochs=20,
                  validation_split=validation_split)
    except KeyboardInterrupt:
        pass
    print("Saving model...")
    model.save("trained_model.keras")


def find_outputs(path):
    return glob(path)


def main(sys_args=sys.argv):
    discriminate({
        0: find_outputs("data/attack_agent/*.npy.gz"),
        1: find_outputs("data/random_agent/*.npy.gz"),
    })


if __name__ == '__main__':
    main()
