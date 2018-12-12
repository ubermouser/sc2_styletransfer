import numpy as np
import keras as k
import tensorflow as tf

from pysc2.lib import features


FEATURE_MINIMAP_PLAYER_RELATIVE = 5
FEATURE_SCREEN_PLAYER_RELATIVE = 5
FEATURE_SCREEN_UNIT_CATEGORY = 6


def build_minimap_input(dataset):
    input = k.layers.Input(
        shape=dataset.feature_minimap.shape[1:],
        dtype=dataset.feature_minimap.dtype,
        name='feature_minimap')

    def minimap_input(out):
        out = k.backend.cast(out, np.int32)
        player_mask = out[:, FEATURE_MINIMAP_PLAYER_RELATIVE, :, :]
        one_hot = k.backend.one_hot(player_mask, len(features.PlayerRelative))
        return one_hot
    return input, k.layers.Lambda(minimap_input)(input)


def build_screen_input(dataset):
    input = k.layers.Input(
        shape=dataset.feature_screen.shape[1:],
        dtype=dataset.feature_screen.dtype,
        name='feature_screen')

    def screen_input(out):
        out = k.backend.cast(out, np.int32)
        player_mask = out[:, FEATURE_SCREEN_PLAYER_RELATIVE, :, :]
        unit_mask = out[:, FEATURE_SCREEN_UNIT_CATEGORY, :, :]

        one_hot_player = k.backend.one_hot(player_mask, len(features.PlayerRelative))
        one_hot_units = k.backend.one_hot(unit_mask, len(features.static_data.UNIT_TYPES) + 1)
        return k.layers.concatenate([one_hot_player, one_hot_units])

    return input, k.layers.Lambda(screen_input)(input)


def chain_conv_2d(filters, input, regularizer):
    out = input
    for filter in filters:
        out = k.layers.Conv2D(
            filters=filter,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation='relu',
            kernel_regularizer=regularizer,
        )(out)

    return out


def build_model(dataset):
    feature_minimap, minimap_input = build_minimap_input(dataset)
    feature_screen, screen_input = build_screen_input(dataset)

    regularizer = k.regularizers.l2(l=1e-4)
    optimizer = k.optimizers.Adam(lr=1e-3)
    metrics = [k.metrics.categorical_accuracy]
    # minimap channel:
    # reduce # of channels in the input
    out_1 = k.layers.Conv2D(
        filters=2,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation='relu',
        kernel_regularizer=regularizer
    )(minimap_input)
    # collapse spatial dimensions
    out_1 = chain_conv_2d([12, 18, 24, 30], out_1, regularizer)
    # screen channel:
    # reduce # of channels in the input
    out_2 = k.layers.Conv2D(
        filters=3,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation='relu',
        kernel_regularizer=regularizer
    )(screen_input)
    # collapse spatial dimensions
    out_2 = chain_conv_2d([8, 12, 16, 20], out_2, regularizer)
    # higher-level decision-making:
    out_shape = dataset.y.shape[-1]
    out_1 = k.layers.Flatten()(out_1)
    out_2 = k.layers.Flatten()(out_2)
    out = k.layers.Concatenate()([out_1, out_2])
    out = k.layers.Dense(
        100,
        activation='relu',
        kernel_regularizer=regularizer,
        name='out_descriptor'
    )(out)
    out = k.layers.Dense(
        out_shape,
        activation='softmax',
        kernel_regularizer=regularizer,
        name='out_classifier'
    )(out)

    model = k.models.Model(inputs=[feature_screen, feature_minimap], output=out)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model
