import numpy as np
import keras as k
import tensorflow as tf

from pysc2.lib import features


FEATURE_MINIMAP_PLAYER_RELATIVE = 5


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


def build_model(dataset):
    feature_minimap, minimap_input = build_minimap_input(dataset)

    regularizer = k.regularizers.l2(l=1e-4)
    optimizer = k.optimizers.Adam(lr=1e-3)
    metrics = [k.metrics.categorical_accuracy]
    # normalize the input per channel
    # out = k.layers.BatchNormalization(
    #     axis=3,
    #     momentum=0.975
    # )(input)
    # reduce # of channels in the input
    out = k.layers.Conv2D(
        filters=2,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation='relu',
        kernel_regularizer=regularizer
    )(minimap_input)
    # collapse spatial dimensions
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
    # higher-level decision-making:
    out = k.layers.Flatten()(out)
    out = k.layers.Dense(
        72,
        activation='relu',
        kernel_regularizer=regularizer,
        name='out_descriptor'
    )(out)
    out = k.layers.Dense(
        dataset.y.shape[-1],
        activation='softmax',
        kernel_regularizer=regularizer,
        name='out_classifier'
    )(out)

    model = k.models.Model(inputs=[feature_minimap], output=out)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model