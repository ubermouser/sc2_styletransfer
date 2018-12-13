import numpy as np
import keras as k
import tensorflow as tf

from pysc2.lib import features

AUGMENTATION_SEED = 1
FEATURE_MINIMAP_PLAYER_RELATIVE = 5
FEATURE_SCREEN_PLAYER_RELATIVE = 5
FEATURE_SCREEN_UNIT_CATEGORY = 6


def build_minimap_input(dataset, regularizer, training=False):
    input = k.layers.Input(
        shape=dataset.feature_minimap.shape[1:],
        dtype=dataset.feature_minimap.dtype,
        name='feature_minimap')

    minimap = k.layers.Lambda(lambda x: x[:, FEATURE_MINIMAP_PLAYER_RELATIVE, :, :])(input)
    embedding = k.layers.Embedding(
        len(features.PlayerRelative), 2,
        embeddings_regularizer=regularizer,
        name="embedding_minimap_player"
    )(minimap)

    return input, embedding


def build_screen_input(dataset, regularizer, training=False):
    input = k.layers.Input(
        shape=dataset.feature_screen.shape[1:],
        dtype=dataset.feature_screen.dtype,
        name='feature_screen')

    screen_units = k.layers.Lambda(lambda x: x[:, FEATURE_SCREEN_UNIT_CATEGORY, :, :])(input)
    embedding_units = k.layers.Embedding(
        len(features.static_data.UNIT_TYPES) + 1, 6,
        embeddings_regularizer=regularizer,
        name="embedding_screen_units"
    )(screen_units)

    screen_player = k.layers.Lambda(lambda x: x[:, FEATURE_SCREEN_PLAYER_RELATIVE, :, :])(input)
    embedding_player = k.layers.Embedding(
        len(features.PlayerRelative), 2,
        embeddings_regularizer=regularizer,
        name="embedding_screen_player"
    )(screen_player)

    return input, k.layers.Concatenate()([embedding_player, embedding_units])


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


def build_model(dataset, training=False):
    regularizer = k.regularizers.l2(l=1.0e-5)
    optimizer = k.optimizers.Adam(lr=1.0e-4)
    metrics = [k.metrics.categorical_accuracy]

    # minimap channel:
    # reduce # of channels in the input
    feature_minimap, minimap_input = build_minimap_input(dataset, regularizer, training)
    # collapse spatial dimensions
    out_1 = chain_conv_2d([4, 6, 8, 10], minimap_input, regularizer)

    # screen channel:
    # reduce # of channels in the input
    feature_screen, screen_input = build_screen_input(dataset, regularizer, training)
    # collapse spatial dimensions
    out_2 = chain_conv_2d([16, 18, 20, 22], screen_input, regularizer)

    # higher-level decision-making:
    out_1 = k.layers.Flatten()(out_1)
    out_2 = k.layers.Flatten()(out_2)
    out = k.layers.Concatenate()([out_1, out_2])
    out = k.layers.Dense(
        50,
        activation='relu',
        kernel_regularizer=regularizer,
        name='out_descriptor'
    )(out)
    out_name = k.layers.Dense(
        dataset.y.shape[-1],
        activation='softmax',
        kernel_regularizer=regularizer,
        name='out_name'
    )(out)
    out_race = k.layers.Dense(
        dataset.race.shape[-1],
        activation='softmax',
        kernel_regularizer=regularizer,
        name='out_race'
    )(out)

    model = k.models.Model(inputs=[
        feature_screen,
        feature_minimap
    ], outputs=[
        out_name,
        out_race
    ])
    model.compile(
        loss={
            'out_name': 'categorical_crossentropy',
            'out_race': 'categorical_crossentropy'
        },
        loss_weights={
            'out_name': 0.6,
            'out_race': 0.4
        },
        optimizer=optimizer,
        metrics=metrics)
    return model
