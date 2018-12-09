#!/usr/bin/ python3
import os
import gzip
import sys

import numpy as np
import keras as k
import tensorflow as tf
import tqdm
from keras.backend.tensorflow_backend import set_session

from sc2_dataset import starcraft_dataset

if os.name == 'nt':
    DATASET_PATH = os.path.join("B:", "downloads", "test_output.hdf5")
    set_session(tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})))
else:
    DATASET_PATH = os.path.join("/media", "sf_B_DRIVE", "downloads", "test_output.hdf5")

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


def discriminate(dataset_path, validation_split=0.10):
    dataset = starcraft_dataset(dataset_path, batch_size=2048)

    model = build_model(input_shape=dataset.x.shape[1:], output_shape=dataset.y.shape[1:])
    model.summary()
    try:
        model.fit_generator(dataset, shuffle=True, epochs=20, use_multiprocessing=False, workers=2)
    except KeyboardInterrupt:
        pass
    print("Saving model...")
    #model.save("trained_model.keras")


def main(sys_args=sys.argv):
    discriminate(DATASET_PATH)


if __name__ == '__main__':
    main()
