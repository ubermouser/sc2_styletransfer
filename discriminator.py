#!/usr/bin/ python3
import os
import sys

import keras as k
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from encoder.sc2_dataset import starcraft_dataset

if os.name == 'nt':
    DATASET_PATH = os.path.join("B:", "documents", "sc2_datasets", "wcs_global.h5py")
    VALIDATION_PATH = os.path.join("B:", "documents", "sc2_datasets", "wcs_montreal_0.h5py")
    OUT_PATH = os.path.join("B:", "documents", "sc2_trained_model.keras")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
else:
    DATASET_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_datasets", "wcs_global.h5py")
    VALIDATION_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_datasets", "wcs_montreal_0.h5py")
    OUT_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_trained_model.keras")

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


def discriminate(dataset_path, validation_path, out_path=None):
    train_set = starcraft_dataset(dataset_path, batch_size=1024)
    validation_set = starcraft_dataset(validation_path, batch_size=1024)

    model = build_model(input_shape=train_set.x.shape[1:], output_shape=train_set.y.shape[1:])
    model.summary()
    try:
        model.fit_generator(
            train_set,
            # steps_per_epoch=10,
            validation_data=validation_set,
            # validation_steps=10,
            shuffle=True,
            epochs=20,
            use_multiprocessing=True,
            workers=4)
    except KeyboardInterrupt:
        pass
    if out_path is not None:
        print("Saving model...")
        model.save("trained_model.keras")


def main(sys_args=sys.argv):
    discriminate(DATASET_PATH, VALIDATION_PATH, OUT_PATH)


if __name__ == '__main__':
    main()
