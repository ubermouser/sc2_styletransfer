#!/usr/bin/python3
import argparse
import os
import sys

import keras as k
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from encoder.sc2_model import build_model
from encoder.sc2_dataset import starcraft_dataset

tf.logging.set_verbosity(tf.logging.WARN)
USE_MULTIPROCESSING = True
if os.name == 'nt':
    DATASET_PATH = os.path.join("B:", "documents", "sc2_datasets", "wcs_montreal.h5py")
    VALIDATION_PATH = os.path.join("B:", "documents", "sc2_datasets", "wcs_global.h5py")
    OUT_PATH = os.path.join("B:", "documents", "sc2_discriminator-splitplayer.keras")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
else:
    DATASET_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_datasets", "wcs_montreal.h5py")
    VALIDATION_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_datasets", "wcs_global.h5py")
    OUT_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_trained_model_shuffled.keras")


def discriminate(dataset_path, validation_path=None, out_path=None):
    print("Loading training set %s..." % dataset_path)
    train_set = starcraft_dataset(dataset_path, batch_size=128)

    if validation_path is not None:
        print("Loading validation set %s..." % validation_path)
        validation_set = starcraft_dataset(validation_path, batch_size=128)
    else:
        validation_set = None

    model = build_model(train_set)
    model.summary()
    try:
        model.fit_generator(
            train_set,
            validation_data=validation_set,
            shuffle=True,
            epochs=8,
            use_multiprocessing=USE_MULTIPROCESSING,
            workers=5)
    except KeyboardInterrupt:
        pass
    if out_path is not None:
        print("Saving model to %s..." % out_path)
        model.save_weights(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-set", default=DATASET_PATH, type=str)
    parser.add_argument("--out-path", default=None, type=str)
    parser.add_argument("--validation-set", default=None, type=str)
    args = parser.parse_args()

    discriminate(args.training_set, args.validation_set, args.out_path)
