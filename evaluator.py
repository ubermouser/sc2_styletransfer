#!/usr/bin/python3
import argparse
import itertools
import os

import numpy as np
import tensorflow as tf
import keras as k
from keras.backend.tensorflow_backend import set_session

from encoder.sc2_model import build_model
from encoder.sc2_dataset import starcraft_dataset, starcraft_labels
from encoder.sc2_evaluator import evaluate_feature

if os.name == 'nt':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    VALIDATION_PATH = os.path.join("B:", "documents", "sc2_datasets", "wcs_global.h5py")
    OUT_PATH = os.path.join("B:", "documents", "sc2_discriminator-splitplayer.keras")
else:
    VALIDATION_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_datasets", "wcs_global.h5py")
    OUT_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_trained_model_shuffled.keras")


def evaluate(validation_path, model_path):
    print("Loading validation set %s..." % validation_path)
    validation_set = starcraft_dataset(validation_path, batch_size=128)

    print("Loading model %s..." % model_path)
    model = build_model(validation_set)
    model.load_weights(model_path)
    model.summary()

    y_pred = model.predict_generator(
        validation_set,
        use_multiprocessing=True,
        workers=5,
        verbose=1)
    print("Evaluating...")
    y_true = np.argmax(validation_set.y[:len(y_pred)], axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    target_names = starcraft_labels()

    evaluate_feature(y_true, y_pred, target_names, filter_empty=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--validation-set", default=VALIDATION_PATH, type=str)
    args = parser.parse_args()

    evaluate(args.validation_set, args.model)
