#!/usr/bin/python3
import argparse
import os

import numpy as np

from encoder.sc2_evaluator import visualize_embedding
from encoder.sc2_dask_dataset import starcraft_dataset, starcraft_labels

BATCH_SIZE = 2048
if os.name == 'nt':
    VALIDATION_PATH = os.path.join("B:\\", "documents", "sc2_datasets", "wcs_global.zarr")
else:
    VALIDATION_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_datasets", "wcs_global.zarr")


def visualize(validation_path, model_path):
    print("Loading validation set %s..." % validation_path)
    validation_set = starcraft_dataset(validation_path, train_percentage=0.005, batch_size=BATCH_SIZE)

    print("Loading model %s..." % model_path)
    model = build_model(validation_set)
    beheaded_model = k.Model(inputs=model.inputs, outputs=model.get_layer("out_descriptor").output)

    feature_outputs = beheaded_model.predict(
        validation_set.all()[0],
        batch_size=BATCH_SIZE,
        verbose=1)

    y_true = np.argmax(validation_set.y[:len(feature_outputs)], axis=1)
    race_true = np.argmax(validation_set.race[:len(feature_outputs)], axis=1)

    visualize_embedding(feature_outputs, y_true, race_true)


if __name__ == '__main__':
    import keras as k
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    from encoder.sc2_model import build_model

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    parser = argparse.ArgumentParser()
    parser.add_argument("model", default=None, type=str)
    parser.add_argument("--validation-set", default=VALIDATION_PATH, type=str)
    args = parser.parse_args()

    visualize(args.validation_set, args.model)