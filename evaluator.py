#!/usr/bin/python3
import argparse
import os

from encoder.sc2_dask_dataset import starcraft_dataset, starcraft_labels
from encoder.sc2_evaluator import evaluate_top_n_feature

BATCH_SIZE = 2048

if os.name == 'nt':
    VALIDATION_PATH = os.path.join("B:", "documents", "sc2_datasets", "wcs_global.zarr")
    OUT_PATH = os.path.join("B:", "documents", "sc2_discriminator-splitplayer.keras")
else:
    VALIDATION_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_datasets", "wcs_global.zarr")
    OUT_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_trained_model_shuffled.keras")


def evaluate(validation_path, model_path):
    print("Loading validation set %s..." % validation_path)
    validation_set = starcraft_dataset(validation_path, batch_size=BATCH_SIZE)

    print("Loading model %s..." % model_path)
    model = build_model(validation_set)
    model.load_weights(model_path)
    model.summary()

    y_pred, race_pred = model.predict(validation_set.all()[0], batch_size=BATCH_SIZE, verbose=1)
    print("Evaluating...")
    y_true = validation_set.y[:len(y_pred)]
    race_true = validation_set.race[:len(race_pred)]
    target_names = starcraft_labels()

    evaluate_top_n_feature(y_true, y_pred, race_true, race_pred, target_names)


if __name__ == '__main__':
    # some imports down here to prevent child processes from importing tensorflow
    import dask
    import tensorflow as tf
    import keras as k
    from keras.backend.tensorflow_backend import set_session

    from encoder.sc2_model import build_model

    dask.config.set(scheduler="threads")
    tf.logging.set_verbosity(tf.logging.WARN)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--validation-set", default=VALIDATION_PATH, type=str)
    args = parser.parse_args()

    evaluate(args.validation_set, args.model)
