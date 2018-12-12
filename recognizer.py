#!/usr/bin/python3
import os
import sys
import argparse

import numpy as np
import keras as k
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from encoder.sc2_model import build_model
from encoder.sc2_dataset import starcraft_dataset, starcraft_labels
from encoder.sc2_evaluator import evaluate_feature

tf.logging.set_verbosity(tf.logging.WARN)
USE_MULTIPROCESSING = True
if os.name == 'nt':
    DATASET_PATH = os.path.join("B:", "documents", "sc2_datasets", "wcs_montreal.h5py")
    VALIDATION_PATH = os.path.join("B:", "documents", "sc2_datasets", "wcs_global.h5py")
    OUT_PATH = os.path.join("B:", "documents", "sc2_trained_model_shuffled.keras")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
else:
    DATASET_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_datasets", "wcs_montreal.h5py")
    VALIDATION_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_datasets", "wcs_global.h5py")
    OUT_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_trained_model_shuffled.keras")


def construct_identities(corpus, beheaded_model, names=None):
    feature_outputs = beheaded_model.predict_generator(
        corpus,
        use_multiprocessing=USE_MULTIPROCESSING,
        workers=5,
        steps=20,
        verbose=1)

    print("Constructing identity map...")
    if names is None:
        names = np.unique(corpus.y.data[:len(feature_outputs)])
    identities = {
        name: np.average(feature_outputs[corpus.y.data[:len(feature_outputs)] == name], axis=0)
        for name in names
    }
    return identities


def compute_similarity(corpus_identities, beheaded_model, validation_set):
    #  num_identities X feature_size
    identities = np.asarray(list(corpus_identities.values()))
    #  num_rows X feature_size
    feature_outputs = beheaded_model.predict_generator(
        validation_set,
        use_multiprocessing=USE_MULTIPROCESSING,
        workers=5,
        verbose=1)

    cosine_similarities = np.dot(feature_outputs, identities.T) / (
        np.linalg.norm(feature_outputs, axis=1)[:, np.newaxis] *
        np.linalg.norm(identities, axis=1)[np.newaxis, :])

    # filter NaN:
    cosine_similarities[np.isnan(cosine_similarities)] = 0.0

    return cosine_similarities


def recognize(corpus_path, validation_path, model_path):
    print("Loading training set %s..." % corpus_path)
    corpus = starcraft_dataset(corpus_path)

    print("Loading validation set %s..." % validation_path)
    validation_set = starcraft_dataset(validation_path)

    print("Loading model %s..." % model_path)
    model = build_model(corpus)
    beheaded_model = k.Model(inputs=model.inputs, outputs=model.get_layer("out_descriptor").output)

    target_names = starcraft_labels()
    corpus_identities = construct_identities(corpus, beheaded_model, names=target_names)

    print("Computing similarity...")
    y_pred = compute_similarity(corpus_identities, beheaded_model, validation_set)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(validation_set.y[:len(y_pred)], axis=1)

    print("Evaluating...")
    evaluate_feature(y_true, y_pred, target_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", default=None, type=str)
    parser.add_argument("--corpus-set", default=DATASET_PATH, type=str)
    parser.add_argument("--validation-set", default=VALIDATION_PATH, type=str)
    args = parser.parse_args()

    recognize(args.corpus_set, args.validation_set, args.model)