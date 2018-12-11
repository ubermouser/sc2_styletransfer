#!/usr/bin/python3
import os
import sys

import numpy as np
import keras as k
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from encoder.sc2_dataset import starcraft_dataset

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


def construct_identities(corpus_path, beheaded_model):

    corpus = starcraft_dataset(corpus_path)
    feature_outputs = beheaded_model.predict_generator(corpus, verbose=1)

    names = np.unique(corpus.y.data)
    identities = {
        name: np.average(feature_outputs[corpus.y.data[:] == name], axis=0)
        for name in names
    }
    return identities


def recognize(corpus_path, validation_path, model_path):
    model = k.models.load_model(model_path)
    beheaded_model = k.Model(inputs=model.inputs, outputs=model.get_layer("out_descriptor").output)

    corpus_identities = construct_identities(corpus_path, beheaded_model)
