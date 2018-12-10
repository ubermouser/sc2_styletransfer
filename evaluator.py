import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras as k
from keras.backend.tensorflow_backend import set_session
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from encoder.sc2_dataset import starcraft_dataset, starcraft_labels

if os.name == 'nt':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    VALIDATION_PATH = os.path.join("B:", "documents", "sc2_datasets", "wcs_montreal_0.h5py")
    OUT_PATH = os.path.join("B:", "documents", "sc2_trained_model.keras")
else:
    VALIDATION_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_datasets", "wcs_montreal_0.h5py")
    OUT_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_trained_model.keras")


def plot_confusion_matrix(
        cm,
        classes,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # example from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def evaluate(validation_path, model_path):
    print("Loading validation set %s..." % validation_path)
    validation_set = starcraft_dataset(validation_path, batch_size=2048)

    print("Loading model %s..." % model_path)
    model = k.models.load_model(model_path)

    y_pred = model.predict_generator(
        validation_set,
        use_multiprocessing=True,
        workers=5,
        verbose=1)
    print("Evaluating...")
    y_true = np.argmax(validation_set.y[:], axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    target_names = starcraft_labels()

    # classification report:
    print(classification_report(
        y_true, y_pred, labels=np.arange(len(target_names)), target_names=target_names))

    # confusion matrix:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(target_names)))
    print(pd.DataFrame(cm, index=target_names, columns=target_names))

    plt.figure()
    plot_confusion_matrix(cm, target_names, normalize=True)
    plt.show()


if __name__ == '__main__':
    evaluate(VALIDATION_PATH, OUT_PATH)
