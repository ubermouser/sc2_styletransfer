import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


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


def filter_empty_labels(y_true, y_pred, target_names):
    label_counts = np.bincount(y_true, minlength=len(target_names))
    label_mapping = np.zeros(len(target_names), dtype=np.int32)

    target_names = [name for idx, name in enumerate(target_names) if label_counts[idx] > 0]

    filter_mask = np.ones_like(y_true, dtype=np.bool)
    for idx, count in enumerate(label_counts):
        if count == 0:
            filter_mask[y_true == idx] = False

    label_mapping[label_counts > 0] = np.arange(len(target_names))
    y_true = label_mapping[y_true[filter_mask]]
    y_pred = label_mapping[y_pred[filter_mask]]

    return y_true, y_pred, target_names


def evaluate_feature(y_true, y_pred, target_names, filter_empty=True):
    if filter_empty:
        y_true, y_pred, target_names = filter_empty_labels(y_true, y_pred, target_names)

    # classification report:
    print(classification_report(
        y_true, y_pred, labels=np.arange(len(target_names)), target_names=target_names))

    # confusion matrix:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(target_names)))
    print(pd.DataFrame(cm, index=target_names, columns=target_names))

    plt.figure()
    plot_confusion_matrix(cm, target_names, normalize=True)
    plt.show()