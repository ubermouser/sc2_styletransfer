import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE


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


def visualize_embedding(feature_outputs, label, race):
    print("Constructing T-SNE embedding...")
    embedding = TSNE(n_components=2, n_iter=2000, verbose=1).fit_transform(feature_outputs)

    plt.figure()
    plot_embedding(embedding, label)
    plt.show()

    plt.figure()
    plot_embedding(embedding, race)
    plt.show()


def plot_embedding(embedding, label):
    plt.scatter(embedding[:, 0], embedding[:, 1], c=label)


def filter_empty_labels(y_true, y_pred, target_names):
    label_counts = np.bincount(y_true, minlength=len(target_names))
    num_nonzero_labels = np.count_nonzero(label_counts)

    filter_mask = np.ones_like(y_true, dtype=np.bool)
    for idx, count in enumerate(label_counts):
        if count == 0:
            filter_mask[y_true == idx] = False

    label_mapping = np.full(len(target_names), fill_value=num_nonzero_labels-1, dtype=np.int32)
    label_mapping[label_counts > 0] = np.arange(num_nonzero_labels)

    y_true = label_mapping[y_true[filter_mask]]
    y_pred = label_mapping[y_pred[filter_mask]]
    target_names = [name for idx, name in enumerate(target_names) if label_counts[idx] > 0]

    return y_true, y_pred, target_names


def filter_unknown_labels(y_true, y_pred, target_names):
    unknown_label_idx = np.argwhere([target_name == 'UNKNOWN' for target_name in target_names])
    if len(unknown_label_idx) == 0:
        return y_true, y_pred, target_names
    else:
        unknown_label_idx = unknown_label_idx[0]

    unknown_mask = y_pred != unknown_label_idx

    y_true = y_true[unknown_mask]
    y_pred = y_pred[unknown_mask]
    target_names = target_names[:-1]

    return y_true, y_pred, target_names


def top_n_accuracy(y_true, y_pred, top_n):
    if top_n == 1:
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

    y_pred = np.argsort(y_pred, axis=1)[:, -top_n:]
    top_n_match = np.any(np.equal(y_pred, np.expand_dims(y_true, axis=1)), axis=1)

    # take best match:
    y_pred = y_pred[:, -1]
    # overwrite when top-N is correct:
    y_pred[top_n_match] = y_true[top_n_match]
    return y_pred


def evaluate_feature(y_true, y_pred, target_names, filter_empty=True, filter_unknown=True):
    if filter_empty:
        y_true, y_pred, target_names = filter_empty_labels(y_true, y_pred, target_names)

    if filter_unknown:
        y_true, y_pred, target_names = filter_unknown_labels(y_true, y_pred, target_names)

    # classification report:
    print(classification_report(
        y_true, y_pred, labels=np.arange(len(target_names)), target_names=target_names))

    # confusion matrix:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(target_names)))
    print(pd.DataFrame(cm, index=target_names, columns=target_names))

    plt.figure()
    plot_confusion_matrix(cm, target_names, normalize=True)
    plt.show()


def evaluate_top_n_feature(y_true, y_pred, race_true, race_pred, target_names):
    y_true = np.argmax(y_true, axis=1)
    race_true = np.argmax(race_true, axis=1)

    y_pred = top_n_accuracy(y_true, y_pred, 3)
    race_pred = top_n_accuracy(race_true, race_pred, 1)

    evaluate_feature(y_true, y_pred, target_names, filter_empty=True, filter_unknown=True)
    evaluate_feature(race_true, race_pred, ['Terran', 'Zerg', 'Protoss'], filter_empty=False, filter_unknown=False)