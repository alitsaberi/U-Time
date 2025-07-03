import logging
import os
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def get_hypnogram(y_pred, y_true=None, id_=None):
    def format_ax(ax, include_out_of_bounds=True):
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Sleep Stage")
        if include_out_of_bounds:
            ax.set_yticks(range(5))
            ax.set_yticklabels(["Wake", "REM", "N1", "N2", "N3"])
        else:
            ax.set_yticks(range(7))
            ax.set_yticklabels(["Wake", "REM", "N1", "N2", "N3", "Unknown", "Unusable", ])
        ax.invert_yaxis()
        ax.set_xlim(0, len(y_pred) / 120)
        l = ax.legend(loc=3)
        l.get_frame().set_linewidth(0)
    ids = np.arange(len(y_pred)) / 120  # Each period is 30 seconds, so divide by 120 to convert to hours
    fig = plt.figure(figsize=(15, 3))  # Decreased height to make y-labels closer
    ax1 = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1, top=0.85, bottom=0.15, right=0.85)  # Adjust layout to decrease left margin
    fig.suptitle("Hypnogram for Identifier: {}".format(id_ or "???"), fontsize=14, fontweight='bold')
    ax1.step(ids, y_pred, color="teal", label="Predicted", linewidth=3, alpha=0.9)
    if y_true is not None:
        ax1.step(ids, y_true, color="salmon", label="True", linewidth=2, alpha=0.7)
    format_ax(ax1, include_out_of_bounds=False)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.legend(fontsize=10, frameon=True, shadow=True, bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig, ax1


def plot_and_save_hypnogram(out_path, y_pred, y_true=None, id_=None):
    dir_ = os.path.split(out_path)[0]
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    outs = get_hypnogram(y_pred, y_true, id_)
    outs[0].savefig(out_path, dpi=180)
    plt.close(outs[0])


def plot_confusion_matrix(y_true, y_pred, n_classes,
                          normalize=False, id_=None,
                          cmap="Blues"):
    """
    Adapted from sklearn 'plot_confusion_matrix.py'.

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import confusion_matrix
    if normalize:
        title = 'Normalized Confusion Matrix for Identifier: {}'.format(id_ or "???")
    else:
        title = 'Confusion Matrix, Without Normalization for Identifier: {}'.format(id_ or "???")

    # Compute confusion matrix
    classes = np.arange(n_classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Get transformed labels
    from utime import Defaults
    labels = [Defaults.get_class_int_to_stage_string()[i] for i in classes]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(False)
    return fig, ax


def plot_and_save_cm(out_path, pred, true, n_classes, id_=None, normalized=True):
    dir_ = os.path.split(out_path)[0]
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    fig, ax = plot_confusion_matrix(true, pred, n_classes, normalized, id_)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
