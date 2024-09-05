"""Training utilities"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns


def plot_conf_mtx(y_true, y_pred, labels_enc, labels) -> np.ndarray:
    """Plot confusion matrix with accuracy"""
    accuracy = accuracy_score(y_true, y_pred)
    cmtx = confusion_matrix(
        y_true,
        y_pred,
        labels=labels_enc,
    )

    plt.figure(figsize=(10, 7))
    sns.heatmap(cmtx, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predictions')
    plt.ylabel('True classes')
    plt.suptitle('Confusion matrix')
    plt.title(f"Accuracy : {accuracy :0.2f}")
    plt.show()

    return cmtx


if __name__ == "__main__":
    help()
