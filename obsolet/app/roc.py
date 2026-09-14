# app/train/roc.py

import numpy as np
from sklearn.metrics import roc_curve

def compute_optimal_threshold(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores)

    # Youden’s J
    j = tpr - fpr
    idx = np.argmax(j)
    return thr[idx], fpr[idx], tpr[idx]
