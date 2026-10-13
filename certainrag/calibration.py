import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


def _check(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("scores and labels must be 1-D and the same length.")
    if not set(np.unique(labels)) == {0, 1}:
        raise ValueError("labels must contain both 0 and 1.")
    return scores, labels


def fit_platt(logits, labels):
    logits, labels = _check(logits, labels)
    model = LogisticRegression(C=1e6).fit(logits.reshape(-1, 1), labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def apply_platt(logits, a, b):
    return expit(a * np.asarray(logits, dtype=float) + b)


def best_threshold(scores, labels):
    scores, labels = _check(scores, labels)
    pos = np.sort(scores[labels == 1])
    neg = np.sort(scores[labels == 0])
    candidates = np.unique(scores)
    tpr = 1.0 - np.searchsorted(pos, candidates, side="left") / len(pos)
    tnr = np.searchsorted(neg, candidates, side="left") / len(neg)
    # rounding makes ties resolve to the lowest threshold regardless of float noise
    return float(candidates[np.argmax(np.round((tpr + tnr) / 2, 12))])
