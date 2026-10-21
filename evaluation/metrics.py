import numpy as np
from sklearn.metrics import roc_auc_score

from certainrag.calibration import best_threshold

# labels are 1 for supported/relevant, scores are higher-is-better


def auroc(labels, scores):
    return float(roc_auc_score(labels, scores))


def balanced_accuracy(labels, scores, threshold, strict=False):
    labels, scores = np.asarray(labels), np.asarray(scores)
    pred = scores > threshold if strict else scores >= threshold
    return float((pred[labels == 1].mean() + (~pred[labels == 0]).mean()) / 2)


def unsupported_prf(labels, scores, threshold):
    # positive class is the hallucinated/unsupported answer, as reported on RAGTruth
    actual = np.asarray(labels) == 0
    flagged = np.asarray(scores) < threshold
    tp = float((flagged & actual).sum())
    precision = tp / flagged.sum() if flagged.sum() else 0.0
    recall = tp / actual.sum() if actual.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def bootstrap_ci(metric, labels, scores, n=1000, seed=0):
    labels, scores = np.asarray(labels), np.asarray(scores)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() == labels[idx].max():
            continue
        values.append(metric(labels[idx], scores[idx]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def ece(labels, probs, bins=15):
    labels, probs = np.asarray(labels, dtype=float), np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(total)


def risk_coverage(labels, scores):
    # accept the most-supported answers first; risk is the unsupported rate among accepted
    order = np.argsort(-np.asarray(scores), kind="stable")
    unsupported = (np.asarray(labels)[order] == 0).astype(float)
    accepted = np.arange(1, len(order) + 1)
    risk = np.cumsum(unsupported) / accepted
    return accepted / len(order), risk, float(risk.mean())


def tuned_threshold(dev_labels, dev_scores):
    return best_threshold(dev_scores, dev_labels)


def percentiles(values):
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, 50)), float(np.percentile(values, 95))
