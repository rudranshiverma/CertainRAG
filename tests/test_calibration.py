import warnings

import numpy as np
import pytest

from certainrag.calibration import apply_platt, best_threshold, fit_platt


def brute_force_threshold(scores, labels):
    best_bacc, best_t = -1.0, None
    for t in np.unique(scores):
        pred = scores >= t
        bacc = (pred[labels == 1].mean() + (~pred[labels == 0]).mean()) / 2
        if bacc > best_bacc + 1e-9:
            best_bacc, best_t = bacc, t
    return best_t


def test_fit_platt_recovers_parameters():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 3, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(0.7 * x - 1.2)))).astype(int)
    a, b = fit_platt(x, y)
    assert a == pytest.approx(0.7, abs=0.05)
    assert b == pytest.approx(-1.2, abs=0.05)


def test_apply_platt_is_stable_at_extremes():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert apply_platt([-1000.0, 1000.0], 1.0, 0.0).tolist() == [0.0, 1.0]
    assert apply_platt([0.0], 2.0, 0.0)[0] == pytest.approx(0.5)


def test_best_threshold_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(500):
        n = int(rng.integers(5, 60))
        scores = np.round(rng.random(n), int(rng.integers(1, 3)))
        labels = rng.integers(0, 2, n)
        if len(set(labels)) < 2:
            continue
        assert best_threshold(scores, labels) == brute_force_threshold(scores, labels)


def test_best_threshold_separable():
    assert best_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 0.8


@pytest.mark.parametrize(
    "scores, labels",
    [([0.1, 0.2], [1, 1]), ([0.1], [0, 1]), ([[0.1]], [[1]])],
)
def test_invalid_input_raises(scores, labels):
    with pytest.raises(ValueError):
        best_threshold(scores, labels)
