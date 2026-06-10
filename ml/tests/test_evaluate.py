# ml/tests/test_evaluate.py
import numpy as np
import torch
from ml.pipeline.p06_evaluate import fit_temperature, expected_calibration_error


def test_temperature_softens_overconfident_but_inaccurate_predictions():
    # A model that predicts with near-certainty (logit magnitude 8) but is only ~55%
    # accurate is overconfident relative to its accuracy; temperature scaling should
    # raise T above 1 to soften the probabilities. (Confident AND correct would push
    # T below 1 — that is calibration working in the other direction.)
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 4, size=n)
    pred = y.copy()
    wrong = rng.random(n) < 0.45            # ~45% of predictions are wrong
    pred[wrong] = (y[wrong] + 1) % 4
    base = np.zeros((n, 4)); base[np.arange(n), pred] = 1.0
    logits = torch.tensor(base * 8.0, dtype=torch.float32)  # extreme confidence
    labels = torch.tensor(y, dtype=torch.long)
    T = fit_temperature(logits, labels)
    assert T > 1.0


def test_ece_is_zero_for_perfect_calibration():
    probs = np.eye(4)[np.array([0, 1, 2, 3])]
    y = np.array([0, 1, 2, 3])
    assert expected_calibration_error(probs, y) < 1e-6
