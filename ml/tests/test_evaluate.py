# ml/tests/test_evaluate.py
import numpy as np
import torch
from ml.pipeline.p06_evaluate import fit_temperature, expected_calibration_error


def test_temperature_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=400)
    base = np.zeros((400, 4)); base[np.arange(400), y] = 1.0
    logits = torch.tensor(base * 8.0, dtype=torch.float32)  # very confident
    labels = torch.tensor(y, dtype=torch.long)
    T = fit_temperature(logits, labels)
    assert T > 1.0


def test_ece_is_zero_for_perfect_calibration():
    probs = np.eye(4)[np.array([0, 1, 2, 3])]
    y = np.array([0, 1, 2, 3])
    assert expected_calibration_error(probs, y) < 1e-6
