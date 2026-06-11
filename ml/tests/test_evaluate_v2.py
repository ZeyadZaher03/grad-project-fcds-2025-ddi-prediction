# ml/tests/test_evaluate_v2.py
import numpy as np
from ml.pipeline.p06b_evaluate_v2 import combine_four_class


def test_combine_four_class_thresholds_none():
    p_interact = np.array([0.2, 0.9])
    sev = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    four = combine_four_class(p_interact, sev)
    assert four.shape == (2, 4)
    assert np.allclose(four.sum(1), 1.0)
    assert four[0].argmax() == 0
    assert four[1].argmax() == 3


def test_combine_four_class_rows_sum_to_one():
    four = combine_four_class(np.array([0.6]), np.array([[0.2, 0.5, 0.3]]))
    assert abs(four.sum() - 1.0) < 1e-6
