# ml/tests/test_models_v2.py
import torch
from ml.models import MLPHead


def test_mlphead_binary_output():
    m = MLPHead(in_dim=20, n_out=1, hidden=16, dropout=0.1)
    out = m(torch.randn(4, 20))
    assert out.shape == (4, 1)


def test_mlphead_three_class_output():
    m = MLPHead(in_dim=20, n_out=3, hidden=16, dropout=0.1)
    out = m(torch.randn(5, 20))
    assert out.shape == (5, 3)
