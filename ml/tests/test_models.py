# ml/tests/test_models.py
import torch
from ml.models import PairMLP4, SageDDI


def test_pairmlp_output_shape_is_4_classes():
    m = PairMLP4(in_dim=30, hidden=16, dropout=0.1)
    out = m(torch.randn(5, 30))
    assert out.shape == (5, 4)


def test_sage_output_shape_is_4_classes():
    m = SageDDI(in_channels=12, hidden=16)
    x = torch.randn(6, 12)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    pair_index = torch.tensor([[0, 2], [3, 5]], dtype=torch.long)  # 2 pairs
    out = m(x, edge_index, pair_index)
    assert out.shape == (2, 4)
