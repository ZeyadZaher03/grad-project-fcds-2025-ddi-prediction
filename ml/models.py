# ml/models.py
"""Two 4-class severity models: deployable MLP and an inductive GraphSAGE."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

N_CLASSES = 4


class PairMLP4(nn.Module):
    """Inductive: consumes symmetric pair features -> 4-class logits."""
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class SageDDI(nn.Module):
    """Inductive GraphSAGE encoder + symmetric pair decoder -> 4-class logits."""
    def __init__(self, in_channels: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, N_CLASSES),
        )

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x

    def forward(self, x, edge_index, pair_index):
        z = self.encode(x, edge_index)
        a, b = pair_index
        # Symmetric combination so predict(A,B) == predict(B,A).
        h = torch.cat([z[a] + z[b], torch.abs(z[a] - z[b])], dim=-1)
        return self.head(h)
