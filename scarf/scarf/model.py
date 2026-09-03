"""
Model components matching the paper's architecture:
  - Encoder f: 4-layer ReLU MLP, hidden dim 256.
  - Pre-training head g: 2-layer ReLU MLP, hidden dim 256, L2-normalized output.
  - Classification head h: 2-layer ReLU MLP, hidden dim 256.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout_rate: float = 0.0) -> nn.Sequential:
    """An n_layers ReLU MLP. n_layers counts the number of Linear layers, so
    n_layers=4 means 3 hidden ReLU layers + 1 output linear layer (no final
    activation), matching common encoder depth conventions used in the paper.
    Optional dropout (default 0.0) applied after each hidden ReLU.
    """
    assert n_layers >= 1
    layers = []
    d_in = in_dim
    for i in range(n_layers - 1):
        layers.append(nn.Linear(d_in, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        d_in = hidden_dim
    layers.append(nn.Linear(d_in, out_dim))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """f: input features -> representation. 4-layer ReLU MLP, hidden dim 256."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_layers: int = 4, dropout_rate: float = 0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.net = _mlp(input_dim, hidden_dim, hidden_dim, n_layers, dropout_rate=dropout_rate)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(nn.Module):
    """g: representation -> embedding used for the contrastive loss.
    2-layer ReLU MLP, hidden dim 256, L2-normalized output (unit hypersphere).
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, output_dim: int = 256, n_layers: int = 2):
        super().__init__()
        self.net = _mlp(input_dim, hidden_dim, output_dim, n_layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.net(h)
        return F.normalize(z, p=2, dim=-1)


class ClassificationHead(nn.Module):
    """h: representation -> class logits. 2-layer ReLU MLP, hidden dim 256."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, n_classes: int = 2, n_layers: int = 2, dropout_rate: float = 0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.net = _mlp(input_dim, hidden_dim, n_classes, n_layers, dropout_rate=dropout_rate)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SCARFModel(nn.Module):
    """Convenience wrapper bundling encoder f with a pre-training head g
    (used during contrastive pre-training) or a classification head h (used
    during fine-tuning). Only one of `g` / `h` is active at a time.
    """

    def __init__(self, encoder: Encoder, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
