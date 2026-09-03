"""
SCARF corruption: for a batch of examples, sample a random subset of feature
indices (size q = floor(c * M)) per example and replace those entries with a
draw from that feature's empirical marginal distribution (i.e. a value drawn
uniformly from the values that feature takes across the training set).

This mirrors Algorithm 1, lines 2-5 of the SCARF paper.
"""
from __future__ import annotations

import torch


class MarginalSampler:
    """Holds, for every feature, the pool of values seen in the training data
    so that we can draw i.i.d. samples from each feature's empirical marginal.
    """

    def __init__(self, train_features: torch.Tensor):
        # train_features: (N, M) float tensor -- used only to build per-feature pools.
        if train_features.ndim != 2:
            raise ValueError("train_features must be a 2D (N, M) tensor")
        self.feature_pool = train_features.detach().clone()  # (N, M)
        self.n, self.m = self.feature_pool.shape

    def sample(self, num_rows: int, num_cols: int, device=None) -> torch.Tensor:
        """Draw a (num_rows, num_cols) tensor where entry (i, j) is a random
        draw from feature j's empirical marginal (uniform over observed values).
        """
        device = device or self.feature_pool.device
        idx = torch.randint(0, self.n, (num_rows, num_cols), device=self.feature_pool.device)
        cols = torch.arange(num_cols, device=self.feature_pool.device).unsqueeze(0).expand(num_rows, -1)
        drawn = self.feature_pool[idx, cols]
        return drawn.to(device)


def scarf_corruption(
    x: torch.Tensor,
    sampler: MarginalSampler,
    corruption_rate: float = 0.6,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Produce a corrupted view x_tilde of a batch x (N, M).

    For each row, sample q = floor(corruption_rate * M) feature indices
    uniformly without replacement and replace those entries with draws from
    the respective feature's empirical marginal distribution.
    """
    n, m = x.shape
    q = int(corruption_rate * m)
    q = max(1, min(q, m))  # always corrupt at least 1 feature, never more than all

    # For each row, pick q indices out of m without replacement.
    # Efficient vectorized approach: generate random scores per (row, feature)
    # and take the top-q smallest as the corrupted indices.
    scores = torch.rand(n, m, generator=generator, device=x.device)
    _, corrupt_idx = torch.topk(scores, q, dim=1, largest=False)  # (n, q)

    mask = torch.zeros(n, m, dtype=torch.bool, device=x.device)
    mask.scatter_(1, corrupt_idx, True)

    replacements = sampler.sample(n, m, device=x.device)
    x_tilde = torch.where(mask, replacements, x)
    return x_tilde
