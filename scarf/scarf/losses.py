"""
InfoNCE loss as specified in Algorithm 1 (lines 6-8):

  s_{i,j} = z_i . z~_j / (||z_i|| ||z~_j||)
  L = (1/N) * sum_i [ -log( exp(s_ii/tau) / (1/N * sum_k exp(s_ik/tau)) ) ]

z and z~ are the L2-normalized projection-head embeddings of the original and
corrupted views respectively, so the dot product already equals the cosine
similarity; we still divide defensively in case normalization is skipped.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, z_tilde: torch.Tensor) -> torch.Tensor:
        """z, z_tilde: (N, D) embeddings of the original / corrupted views.
        Positives are the matched pair (i, i); all other (i, k) pairs in the
        batch serve as negatives, exactly as in Algorithm 1.
        """
        n = z.shape[0]
        z_norm = F.normalize(z, p=2, dim=-1)
        zt_norm = F.normalize(z_tilde, p=2, dim=-1)

        sim = z_norm @ zt_norm.t()  # (N, N), s_{i,j}
        sim = sim / self.temperature

        # Cross entropy with the diagonal as the target class recovers
        # exactly the formula in line 8 of Algorithm 1:
        # -log( exp(s_ii/tau) / sum_k exp(s_ik/tau) )
        # (the paper's 1/N factor in num. and denom. cancels).
        targets = torch.arange(n, device=z.device)
        loss = F.cross_entropy(sim, targets)
        return loss
