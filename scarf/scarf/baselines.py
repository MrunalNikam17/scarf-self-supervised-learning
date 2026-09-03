"""
Baseline pre-training methods used as ablations/comparisons in the paper:
  - Autoencoders: no-noise AE, additive-Gaussian-noise AE, SCARF-corruption AE
  - Discriminative SCARF: binary logistic pretext task (real vs. corrupted)

All follow the same recipe as SCARF pre-training: Adam, batch 128, early
stopping with patience 3.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .corruption import MarginalSampler, scarf_corruption
from .model import Encoder, _mlp
from .trainer import TrainHistory


class AutoEncoder(nn.Module):
    """Decoder g reconstructs the (uncorrupted) input from the encoder's
    representation of a (possibly corrupted) input. Uses the same 2-layer
    MLP architecture as the SCARF projection head, but outputs input_dim
    values and is trained with MSE reconstruction loss.
    """

    def __init__(self, hidden_dim: int, input_dim: int, n_layers: int = 2):
        super().__init__()
        self.net = _mlp(hidden_dim, hidden_dim, input_dim, n_layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def _make_noisy(x: torch.Tensor, noise_type: str, sampler: Optional[MarginalSampler], corruption_rate: float, noise_std: float) -> torch.Tensor:
    if noise_type == "none":
        return x
    elif noise_type == "gaussian":
        return x + torch.randn_like(x) * noise_std
    elif noise_type == "scarf":
        assert sampler is not None
        return scarf_corruption(x, sampler, corruption_rate)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")


def pretrain_autoencoder(
    x_train: np.ndarray,
    x_val: np.ndarray,
    input_dim: int,
    noise_type: str = "none",  # "none" | "gaussian" | "scarf"
    noise_std: float = 0.5,
    corruption_rate: float = 0.6,
    hidden_dim: int = 256,
    encoder_layers: int = 4,
    decoder_layers: int = 2,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_epochs: int = 1000,
    patience: int = 3,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[Encoder, TrainHistory]:
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)

    sampler = MarginalSampler(x_train_t) if noise_type == "scarf" else None

    encoder = Encoder(input_dim, hidden_dim, encoder_layers).to(device)
    decoder = AutoEncoder(encoder.output_dim, input_dim, decoder_layers).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    n = x_train_t.shape[0]
    history = TrainHistory()
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        encoder.train()
        decoder.train()
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = x_train_t[idx]
            xb_noisy = _make_noisy(xb, noise_type, sampler, corruption_rate, noise_std)
            recon = decoder(encoder(xb_noisy))
            loss = F.mse_loss(recon, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))
        history.train_loss.append(train_loss)

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            xb_noisy = _make_noisy(x_val_t, noise_type, sampler, corruption_rate, noise_std)
            recon = decoder(encoder(xb_noisy))
            val_loss = F.mse_loss(recon, x_val_t).item()
        history.val_loss.append(val_loss)

        if verbose:
            print(f"[AE-{noise_type}] epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
            history.best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        encoder.load_state_dict(best_state)
    return encoder, history


class _DiscriminativeHead(nn.Module):
    """2-layer MLP + final linear projection to a single logit, used to
    discriminate original (label=1) vs. SCARF-corrupted (label=0) inputs.
    """

    def __init__(self, hidden_dim: int, mid_layers: int = 2):
        super().__init__()
        self.trunk = _mlp(hidden_dim, hidden_dim, hidden_dim, mid_layers)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.trunk(h))).squeeze(-1)


def pretrain_scarf_discriminative(
    x_train: np.ndarray,
    x_val: np.ndarray,
    input_dim: int,
    corruption_rate: float = 0.6,
    hidden_dim: int = 256,
    encoder_layers: int = 4,
    head_layers: int = 2,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_epochs: int = 1000,
    patience: int = 3,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[Encoder, TrainHistory]:
    """Discriminative SCARF: instead of InfoNCE, train the head to
    distinguish original examples from SCARF-corrupted ones via a binary
    logistic loss. Early stopping uses classification error (as in the
    paper), not the logistic loss itself.
    """
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    sampler = MarginalSampler(x_train_t)

    encoder = Encoder(input_dim, hidden_dim, encoder_layers).to(device)
    head = _DiscriminativeHead(encoder.output_dim, head_layers).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)

    n = x_train_t.shape[0]
    history = TrainHistory()
    best_val_err = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        encoder.train()
        head.train()
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = x_train_t[idx]
            xtb = scarf_corruption(xb, sampler, corruption_rate)

            x_cat = torch.cat([xb, xtb], dim=0)
            y_cat = torch.cat([torch.ones(xb.shape[0]), torch.zeros(xtb.shape[0])]).to(device)

            logits = head(encoder(x_cat))
            loss = F.binary_cross_entropy_with_logits(logits, y_cat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))
        history.train_loss.append(train_loss)

        encoder.eval()
        head.eval()
        with torch.no_grad():
            xtb_val = scarf_corruption(x_val_t, sampler, corruption_rate)
            x_cat = torch.cat([x_val_t, xtb_val], dim=0)
            y_cat = torch.cat([torch.ones(x_val_t.shape[0]), torch.zeros(xtb_val.shape[0])]).to(device)
            logits = head(encoder(x_cat))
            preds = (torch.sigmoid(logits) > 0.5).float()
            val_err = (preds != y_cat).float().mean().item()
        history.val_loss.append(val_err)

        if verbose:
            print(f"[SCARF-disc] epoch {epoch}: train_loss={train_loss:.4f} val_err={val_err:.4f}")

        if val_err < best_val_err - 1e-6:
            best_val_err = val_err
            best_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
            history.best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        encoder.load_state_dict(best_state)
    return encoder, history
