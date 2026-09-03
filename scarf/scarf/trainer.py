"""
Pre-training (Algorithm 1) and fine-tuning loops, using the paper's default
hyperparameters: Adam (lr=1e-3), batch size 128, corruption rate c=0.6,
temperature tau=1, early stopping with patience 3 on a validation metric,
max 1000 pre-training epochs / 200 fine-tuning epochs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .corruption import MarginalSampler, scarf_corruption
from .losses import InfoNCELoss
from .model import ClassificationHead, Encoder, ProjectionHead


@dataclass
class TrainHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    best_epoch: int = 0


def _build_static_val_pairs(
    x_val: torch.Tensor,
    sampler: MarginalSampler,
    corruption_rate: float,
    n_epochs: int = 10,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cycle through validation data `n_epochs` times, generating corrupted
    views each pass, and freeze the resulting (x, x_tilde) pairs into a
    static set used for InfoNCE validation loss tracking throughout
    pre-training (as described in the paper).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    xs, xts = [], []
    for _ in range(n_epochs):
        xt = scarf_corruption(x_val, sampler, corruption_rate, generator=gen)
        xs.append(x_val.clone())
        xts.append(xt)
    return torch.cat(xs, dim=0), torch.cat(xts, dim=0)


def pretrain_scarf(
    x_train: np.ndarray,
    x_val: np.ndarray,
    input_dim: int,
    encoder: Optional[Encoder] = None,
    hidden_dim: int = 256,
    proj_dim: int = 256,
    encoder_layers: int = 4,
    head_layers: int = 2,
    batch_size: int = 128,
    lr: float = 1e-3,
    corruption_rate: float = 0.6,
    temperature: float = 1.0,
    max_epochs: int = 1000,
    patience: int = 3,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[Encoder, TrainHistory]:
    """Run SCARF contrastive pre-training (Algorithm 1) with early stopping
    on a static validation InfoNCE loss. Returns the trained encoder f.
    """
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)

    sampler = MarginalSampler(x_train_t)

    encoder = encoder or Encoder(input_dim, hidden_dim, encoder_layers)
    encoder = encoder.to(device)
    proj_head = ProjectionHead(encoder.output_dim, hidden_dim, proj_dim, head_layers).to(device)

    criterion = InfoNCELoss(temperature)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()), lr=lr)

    val_x, val_xt = _build_static_val_pairs(x_val_t, sampler, corruption_rate, n_epochs=10)

    n = x_train_t.shape[0]
    history = TrainHistory()
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        encoder.train()
        proj_head.train()
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            if batch_idx.shape[0] < 2:
                continue  # InfoNCE needs >= 2 examples
            xb = x_train_t[batch_idx]
            xtb = scarf_corruption(xb, sampler, corruption_rate)

            z = proj_head(encoder(xb))
            z_tilde = proj_head(encoder(xtb))
            loss = criterion(z, z_tilde)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.train_loss.append(train_loss)

        # Validation InfoNCE loss on the static pair set.
        encoder.eval()
        proj_head.eval()
        with torch.no_grad():
            val_losses = []
            for start in range(0, val_x.shape[0], batch_size):
                xb = val_x[start:start + batch_size]
                xtb = val_xt[start:start + batch_size]
                if xb.shape[0] < 2:
                    continue
                z = proj_head(encoder(xb))
                z_tilde = proj_head(encoder(xtb))
                val_losses.append(criterion(z, z_tilde).item())
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.val_loss.append(val_loss)

        if verbose:
            print(f"[pretrain] epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")

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


def finetune_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    encoder: Optional[Encoder] = None,
    input_dim: Optional[int] = None,
    hidden_dim: int = 256,
    head_layers: int = 2,
    encoder_layers: int = 4,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 3,
    label_smoothing: float = 0.0,
    mixup_alpha: Optional[float] = None,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[Encoder, ClassificationHead, TrainHistory]:
    """Supervised fine-tuning: attach a classification head h on top of
    (optionally pre-trained) encoder f and train both with cross-entropy,
    early-stopping on validation classification error.

    Set `encoder=None` to train from a randomly-initialized encoder (the
    "control" baseline). Set `mixup_alpha` to enable mixup, or
    `label_smoothing` > 0 to enable label smoothing.
    """
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)

    if encoder is None:
        assert input_dim is not None, "input_dim required when encoder is None"
        encoder = Encoder(input_dim, hidden_dim, encoder_layers)
    encoder = encoder.to(device)
    head = ClassificationHead(encoder.output_dim, hidden_dim, n_classes, head_layers).to(device)

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
        epoch_losses = []
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            xb = x_train_t[batch_idx]
            yb = y_train_t[batch_idx]

            if mixup_alpha is not None and mixup_alpha > 0 and xb.shape[0] > 1:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                shuffle = torch.randperm(xb.shape[0], device=device)
                xb_mixed = lam * xb + (1 - lam) * xb[shuffle]
                logits = head(encoder(xb_mixed))
                loss = lam * F.cross_entropy(logits, yb, label_smoothing=label_smoothing) + \
                    (1 - lam) * F.cross_entropy(logits, yb[shuffle], label_smoothing=label_smoothing)
            else:
                logits = head(encoder(xb))
                loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.train_loss.append(train_loss)

        encoder.eval()
        head.eval()
        with torch.no_grad():
            logits = head(encoder(x_val_t))
            preds = logits.argmax(dim=-1)
            val_err = (preds != y_val_t).float().mean().item()
        history.val_loss.append(val_err)

        if verbose:
            print(f"[finetune] epoch {epoch}: train_loss={train_loss:.4f} val_err={val_err:.4f}")

        if val_err < best_val_err - 1e-6:
            best_val_err = val_err
            best_state = {
                "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
            }
            history.best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        head.load_state_dict(best_state["head"])
    return encoder, head, history


@torch.no_grad()
def evaluate_accuracy(encoder: Encoder, head: ClassificationHead, x: np.ndarray, y: np.ndarray, device: str = "cpu") -> float:
    encoder.eval()
    head.eval()
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    logits = head(encoder(x_t))
    preds = logits.argmax(dim=-1)
    return (preds == y_t).float().mean().item()
