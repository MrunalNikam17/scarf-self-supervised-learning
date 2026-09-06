"""
Ablation experiments reproducing Section 4.4 and Appendix B of the SCARF paper:
  - Corruption strategies: marginal, none, mean, additive Gaussian noise, joint sampling,
    missing-feature (learnable missing embedding), feature dropout.
  - Hyperparameter sweeps: batch sizes, corruption rates, softmax temperatures.
  - Alternative losses: Barlow Twins (Zbontar et al. 2021), Alignment + Uniformity (Wang & Isola 2020).
  - Validation metrics: InfoNCE loss vs. InfoNCE error (off-diagonal argmax).
  - Pre-training vs. Co-training: supervised cross-entropy + lambda * L_cont.
  - Pre-training vs. Data Augmentation: online SCARF corruption during supervised training.
"""
from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .corruption import MarginalSampler, scarf_corruption
from .losses import InfoNCELoss
from .model import ClassificationHead, Encoder, ProjectionHead
from .trainer import TrainHistory, finetune_classifier


# ---------------------------------------------------------------------------
# Corruption Strategies (Section 4.4 & Appendix B)
# ---------------------------------------------------------------------------

def corrupt_view(
    x: torch.Tensor,
    strategy: str,
    sampler: Optional[MarginalSampler] = None,
    marginal_mean: Optional[torch.Tensor] = None,
    train_pool: Optional[torch.Tensor] = None,
    missing_embedding: Optional[nn.Parameter] = None,
    corruption_rate: float = 0.6,
    noise_std: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply the chosen corruption strategy to generate corrupted view x_tilde.

    Strategies:
      - "marginal": Algorithm 1 marginal sampling (SCARF default).
      - "none": no corruption (x_tilde = x).
      - "mean": replace selected features with empirical marginal mean.
      - "gaussian": add i.i.d. N(0, noise_std^2) to features.
      - "joint": replace selected features with draws from empirical joint distribution.
      - "missing_feature": replace selected features with learnable missing embeddings.
      - "feature_dropout": zero-out selected features.
    """
    if strategy == "none":
        return x.clone()

    if strategy == "marginal":
        assert sampler is not None, "MarginalSampler required for marginal corruption"
        return scarf_corruption(x, sampler, corruption_rate, generator=generator)

    n, m = x.shape
    q = max(1, min(int(corruption_rate * m), m))

    # Vectorized random selection of q feature indices per row
    scores = torch.rand(n, m, generator=generator, device=x.device)
    _, corrupt_idx = torch.topk(scores, q, dim=1, largest=False)
    mask = torch.zeros(n, m, dtype=torch.bool, device=x.device)
    mask.scatter_(1, corrupt_idx, True)

    if strategy == "mean":
        assert marginal_mean is not None
        replacements = marginal_mean.unsqueeze(0).expand(n, -1).to(x.device)
        return torch.where(mask, replacements, x)

    elif strategy == "gaussian":
        # Additive Gaussian noise to selected features
        noise = torch.randn(n, m, generator=generator, device=x.device) * noise_std
        return torch.where(mask, x + noise, x)

    elif strategy == "joint":
        assert train_pool is not None
        # Draw entire rows from train_pool (joint empirical distribution)
        row_indices = torch.randint(0, train_pool.shape[0], (n,), device=train_pool.device)
        replacements = train_pool[row_indices].to(x.device)
        return torch.where(mask, replacements, x)

    elif strategy == "missing_feature":
        assert missing_embedding is not None
        replacements = missing_embedding.unsqueeze(0).expand(n, -1).to(x.device)
        return torch.where(mask, replacements, x)

    elif strategy == "feature_dropout":
        zeros = torch.zeros_like(x)
        return torch.where(mask, zeros, x)

    else:
        raise ValueError(f"Unknown corruption strategy: {strategy}")


# ---------------------------------------------------------------------------
# Alternative Contrastive Losses (Appendix B)
# ---------------------------------------------------------------------------

class AlignmentUniformityLoss(nn.Module):
    """Alignment and Uniformity loss (Wang & Isola, 2020) with equal weighting:
      L = L_align + L_uniform
      L_align = E[||z - z_tilde||^2]
      L_uniform = log E_{i!=j}[exp(-t ||z_i - z_j||^2)], with t=2.
    """

    def __init__(self, t: float = 2.0):
        super().__init__()
        self.t = t

    def forward(self, z: torch.Tensor, z_tilde: torch.Tensor) -> torch.Tensor:
        z_norm = F.normalize(z, p=2, dim=-1)
        zt_norm = F.normalize(z_tilde, p=2, dim=-1)

        # Alignment
        loss_align = (z_norm - zt_norm).pow(2).sum(dim=-1).mean()

        # Uniformity
        n = z_norm.shape[0]
        if n < 2:
            return loss_align

        diff = z_norm.unsqueeze(1) - z_norm.unsqueeze(0)  # (N, N, D)
        dist_sq = diff.pow(2).sum(dim=-1)
        mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
        loss_uniform = torch.log(torch.exp(-self.t * dist_sq[mask]).mean().clamp(min=1e-8))

        return loss_align + loss_uniform


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins loss (Zbontar et al., 2021):
      L = sum_i (1 - C_ii)^2 + lambd * sum_i sum_{j!=i} C_ij^2
      where C = (1/N) * Z^T * Z_tilde is the cross-correlation matrix.
      Paper hyperparameter: lambd = 5e-3.
    """

    def __init__(self, lambd: float = 5e-3):
        super().__init__()
        self.lambd = lambd

    def forward(self, z: torch.Tensor, z_tilde: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        if n < 2:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # Normalize representations along batch dimension
        z_norm = (z - z.mean(dim=0)) / (z.std(dim=0).clamp(min=1e-6))
        zt_norm = (z_tilde - z_tilde.mean(dim=0)) / (z_tilde.std(dim=0).clamp(min=1e-6))

        c = (z_norm.t() @ zt_norm) / n
        on_diag = torch.diagonal(c).add(-1.0).pow(2).sum()
        off_diag = c.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten().pow(2).sum()
        return on_diag + self.lambd * off_diag


def compute_infonce_error(z: torch.Tensor, z_tilde: torch.Tensor, temperature: float = 1.0) -> float:
    """InfoNCE classification error (off-diagonal argmax):
    Fraction of samples where the highest similarity is not the matched positive pair.
    """
    z_norm = F.normalize(z, p=2, dim=-1)
    zt_norm = F.normalize(z_tilde, p=2, dim=-1)
    sim = (z_norm @ zt_norm.t()) / temperature  # (N, N)
    preds = sim.argmax(dim=1)
    targets = torch.arange(z.shape[0], device=z.device)
    return (preds != targets).float().mean().item()


# ---------------------------------------------------------------------------
# Generalized Pre-training with Ablation Parameters
# ---------------------------------------------------------------------------

def pretrain_ablation(
    x_train: np.ndarray,
    x_val: np.ndarray,
    input_dim: int,
    corruption_strategy: str = "marginal",
    loss_type: str = "infonce",  # "infonce", "barlow", "align_uniform"
    early_stop_metric: str = "loss",  # "loss" or "error" (InfoNCE error)
    batch_size: int = 128,
    corruption_rate: float = 0.6,
    temperature: float = 1.0,
    noise_std: float = 0.5,
    hidden_dim: int = 256,
    proj_dim: int = 256,
    encoder_layers: int = 4,
    head_layers: int = 2,
    lr: float = 1e-3,
    max_epochs: int = 1000,
    patience: int = 3,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[Encoder, TrainHistory]:
    """Flexible contrastive pre-training supporting all Appendix ablations:
    custom corruption strategies, loss objectives, and early stopping metrics.
    """
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)

    sampler = MarginalSampler(x_train_t) if corruption_strategy == "marginal" else None
    marginal_mean = x_train_t.mean(dim=0) if corruption_strategy == "mean" else None
    train_pool = x_train_t if corruption_strategy == "joint" else None

    encoder = Encoder(input_dim, hidden_dim, encoder_layers).to(device)
    proj_head = ProjectionHead(encoder.output_dim, hidden_dim, proj_dim, head_layers).to(device)

    # Missing feature learnable parameter
    missing_param = None
    params = list(encoder.parameters()) + list(proj_head.parameters())
    if corruption_strategy == "missing_feature":
        missing_param = nn.Parameter(torch.zeros(input_dim, device=device))
        params.append(missing_param)

    # Loss criterion
    if loss_type == "infonce":
        criterion = InfoNCELoss(temperature=temperature)
    elif loss_type == "barlow":
        criterion = BarlowTwinsLoss(lambd=5e-3)
    elif loss_type == "align_uniform":
        criterion = AlignmentUniformityLoss(t=2.0)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    optimizer = torch.optim.Adam(params, lr=lr)

    # Build static validation pairs (cycled 10 epochs as in paper)
    val_xs, val_xts = [], []
    gen = torch.Generator(device=x_val_t.device).manual_seed(0)
    for _ in range(10):
        xt = corrupt_view(
            x_val_t, corruption_strategy, sampler=sampler, marginal_mean=marginal_mean,
            train_pool=train_pool, missing_embedding=missing_param,
            corruption_rate=corruption_rate, noise_std=noise_std, generator=gen,
        )
        val_xs.append(x_val_t.clone())
        val_xts.append(xt)
    val_x = torch.cat(val_xs, dim=0)
    val_xt = torch.cat(val_xts, dim=0)

    n = x_train_t.shape[0]
    history = TrainHistory()
    best_val_metric = float("inf")
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
                continue
            xb = x_train_t[batch_idx]
            xtb = corrupt_view(
                xb, corruption_strategy, sampler=sampler, marginal_mean=marginal_mean,
                train_pool=train_pool, missing_embedding=missing_param,
                corruption_rate=corruption_rate, noise_std=noise_std,
            )

            z = proj_head(encoder(xb))
            z_tilde = proj_head(encoder(xtb))
            loss = criterion(z, z_tilde)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.train_loss.append(train_loss)

        # Validation evaluation on static pair set
        encoder.eval()
        proj_head.eval()
        val_losses = []
        val_errors = []
        with torch.no_grad():
            for start in range(0, val_x.shape[0], batch_size):
                xb = val_x[start:start + batch_size]
                xtb = val_xt[start:start + batch_size]
                if xb.shape[0] < 2:
                    continue
                z = proj_head(encoder(xb))
                z_tilde = proj_head(encoder(xtb))
                val_losses.append(criterion(z, z_tilde).item())
                if early_stop_metric == "error":
                    val_errors.append(compute_infonce_error(z, z_tilde, temperature))

        val_metric = float(np.mean(val_errors if early_stop_metric == "error" else val_losses))
        history.val_loss.append(val_metric)

        if verbose:
            print(f"[ablation-{corruption_strategy}] epoch {epoch}: train={train_loss:.4f} val_{early_stop_metric}={val_metric:.4f}")

        if val_metric < best_val_metric - 1e-6:
            best_val_metric = val_metric
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


# ---------------------------------------------------------------------------
# Co-training & Data Augmentation (Appendix B)
# ---------------------------------------------------------------------------

def train_co_training(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    input_dim: int,
    lambda_cont: float = 0.1,
    corruption_rate: float = 0.6,
    temperature: float = 1.0,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 3,
    device: str = "cpu",
) -> Tuple[Encoder, ClassificationHead, TrainHistory]:
    """Co-training baseline (Appendix B, Figure 9):
    Jointly train encoder f + classification head h + projection head g with:
      L = L_supervised + lambda_cont * L_contrastive
    Early stopping on validation classification error.
    """
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)

    sampler = MarginalSampler(x_train_t)

    encoder = Encoder(input_dim, 256, 4).to(device)
    head = ClassificationHead(encoder.output_dim, 256, n_classes, 2).to(device)
    proj_head = ProjectionHead(encoder.output_dim, 256, 256, 2).to(device)

    criterion_cont = InfoNCELoss(temperature=temperature)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()) + list(proj_head.parameters()),
        lr=lr,
    )

    n = x_train_t.shape[0]
    history = TrainHistory()
    best_val_err = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        encoder.train()
        head.train()
        proj_head.train()
        perm = torch.randperm(n, device=device)
        epoch_losses = []

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if idx.shape[0] < 2:
                continue
            xb = x_train_t[idx]
            yb = y_train_t[idx]
            xtb = scarf_corruption(xb, sampler, corruption_rate)

            # Supervised classification loss on original view
            h_orig = encoder(xb)
            logits = head(h_orig)
            loss_sup = F.cross_entropy(logits, yb)

            # Contrastive loss between original and corrupted view
            z = proj_head(h_orig)
            z_tilde = proj_head(encoder(xtb))
            loss_cont = criterion_cont(z, z_tilde)

            loss = loss_sup + lambda_cont * loss_cont

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.train_loss.append(train_loss)

        # Early stopping on clean validation classification error
        encoder.eval()
        head.eval()
        with torch.no_grad():
            val_logits = head(encoder(x_val_t))
            val_err = (val_logits.argmax(dim=-1) != y_val_t).float().mean().item()
        history.val_loss.append(val_err)

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


def train_data_augmentation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    input_dim: int,
    corruption_rate: float = 0.6,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 3,
    device: str = "cpu",
) -> Tuple[Encoder, ClassificationHead, TrainHistory]:
    """Data augmentation baseline (Appendix B, Figure 10):
    Skip pre-training and instead train directly on corrupted inputs during
    supervised training.
    """
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)

    sampler = MarginalSampler(x_train_t)

    encoder = Encoder(input_dim, 256, 4).to(device)
    head = ClassificationHead(encoder.output_dim, 256, n_classes, 2).to(device)
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
            idx = perm[start:start + batch_size]
            xb = x_train_t[idx]
            yb = y_train_t[idx]

            # Corrupt batch as data augmentation
            xb_corrupt = scarf_corruption(xb, sampler, corruption_rate)
            logits = head(encoder(xb_corrupt))
            loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.train_loss.append(train_loss)

        # Validation on clean validation set
        encoder.eval()
        head.eval()
        with torch.no_grad():
            val_logits = head(encoder(x_val_t))
            val_err = (val_logits.argmax(dim=-1) != y_val_t).float().mean().item()
        history.val_loss.append(val_err)

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
