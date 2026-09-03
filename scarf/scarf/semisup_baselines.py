"""
Semi-supervised and label-noise baselines matching the paper (Sections 4.2, 4.3, Appendix C):
  - Self-training (Yarowsky 1995): iterative pseudo-labeling with softmax threshold 0.75, 10 iters.
  - Tri-training (Zhou & Li 2005): 3 models via bootstrap sampling, adding unlabeled points
    where the other two models' predictions agree.
  - Self-distillation (Hinton et al. 2015; Zhang et al. 2019a): train on labeled data, then
    retrain student using teacher's soft labels across labeled + unlabeled data.
  - Deep k-NN (Bahri et al. 2020): filter out points where representation-space k-NN (k=50)
    disagrees with given noisy label, then retrain on filtered subset.
  - Bi-tempered logistic loss (Amid et al. 2019): robust loss with t1=0.8, t2=1.2, 5 iterations.

All functions accept numpy arrays, accept an optional pre-trained Encoder (which is cloned
before each retraining round so every model starts fresh from pre-trained weights without
accumulating drift), and return (Encoder, ClassificationHead, TrainHistory).
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ClassificationHead, Encoder
from .trainer import TrainHistory, evaluate_accuracy, finetune_classifier


# ---------------------------------------------------------------------------
# Bi-tempered Logistic Loss (Amid et al. 2019)
# ---------------------------------------------------------------------------

def _log_t(u: torch.Tensor, t: float) -> torch.Tensor:
    """Tempered logarithm: log_t(u) = (u^(1-t) - 1) / (1-t) if t != 1 else log(u)."""
    if abs(t - 1.0) < 1e-7:
        return torch.log(u)
    return (u.clamp(min=1e-8) ** (1.0 - t) - 1.0) / (1.0 - t)


def _exp_t(u: torch.Tensor, t: float) -> torch.Tensor:
    """Tempered exponential: exp_t(u) = [1 + (1-t)u]_+^(1/(1-t)) if t != 1 else exp(u)."""
    if abs(t - 1.0) < 1e-7:
        return torch.exp(u)
    return F.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))


class BiTemperedLogisticLoss(nn.Module):
    """Robust Bi-tempered Logistic Loss based on Bregman Divergences (Amid et al., 2019).
    Paper hyperparameters: t1=0.8, t2=1.2, 5 fixed-point iterations.
    """

    def __init__(self, t1: float = 0.8, t2: float = 1.2, num_iters: int = 5):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.num_iters = num_iters

    def forward(self, activations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """activations: (N, C) logits, labels: (N,) integer class indices."""
        mu = torch.max(activations, dim=-1, keepdim=True)[0]
        low = mu - 1.0
        high = mu + (1.0 / max(1e-4, self.t2 - 1.0))

        # Binary search / fixed point for normalization constant lambda
        for _ in range(self.num_iters):
            mid = (low + high) / 2.0
            p = _exp_t(activations - mid, self.t2)
            s = torch.sum(p, dim=-1, keepdim=True)
            low = torch.where(s > 1.0, mid, low)
            high = torch.where(s <= 1.0, mid, high)

        lam = (low + high) / 2.0
        probs = _exp_t(activations - lam, self.t2)
        probs = probs / torch.sum(probs, dim=-1, keepdim=True).clamp(min=1e-8)

        # Bi-tempered cross-entropy
        p_y = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        loss = - _log_t(p_y.clamp(min=1e-8), self.t1) - (1.0 - torch.sum(probs ** (2.0 - self.t1), dim=-1)) / (2.0 - self.t1)
        return loss.mean()


def train_bitempered(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    encoder: Optional[Encoder] = None,
    input_dim: Optional[int] = None,
    t1: float = 0.8,
    t2: float = 1.2,
    num_iters: int = 5,
    max_epochs: int = 200,
    patience: int = 3,
    device: str = "cpu",
    **kwargs,
) -> tuple[Encoder, ClassificationHead, TrainHistory]:
    """Train classifier using bi-tempered logistic loss (Amid et al. 2019)."""
    loss_fn = BiTemperedLogisticLoss(t1=t1, t2=t2, num_iters=num_iters)
    return finetune_classifier(
        x_train, y_train, x_val, y_val,
        n_classes=n_classes, encoder=encoder, input_dim=input_dim,
        loss_fn=loss_fn, max_epochs=max_epochs, patience=patience,
        device=device, **kwargs,
    )


# ---------------------------------------------------------------------------
# Deep k-NN (Bahri et al. 2020)
# ---------------------------------------------------------------------------

def train_deep_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    encoder: Optional[Encoder] = None,
    input_dim: Optional[int] = None,
    k: int = 50,
    max_epochs: int = 200,
    patience: int = 3,
    device: str = "cpu",
    **kwargs,
) -> tuple[Encoder, ClassificationHead, TrainHistory]:
    """Deep k-NN label-noise baseline (Bahri et al. 2020):
    1. Initial model training on noisy training data.
    2. Extract representation vectors f(x) for all training data.
    3. Compute k-NN (k=50) majority vote in representation space.
    4. Filter out points where k-NN prediction disagrees with given label.
    5. Retrain fresh model (starting fresh from a clone of pre-trained encoder)
       on the filtered clean subset.
    """
    input_dim = input_dim or x_train.shape[1]
    enc_clone = copy.deepcopy(encoder) if encoder is not None else None

    # Step 1: Initial training
    initial_enc, initial_head, _ = finetune_classifier(
        x_train, y_train, x_val, y_val,
        n_classes=n_classes, encoder=enc_clone, input_dim=input_dim,
        max_epochs=max_epochs, patience=patience, device=device, **kwargs,
    )

    # Step 2: Extract representations
    initial_enc.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
        feats = initial_enc(x_t).cpu().numpy()

    # Step 3: k-NN majority vote
    n_samples = feats.shape[0]
    effective_k = min(k, n_samples - 1)
    if effective_k < 1:
        # Not enough samples for k-NN, return initial model
        return initial_enc, initial_head, TrainHistory()

    # Pairwise Euclidean distances
    dists = np.linalg.norm(feats[:, None, :] - feats[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    neighbor_indices = np.argpartition(dists, effective_k, axis=1)[:, :effective_k]

    neighbor_labels = y_train[neighbor_indices]  # (N, k)
    knn_preds = np.array([stats.mode(row, keepdims=False)[0] for row in neighbor_labels])

    # Step 4: Filter out points where k-NN prediction disagrees with given label
    keep_mask = (knn_preds == y_train)

    # Defensive check: ensure all classes have at least one sample retained
    for c in range(n_classes):
        if not np.any(keep_mask & (y_train == c)):
            class_indices = np.where(y_train == c)[0]
            if len(class_indices) > 0:
                keep_mask[class_indices[0]] = True

    x_filtered = x_train[keep_mask]
    y_filtered = y_train[keep_mask]

    # Step 5: Retrain fresh model on filtered subset, starting fresh from pre-trained encoder
    final_enc_init = copy.deepcopy(encoder) if encoder is not None else None
    return finetune_classifier(
        x_filtered, y_filtered, x_val, y_val,
        n_classes=n_classes, encoder=final_enc_init, input_dim=input_dim,
        max_epochs=max_epochs, patience=patience, device=device, **kwargs,
    )


# ---------------------------------------------------------------------------
# Self-Distillation (Hinton et al. 2015; Zhang et al. 2019a)
# ---------------------------------------------------------------------------

def train_self_distillation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    x_unlabeled: Optional[np.ndarray] = None,
    encoder: Optional[Encoder] = None,
    input_dim: Optional[int] = None,
    max_epochs: int = 200,
    patience: int = 3,
    lr: float = 1e-3,
    batch_size: int = 128,
    device: str = "cpu",
    **kwargs,
) -> tuple[Encoder, ClassificationHead, TrainHistory]:
    """Self-distillation baseline:
    1. Train teacher model on labeled data.
    2. Compute soft prediction probabilities across all data (labeled + unlabeled).
    3. Retrain student model (starting fresh from a clone of pre-trained encoder)
       on the combined data using soft cross-entropy with early stopping.
    """
    input_dim = input_dim or x_train.shape[1]
    enc_clone = copy.deepcopy(encoder) if encoder is not None else None

    # Step 1: Train teacher model on labeled data
    teacher_enc, teacher_head, _ = finetune_classifier(
        x_train, y_train, x_val, y_val,
        n_classes=n_classes, encoder=enc_clone, input_dim=input_dim,
        max_epochs=max_epochs, patience=patience, device=device, **kwargs,
    )

    # Step 2: Generate soft targets for all training data
    if x_unlabeled is not None and len(x_unlabeled) > 0:
        x_all = np.concatenate([x_train, x_unlabeled], axis=0)
    else:
        x_all = x_train

    teacher_enc.eval()
    teacher_head.eval()
    with torch.no_grad():
        x_all_t = torch.as_tensor(x_all, dtype=torch.float32, device=device)
        logits_all = teacher_head(teacher_enc(x_all_t))
        soft_targets = F.softmax(logits_all, dim=-1)

    # Step 3: Train student model on soft targets
    student_enc = copy.deepcopy(encoder) if encoder is not None else Encoder(input_dim, 256, 4)
    student_enc = student_enc.to(device)
    student_head = ClassificationHead(student_enc.output_dim, 256, n_classes, 2).to(device)

    optimizer = torch.optim.Adam(list(student_enc.parameters()) + list(student_head.parameters()), lr=lr)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)

    n_all = x_all_t.shape[0]
    history = TrainHistory()
    best_val_err = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        student_enc.train()
        student_head.train()
        perm = torch.randperm(n_all, device=device)
        epoch_losses = []

        for start in range(0, n_all, batch_size):
            idx = perm[start:start + batch_size]
            xb = x_all_t[idx]
            tb = soft_targets[idx]

            logits = student_head(student_enc(xb))
            # Soft cross-entropy: - sum(t * log_softmax(logits))
            loss = - torch.sum(tb * F.log_softmax(logits, dim=-1), dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.train_loss.append(train_loss)

        # Early stopping on validation classification error
        student_enc.eval()
        student_head.eval()
        with torch.no_grad():
            val_logits = student_head(student_enc(x_val_t))
            val_preds = val_logits.argmax(dim=-1)
            val_err = (val_preds != y_val_t).float().mean().item()
        history.val_loss.append(val_err)

        if val_err < best_val_err - 1e-6:
            best_val_err = val_err
            best_state = {
                "encoder": {k: v.detach().clone() for k, v in student_enc.state_dict().items()},
                "head": {k: v.detach().clone() for k, v in student_head.state_dict().items()},
            }
            history.best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        student_enc.load_state_dict(best_state["encoder"])
        student_head.load_state_dict(best_state["head"])

    return student_enc, student_head, history


# ---------------------------------------------------------------------------
# Self-Training (Yarowsky 1995; McClosky et al. 2006)
# ---------------------------------------------------------------------------

def train_self_training(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    x_unlabeled: Optional[np.ndarray] = None,
    encoder: Optional[Encoder] = None,
    input_dim: Optional[int] = None,
    threshold: float = 0.75,
    n_iterations: int = 10,
    max_epochs: int = 200,
    patience: int = 3,
    device: str = "cpu",
    **kwargs,
) -> tuple[Encoder, ClassificationHead, TrainHistory]:
    """Self-training baseline:
    Iteratively train on pseudo-labeled data (starting from original labeled dataset),
    adding predictions with softmax confidence > 0.75 as pseudo-labels.
    Runs for up to 10 iterations. Each round starts fresh from a clone of the pre-trained encoder.
    """
    input_dim = input_dim or x_train.shape[1]
    if x_unlabeled is None or len(x_unlabeled) == 0:
        # No unlabeled data available, fall back to standard fine-tuning
        enc_clone = copy.deepcopy(encoder) if encoder is not None else None
        return finetune_classifier(
            x_train, y_train, x_val, y_val,
            n_classes=n_classes, encoder=enc_clone, input_dim=input_dim,
            max_epochs=max_epochs, patience=patience, device=device, **kwargs,
        )

    l_x = x_train.copy()
    l_y = y_train.copy()
    u_x = x_unlabeled.copy()

    current_enc = None
    current_head = None
    last_history = TrainHistory()

    for it in range(n_iterations):
        if len(u_x) == 0:
            break

        enc_init = copy.deepcopy(encoder) if encoder is not None else None
        current_enc, current_head, last_history = finetune_classifier(
            l_x, l_y, x_val, y_val,
            n_classes=n_classes, encoder=enc_init, input_dim=input_dim,
            max_epochs=max_epochs, patience=patience, device=device, **kwargs,
        )

        current_enc.eval()
        current_head.eval()
        with torch.no_grad():
            u_t = torch.as_tensor(u_x, dtype=torch.float32, device=device)
            probs = F.softmax(current_head(current_enc(u_t)), dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)
            max_probs = max_probs.cpu().numpy()
            preds = preds.cpu().numpy()

        confident_mask = (max_probs >= threshold)
        if not np.any(confident_mask):
            break

        # Add confident predictions to labeled set
        l_x = np.concatenate([l_x, u_x[confident_mask]], axis=0)
        l_y = np.concatenate([l_y, preds[confident_mask]], axis=0)
        u_x = u_x[~confident_mask]

    # Final training round on accumulated dataset
    final_enc_init = copy.deepcopy(encoder) if encoder is not None else None
    return finetune_classifier(
        l_x, l_y, x_val, y_val,
        n_classes=n_classes, encoder=final_enc_init, input_dim=input_dim,
        max_epochs=max_epochs, patience=patience, device=device, **kwargs,
    )


# ---------------------------------------------------------------------------
# Tri-Training (Zhou & Li 2005)
# ---------------------------------------------------------------------------

def train_tri_training(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    x_unlabeled: Optional[np.ndarray] = None,
    encoder: Optional[Encoder] = None,
    input_dim: Optional[int] = None,
    n_iterations: int = 10,
    max_epochs: int = 200,
    patience: int = 3,
    random_state: int = 0,
    device: str = "cpu",
    **kwargs,
) -> tuple[Encoder, ClassificationHead, TrainHistory]:
    """Tri-training baseline (Zhou & Li 2005):
    3 models initialized via bootstrap sampling of labeled data.
    Each iteration, model i's training set is updated by adding unlabeled points
    where the other two models' predictions agree (no arbitrary confidence threshold,
    strictly following the paper's description).
    """
    input_dim = input_dim or x_train.shape[1]
    if x_unlabeled is None or len(x_unlabeled) == 0:
        enc_clone = copy.deepcopy(encoder) if encoder is not None else None
        return finetune_classifier(
            x_train, y_train, x_val, y_val,
            n_classes=n_classes, encoder=enc_clone, input_dim=input_dim,
            max_epochs=max_epochs, patience=patience, device=device, **kwargs,
        )

    rng = np.random.RandomState(random_state)
    n_labeled = x_train.shape[0]

    # Bootstrap sample for each model
    l_sets = []
    for _ in range(3):
        boot_idx = rng.choice(n_labeled, size=n_labeled, replace=True)
        # Ensure at least 1 instance per present class in each bootstrap
        present_classes = np.unique(y_train)
        boot_classes = np.unique(y_train[boot_idx])
        missing = set(present_classes) - set(boot_classes)
        for m in missing:
            replace_pos = rng.randint(0, n_labeled)
            boot_idx[replace_pos] = np.where(y_train == m)[0][0]
        l_sets.append((x_train[boot_idx].copy(), y_train[boot_idx].copy()))

    models = [None, None, None]

    for it in range(n_iterations):
        # Train all 3 models on their current sets
        for i in range(3):
            enc_init = copy.deepcopy(encoder) if encoder is not None else None
            m_enc, m_head, _ = finetune_classifier(
                l_sets[i][0], l_sets[i][1], x_val, y_val,
                n_classes=n_classes, encoder=enc_init, input_dim=input_dim,
                max_epochs=max_epochs, patience=patience, device=device, **kwargs,
            )
            models[i] = (m_enc, m_head)

        # Generate predictions on unlabeled data for all 3 models
        preds = []
        u_t = torch.as_tensor(x_unlabeled, dtype=torch.float32, device=device)
        for m_enc, m_head in models:
            m_enc.eval()
            m_head.eval()
            with torch.no_grad():
                preds.append(m_head(m_enc(u_t)).argmax(dim=-1).cpu().numpy())

        # For each model i, find unlabeled points where other two models agree
        new_points_added = False
        new_l_sets = []
        for i in range(3):
            j, k = [idx for idx in range(3) if idx != i]
            agree_mask = (preds[j] == preds[k])
            if np.any(agree_mask):
                add_x = x_unlabeled[agree_mask]
                add_y = preds[j][agree_mask]
                cur_x = np.concatenate([l_sets[i][0], add_x], axis=0)
                cur_y = np.concatenate([l_sets[i][1], add_y], axis=0)
                new_l_sets.append((cur_x, cur_y))
                new_points_added = True
            else:
                new_l_sets.append(l_sets[i])

        l_sets = new_l_sets
        if not new_points_added:
            break

    # Train final classifier on union of all 3 models' expanded datasets
    all_x = np.concatenate([l_sets[0][0], l_sets[1][0], l_sets[2][0]], axis=0)
    all_y = np.concatenate([l_sets[0][1], l_sets[1][1], l_sets[2][1]], axis=0)
    final_enc_init = copy.deepcopy(encoder) if encoder is not None else None
    return finetune_classifier(
        all_x, all_y, x_val, y_val,
        n_classes=n_classes, encoder=final_enc_init, input_dim=input_dim,
        max_epochs=max_epochs, patience=patience, device=device, **kwargs,
    )
