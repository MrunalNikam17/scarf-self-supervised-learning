"""
Shared experiment runner used by both the semi-supervised and label-noise
scripts. Reproduces the paper's Table 1 structure: for a set of reference
fine-tuning recipes (control, mixup, label smoothing, ...) and a set of
pre-training methods (none, SCARF, SCARF-AE, ...), train every combination
and return test accuracy for each.

This intentionally does NOT touch corruption.py, model.py, losses.py, or
trainer.py -- it only composes the existing building blocks.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np

from .baselines import pretrain_autoencoder
from .model import Encoder
from .trainer import evaluate_accuracy, finetune_classifier, pretrain_scarf

# Reference fine-tuning recipes to combine with each pre-training method.
# Keys become part of the result dict's method names; values are kwargs
# forwarded straight into finetune_classifier (which already supports
# mixup_alpha and label_smoothing).
DEFAULT_REFERENCE_METHODS: Dict[str, dict] = {
    "control": {},
    "mixup": {"mixup_alpha": 0.2},
    "label_smooth": {"label_smoothing": 0.1},
}

# Pre-training methods to try on top of each reference recipe.
# "none" means: fine-tune from a randomly-initialized encoder (no pre-training).
DEFAULT_PRETRAIN_METHODS: List[str] = ["none", "scarf", "scarf_ae"]


def _clone_encoder(encoder: Optional[Encoder]) -> Optional[Encoder]:
    """finetune_classifier trains the encoder it's given in place. When we
    reuse one pre-trained encoder across several reference recipes (control,
    mixup, label_smooth, ...) we must give each recipe its own copy of the
    pre-trained weights, or later recipes would start from an encoder
    already fine-tuned by an earlier recipe.
    """
    if encoder is None:
        return None
    return copy.deepcopy(encoder)


def get_pretrained_encoder(
    method: str,
    x_pretrain: np.ndarray,
    x_val: np.ndarray,
    input_dim: int,
    device: str,
    max_pretrain_epochs: int,
) -> Optional[Encoder]:
    """Run the requested pre-training method and return the resulting
    encoder f, or None for "no pre-training" (method == "none").
    """
    if method == "none":
        return None
    elif method == "scarf":
        encoder, _ = pretrain_scarf(
            x_pretrain, x_val, input_dim, max_epochs=max_pretrain_epochs, device=device,
        )
        return encoder
    elif method == "scarf_ae":
        encoder, _ = pretrain_autoencoder(
            x_pretrain, x_val, input_dim, noise_type="scarf",
            max_epochs=max_pretrain_epochs, device=device,
        )
        return encoder
    elif method == "no_noise_ae":
        encoder, _ = pretrain_autoencoder(
            x_pretrain, x_val, input_dim, noise_type="none",
            max_epochs=max_pretrain_epochs, device=device,
        )
        return encoder
    elif method == "add_noise_ae":
        encoder, _ = pretrain_autoencoder(
            x_pretrain, x_val, input_dim, noise_type="gaussian",
            max_epochs=max_pretrain_epochs, device=device,
        )
        return encoder
    else:
        raise ValueError(f"Unknown pretrain method: {method}")


def run_all_combinations(
    x_pretrain: np.ndarray,
    x_train_ft: np.ndarray,
    y_train_ft: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    input_dim: int,
    device: str = "cpu",
    max_pretrain_epochs: int = 300,
    max_finetune_epochs: int = 150,
    reference_methods: Optional[Dict[str, dict]] = None,
    pretrain_methods: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Run every (reference recipe) x (pretrain method) combination for a
    single trial and return {combo_key: test_accuracy}.

    combo_key is "control", "mixup", ... for pretrain_method == "none", and
    "control+scarf", "mixup+scarf_ae", ... otherwise -- matching the naming
    convention expected by scarf.evaluate.average_relative_gain_table.

    Callers control what "pre-training data" vs. "fine-tuning data" means:
      - Semi-supervised setting: x_pretrain = full training split (labels
        ignored), x_train_ft/y_train_ft = only the labeled subset.
      - Label-noise setting: x_pretrain = full training split,
        x_train_ft = full training split, y_train_ft = noisy labels.
    """
    reference_methods = reference_methods or DEFAULT_REFERENCE_METHODS
    pretrain_methods = pretrain_methods or DEFAULT_PRETRAIN_METHODS

    # Pre-train once per method, then reuse (via cloning) across every
    # reference fine-tuning recipe -- avoids redoing pre-training 3x.
    pretrained = {
        pm: get_pretrained_encoder(pm, x_pretrain, x_val, input_dim, device, max_pretrain_epochs)
        for pm in pretrain_methods
    }

    results: Dict[str, float] = {}
    for ref_name, ref_kwargs in reference_methods.items():
        for pm in pretrain_methods:
            encoder_copy = _clone_encoder(pretrained[pm])
            enc, head, _ = finetune_classifier(
                x_train_ft, y_train_ft, x_val, y_val,
                n_classes=n_classes, encoder=encoder_copy, input_dim=input_dim,
                max_epochs=max_finetune_epochs, device=device, **ref_kwargs,
            )
            acc = evaluate_accuracy(enc, head, x_test, y_test, device=device)
            key = ref_name if pm == "none" else f"{ref_name}+{pm}"
            results[key] = acc
    return results
