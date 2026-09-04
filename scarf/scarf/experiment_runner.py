"""
Shared experiment runner used by both the semi-supervised, label-noise,
and benchmark scripts. Reproduces the paper's Table 1 structure: for a set
of reference fine-tuning recipes (control, dropout, mixup, label_smooth,
distill, self_train, tri_train, deep_knn, bitempered) and a set of
pre-training methods (none, scarf, scarf_ae, no_noise_ae, add_noise_ae, scarf_disc),
train every combination and return test accuracy for each.

Pre-training is performed once per trial per method and cached; each reference
method receives an independent deep copy of the pre-trained weights so retraining
does not accumulate drift across recipes.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Union

import numpy as np

from .baselines import pretrain_autoencoder, pretrain_scarf_discriminative
from .model import Encoder
from .semisup_baselines import (
    train_bitempered,
    train_deep_knn,
    train_self_distillation,
    train_self_training,
    train_tri_training,
)
from .trainer import evaluate_accuracy, finetune_classifier, pretrain_scarf

# Default lists matching Table 1 in the paper
ALL_PRETRAIN_METHODS: List[str] = [
    "none", "scarf", "scarf_ae", "no_noise_ae", "add_noise_ae", "scarf_disc"
]
DEFAULT_PRETRAIN_METHODS: List[str] = ["none", "scarf", "scarf_ae"]

DEFAULT_REFERENCE_METHODS: List[str] = ["control", "mixup", "label_smooth"]
SUPERVISED_REFERENCE_METHODS: List[str] = ["control", "dropout", "mixup", "label_smooth", "distill"]
SEMI_SUPERVISED_REFERENCE_METHODS: List[str] = [
    "control", "dropout", "mixup", "label_smooth", "distill", "self_train", "tri_train"
]
LABEL_NOISE_REFERENCE_METHODS: List[str] = [
    "control", "dropout", "mixup", "label_smooth", "distill", "deep_knn", "bitempered"
]


def _encoder_checksum(encoder: Optional[Encoder]) -> float:
    """Compute a floating point checksum of encoder parameters."""
    if encoder is None:
        return 0.0
    return float(sum(p.sum().item() for p in encoder.parameters()))


def _clone_encoder(encoder: Optional[Encoder]) -> Optional[Encoder]:
    """Return an independent deep copy of the encoder weights, or None."""
    if encoder is None:
        return None
    return copy.deepcopy(encoder)



def get_pretrained_encoder(
    method: str,
    x_pretrain: np.ndarray,
    x_val: np.ndarray,
    input_dim: int,
    device: str = "cpu",
    max_pretrain_epochs: int = 300,
) -> Optional[Encoder]:
    """Run the requested pre-training method and return the resulting
    encoder f, or None for 'no pre-training' (method == 'none').
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
    elif method == "scarf_disc":
        encoder, _ = pretrain_scarf_discriminative(
            x_pretrain, x_val, input_dim,
            max_epochs=max_pretrain_epochs, device=device,
        )
        return encoder
    else:
        raise ValueError(f"Unknown pretrain method: {method}")


def _train_reference_model(
    ref_name: str,
    x_train_ft: np.ndarray,
    y_train_ft: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    input_dim: int,
    encoder: Optional[Encoder],
    x_unlabeled: Optional[np.ndarray] = None,
    device: str = "cpu",
    max_finetune_epochs: int = 150,
    ref_kwargs: Optional[dict] = None,
) -> tuple[Encoder, nn.Module]:
    """Dispatch training for a specific reference recipe starting from an
    optionally pre-trained encoder.
    """
    kwargs = ref_kwargs.copy() if ref_kwargs else {}

    if ref_name == "control":
        enc, head, _ = finetune_classifier(
            x_train_ft, y_train_ft, x_val, y_val,
            n_classes=n_classes, encoder=encoder, input_dim=input_dim,
            max_epochs=max_finetune_epochs, device=device, **kwargs,
        )
    elif ref_name == "dropout":
        enc, head, _ = finetune_classifier(
            x_train_ft, y_train_ft, x_val, y_val,
            n_classes=n_classes, encoder=encoder, input_dim=input_dim,
            dropout_rate=kwargs.get("dropout_rate", 0.04),
            max_epochs=max_finetune_epochs, device=device,
        )
    elif ref_name == "mixup":
        enc, head, _ = finetune_classifier(
            x_train_ft, y_train_ft, x_val, y_val,
            n_classes=n_classes, encoder=encoder, input_dim=input_dim,
            mixup_alpha=kwargs.get("mixup_alpha", 0.2),
            max_epochs=max_finetune_epochs, device=device,
        )
    elif ref_name == "label_smooth":
        enc, head, _ = finetune_classifier(
            x_train_ft, y_train_ft, x_val, y_val,
            n_classes=n_classes, encoder=encoder, input_dim=input_dim,
            label_smoothing=kwargs.get("label_smoothing", 0.1),
            max_epochs=max_finetune_epochs, device=device,
        )
    elif ref_name == "distill":
        enc, head, _ = train_self_distillation(
            x_train_ft, y_train_ft, x_val, y_val,
            n_classes=n_classes, x_unlabeled=x_unlabeled,
            encoder=encoder, input_dim=input_dim,
            max_epochs=max_finetune_epochs, device=device, **kwargs,
        )
    elif ref_name == "self_train":
        enc, head, _ = train_self_training(
            x_train_ft, y_train_ft, x_val, y_val,
            n_classes=n_classes, x_unlabeled=x_unlabeled,
            encoder=encoder, input_dim=input_dim,
            threshold=kwargs.get("threshold", 0.75),
            n_iterations=kwargs.get("n_iterations", 10),
            max_epochs=max_finetune_epochs, device=device,
        )
    elif ref_name == "tri_train":
        enc, head, _ = train_tri_training(
            x_train_ft, y_train_ft, x_val, y_val,
            n_classes=n_classes, x_unlabeled=x_unlabeled,
            encoder=encoder, input_dim=input_dim,
            n_iterations=kwargs.get("n_iterations", 10),
            max_epochs=max_finetune_epochs, device=device,
        )
    elif ref_name == "deep_knn":
        enc, head, _ = train_deep_knn(
            x_train_ft, y_train_ft, x_val, y_val,
            n_classes=n_classes, encoder=encoder, input_dim=input_dim,
            k=kwargs.get("k", 50),
            max_epochs=max_finetune_epochs, device=device,
        )
    elif ref_name == "bitempered":
        enc, head, _ = train_bitempered(
            x_train_ft, y_train_ft, x_val, y_val,
            n_classes=n_classes, encoder=encoder, input_dim=input_dim,
            t1=kwargs.get("t1", 0.8), t2=kwargs.get("t2", 1.2),
            num_iters=kwargs.get("num_iters", 5),
            max_epochs=max_finetune_epochs, device=device,
        )
    else:
        # Fallback for custom recipe dictionary
        enc, head, _ = finetune_classifier(
            x_train_ft, y_train_ft, x_val, y_val,
            n_classes=n_classes, encoder=encoder, input_dim=input_dim,
            max_epochs=max_finetune_epochs, device=device, **kwargs,
        )

    return enc, head


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
    x_unlabeled: Optional[np.ndarray] = None,
    device: str = "cpu",
    max_pretrain_epochs: int = 300,
    max_finetune_epochs: int = 150,
    reference_methods: Optional[Union[List[str], Dict[str, dict]]] = None,
    pretrain_methods: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Run every (reference recipe) x (pretrain method) combination for a
    single trial and return {combo_key: test_accuracy}.

    combo_key is 'control', 'mixup', ... for pretrain_method == 'none', and
    'control+scarf', 'mixup+scarf_ae', ... otherwise -- matching the naming
    convention expected by scarf.evaluate.average_relative_gain_table.

    Callers control what 'pre-training data' vs. 'fine-tuning data' means:
      - Semi-supervised setting: x_pretrain = full training split (labels
        ignored), x_train_ft/y_train_ft = only the labeled subset,
        x_unlabeled = the unlabeled remainder.
      - Label-noise setting: x_pretrain = full training split,
        x_train_ft = full training split, y_train_ft = noisy labels.
    """
    if reference_methods is None:
        ref_dict: Dict[str, dict] = {"control": {}, "mixup": {}, "label_smooth": {}}
    elif isinstance(reference_methods, list):
        ref_dict = {m: {} for m in reference_methods}
    else:
        ref_dict = reference_methods

    pretrain_methods = pretrain_methods or DEFAULT_PRETRAIN_METHODS

    # Pre-train once per method, then reuse (via cloning) across every
    # reference fine-tuning recipe -- avoids redundant pre-training work.
    pretrained: Dict[str, Optional[Encoder]] = {}
    initial_checksums: Dict[str, float] = {}
    for pm in pretrain_methods:
        enc_pm = get_pretrained_encoder(
            pm, x_pretrain, x_val, input_dim, device, max_pretrain_epochs
        )
        pretrained[pm] = enc_pm
        initial_checksums[pm] = _encoder_checksum(enc_pm)

    results: Dict[str, float] = {}
    for ref_name, ref_kwargs in ref_dict.items():
        for pm in pretrain_methods:
            # Verify cached pre-trained encoder has not drifted
            cached_sum = _encoder_checksum(pretrained[pm])
            assert abs(cached_sum - initial_checksums[pm]) < 1e-6, (
                f"State leak detected: cached encoder for {pm} mutated! "
                f"Expected {initial_checksums[pm]}, got {cached_sum}"
            )
            # Each fine-tune recipe starts fresh from the cached pre-trained weights
            encoder_copy = _clone_encoder(pretrained[pm])
            copy_sum = _encoder_checksum(encoder_copy)
            assert abs(copy_sum - initial_checksums[pm]) < 1e-6, (
                f"State leak detected: cloned copy for {ref_name}+{pm} differs from cached! "
                f"Expected {initial_checksums[pm]}, got {copy_sum}"
            )
            enc, head = _train_reference_model(
                ref_name=ref_name,
                x_train_ft=x_train_ft,
                y_train_ft=y_train_ft,
                x_val=x_val,
                y_val=y_val,
                n_classes=n_classes,
                input_dim=input_dim,
                encoder=encoder_copy,
                x_unlabeled=x_unlabeled,
                device=device,
                max_finetune_epochs=max_finetune_epochs,
                ref_kwargs=ref_kwargs,
            )
            acc = evaluate_accuracy(enc, head, x_test, y_test, device=device)
            key = ref_name if pm == "none" else f"{ref_name}+{pm}"
            results[key] = acc

    return results
