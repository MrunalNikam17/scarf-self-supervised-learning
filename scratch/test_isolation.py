import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scarf"))

import torch
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from scarf.data import preprocess_dataset
from scarf.experiment_runner import get_pretrained_encoder, _clone_encoder, _train_reference_model

def get_checksum(model):
    if model is None:
        return 0.0
    return sum(p.sum().item() for p in model.parameters())

d = load_wine()
X = pd.DataFrame(d.data, columns=d.feature_names)
y = pd.Series(d.target)
splits = preprocess_dataset(X, y, scale="zscore", random_state=42)

input_dim = splits.x_train.shape[1]
pretrain_methods = ["none", "scarf", "scarf_ae"]
ref_dict = {"control": {}, "mixup": {}, "label_smooth": {}}

print("=== Pre-training ===")
pretrained = {}
for pm in pretrain_methods:
    pretrained[pm] = get_pretrained_encoder(pm, splits.x_train, splits.x_val, input_dim, max_pretrain_epochs=5)
    print(f"Pretrained [{pm}] initial checksum: {get_checksum(pretrained[pm]):.8f}")

print("\n=== Fine-tuning loop ===")
for ref_name in ref_dict:
    for pm in pretrain_methods:
        c_pre_cached_before = get_checksum(pretrained[pm])
        encoder_copy = _clone_encoder(pretrained[pm])
        c_copy_before = get_checksum(encoder_copy)
        
        enc, head = _train_reference_model(
            ref_name=ref_name,
            x_train_ft=splits.x_train,
            y_train_ft=splits.y_train,
            x_val=splits.x_val,
            y_val=splits.y_val,
            n_classes=splits.n_classes,
            input_dim=input_dim,
            encoder=encoder_copy,
            max_finetune_epochs=5,
        )
        
        c_copy_after = get_checksum(enc)
        c_pre_cached_after = get_checksum(pretrained[pm])
        
        print(f"[{ref_name:12s} + {pm:8s}] "
              f"cached_before={c_pre_cached_before:.8f} | "
              f"copy_before={c_copy_before:.8f} | "
              f"copy_after={c_copy_after:.8f} | "
              f"cached_after={c_pre_cached_after:.8f}")
        
        if pm != "none":
            assert c_pre_cached_before == c_pre_cached_after, f"LEAK! Cached {pm} was mutated by {ref_name}!"
            assert c_pre_cached_before == c_copy_before, f"LEAK! Cloned copy for {ref_name}+{pm} doesn't match cached!"

print("\nALL ASSERTIONS PASSED! No state leaked to cached encoders.")
