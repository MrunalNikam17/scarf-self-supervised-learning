import os
import sys
from collections import Counter
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scarf"))

from scripts.run_benchmark import load_dataset_auto
from scarf.data import preprocess_dataset, make_semi_supervised, corrupt_labels
from scarf.trainer import pretrain_scarf, finetune_classifier, evaluate_accuracy
from scarf.experiment_runner import run_all_combinations

def run_investigation():
    print("=================================================================")
    print(" INVESTIGATION 1: VEHICLE (ID 54) PER-CLASS COUNTS ACROSS 15 TRIALS")
    print("=================================================================")
    X_veh, y_veh, cat_veh, dname_veh = load_dataset_auto(54)
    print(f"Total dataset: {X_veh.shape[0]} rows, {X_veh.shape[1]} features, {y_veh.nunique()} classes")
    print(f"Overall class distribution:\n{y_veh.value_counts(normalize=True).round(3).to_dict()}")

    trial_imbalances = []
    for trial in range(15):
        splits = preprocess_dataset(X_veh, y_veh, cat_veh, scale="zscore", random_state=trial)
        n_train = len(splits.y_train)
        lab_idx, unlab_idx = make_semi_supervised(splits, labeled_frac=0.25, random_state=trial)
        y_lab = splits.y_train[lab_idx]
        
        train_c = Counter(splits.y_train)
        lab_c = Counter(y_lab)
        
        # Check min/max counts
        min_class = min(lab_c.values())
        max_class = max(lab_c.values())
        trial_imbalances.append((min_class, max_class, len(lab_idx)))
        print(f"Trial {trial:2d}: Train={dict(sorted(train_c.items()))} | "
              f"Labeled(25%={len(lab_idx)})={dict(sorted(lab_c.items()))} "
              f"[min={min_class}, max={max_class}]")

    print("\n=================================================================")
    print(" INVESTIGATION 2: SCARF PRETRAINING CONVERGENCE (VEHICLE vs WDBC)")
    print("=================================================================")
    X_wdbc, y_wdbc, cat_wdbc, dname_wdbc = load_dataset_auto(1510)
    
    datasets_to_test = [
        ("openml_54_vehicle", X_veh, y_veh, cat_veh),
        ("openml_1510_wdbc", X_wdbc, y_wdbc, cat_wdbc),
    ]

    for name, X, y, cat in datasets_to_test:
        print(f"\n--- Testing Pretraining on {name} ---")
        for trial in [0, 1]:
            splits = preprocess_dataset(X, y, cat, scale="zscore", random_state=trial)
            input_dim = splits.x_train.shape[1]
            
            enc, hist = pretrain_scarf(
                splits.x_train, splits.x_val, input_dim,
                max_epochs=300, patience=3, device="cpu", verbose=False
            )
            print(f"  Trial {trial}: Pretraining stopped at epoch {len(hist.val_loss)} (best_epoch={hist.best_epoch}). "
                  f"Start Val InfoNCE: {hist.val_loss[0]:.4f}, Best Val InfoNCE: {min(hist.val_loss):.4f}, "
                  f"Final Train InfoNCE: {hist.train_loss[-1]:.4f}")

    print("\n=================================================================")
    print(" INVESTIGATION 3: DOWNSTREAM FINETUNING DYNAMICS ON VEHICLE")
    print("=================================================================")
    # Look at why SCARF causes 16-21 pp drop on vehicle specifically
    splits = preprocess_dataset(X_veh, y_veh, cat_veh, scale="zscore", random_state=0)
    lab_idx, unlab_idx = make_semi_supervised(splits, labeled_frac=0.25, random_state=0)
    x_lab = splits.x_train[lab_idx]
    y_lab = splits.y_train[lab_idx]
    input_dim = splits.x_train.shape[1]

    # Pretrain SCARF
    scarf_enc, scarf_hist = pretrain_scarf(
        splits.x_train, splits.x_val, input_dim, max_epochs=300, patience=3, device="cpu"
    )

    # Compare Control vs Control+SCARF
    print("\n[Control vs SCARF - Trial 0]")
    # Control
    ctrl_enc, ctrl_head, ctrl_hist = finetune_classifier(
        x_train=x_lab, y_train=y_lab, x_val=splits.x_val, y_val=splits.y_val,
        n_classes=splits.n_classes, input_dim=input_dim, encoder=None,
        max_epochs=150, patience=15, device="cpu"
    )
    ctrl_val_acc = evaluate_accuracy(ctrl_enc, ctrl_head, splits.x_val, splits.y_val)
    ctrl_test_acc = evaluate_accuracy(ctrl_enc, ctrl_head, splits.x_test, splits.y_test)
    ctrl_train_acc = evaluate_accuracy(ctrl_enc, ctrl_head, x_lab, y_lab)
    print(f"Control (from scratch): Stopped at epoch {ctrl_hist.best_epoch}, "
          f"Train Acc={ctrl_train_acc:.3f}, Val Acc={ctrl_val_acc:.3f}, Test Acc={ctrl_test_acc:.3f}")

    # SCARF + fine-tune
    import copy
    sc_enc_copy = copy.deepcopy(scarf_enc)
    sc_enc, sc_head, sc_hist = finetune_classifier(
        x_train=x_lab, y_train=y_lab, x_val=splits.x_val, y_val=splits.y_val,
        n_classes=splits.n_classes, input_dim=input_dim, encoder=sc_enc_copy,
        max_epochs=150, patience=15, device="cpu"
    )
    sc_val_acc = evaluate_accuracy(sc_enc, sc_head, splits.x_val, splits.y_val)
    sc_test_acc = evaluate_accuracy(sc_enc, sc_head, splits.x_test, splits.y_test)
    sc_train_acc = evaluate_accuracy(sc_enc, sc_head, x_lab, y_lab)
    print(f"SCARF+FT: Stopped at epoch {sc_hist.best_epoch}, "
          f"Train Acc={sc_train_acc:.3f}, Val Acc={sc_val_acc:.3f}, Test Acc={sc_test_acc:.3f}")

    # Compare Label Smoothing vs Label Smoothing+SCARF
    print("\n[Label Smoothing vs Label Smoothing+SCARF - Trial 0]")
    ls_enc, ls_head, ls_hist = finetune_classifier(
        x_train=x_lab, y_train=y_lab, x_val=splits.x_val, y_val=splits.y_val,
        n_classes=splits.n_classes, input_dim=input_dim, encoder=None,
        label_smoothing=0.1, max_epochs=150, patience=15, device="cpu"
    )
    ls_val_acc = evaluate_accuracy(ls_enc, ls_head, splits.x_val, splits.y_val)
    ls_test_acc = evaluate_accuracy(ls_enc, ls_head, splits.x_test, splits.y_test)
    print(f"Label Smooth (scratch): Stopped at epoch {ls_hist.best_epoch}, "
          f"Val Acc={ls_val_acc:.3f}, Test Acc={ls_test_acc:.3f}")

    sc_ls_enc, sc_ls_head, sc_ls_hist = finetune_classifier(
        x_train=x_lab, y_train=y_lab, x_val=splits.x_val, y_val=splits.y_val,
        n_classes=splits.n_classes, input_dim=input_dim, encoder=copy.deepcopy(scarf_enc),
        label_smoothing=0.1, max_epochs=150, patience=15, device="cpu"
    )
    sc_ls_val_acc = evaluate_accuracy(sc_ls_enc, sc_ls_head, splits.x_val, splits.y_val)
    sc_ls_test_acc = evaluate_accuracy(sc_ls_enc, sc_ls_head, splits.x_test, splits.y_test)
    print(f"Label Smooth+SCARF: Stopped at epoch {sc_ls_hist.best_epoch}, "
          f"Val Acc={sc_ls_val_acc:.3f}, Test Acc={sc_ls_test_acc:.3f}")

    print("\n=================================================================")
    print(" INVESTIGATION 4: CHECK CORRUPT_LABELS STRATIFICATION")
    print("=================================================================")
    for trial in range(3):
        splits = preprocess_dataset(X_veh, y_veh, cat_veh, scale="zscore", random_state=trial)
        y_noisy = corrupt_labels(splits.y_train, splits.n_classes, noise_frac=0.3, random_state=trial)
        print(f"Trial {trial}: Orig train classes={dict(sorted(Counter(splits.y_train).items()))}")
        print(f"         Noisy train classes={dict(sorted(Counter(y_noisy).items()))}")

if __name__ == "__main__":
    run_investigation()
