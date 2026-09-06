import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scarf"))

from scripts.run_benchmark import load_dataset_auto
from scarf.data import preprocess_dataset, make_semi_supervised, corrupt_labels
from scarf.experiment_runner import run_all_combinations, SEMI_SUPERVISED_REFERENCE_METHODS, LABEL_NOISE_REFERENCE_METHODS

def test_timing():
    print("===============================================================")
    print(" TIMING BENCHMARK BASELINES (1 TRIAL ON BALANCE-SCALE ID 11)")
    print("===============================================================")
    X, y, cat_mask, dname = load_dataset_auto(11)
    input_dim = X.shape[1]
    splits = preprocess_dataset(X, y, cat_mask, scale="zscore", random_state=0)

    # 1. Semi-supervised timing
    print("\n--- Testing Semi-Supervised Reference Methods ---")
    lab_idx, unlab_idx = make_semi_supervised(splits, labeled_frac=0.25, random_state=0)
    x_ft = splits.x_train[lab_idx]
    y_ft = splits.y_train[lab_idx]
    x_unlab = splits.x_train[unlab_idx]

    for ref in SEMI_SUPERVISED_REFERENCE_METHODS:
        t0 = time.time()
        res = run_all_combinations(
            x_pretrain=splits.x_train,
            x_train_ft=x_ft,
            y_train_ft=y_ft,
            x_val=splits.x_val,
            y_val=splits.y_val,
            x_test=splits.x_test,
            y_test=splits.y_test,
            n_classes=splits.n_classes,
            input_dim=input_dim,
            x_unlabeled=x_unlab,
            pretrain_methods=["none", "scarf", "scarf_ae"],
            reference_methods=[ref],
            max_pretrain_epochs=150,
            max_finetune_epochs=100,
        )
        elapsed = time.time() - t0
        print(f"  {ref:15s}: {elapsed:.2f}s | accs: {res}")

    # 2. Label-noise timing
    print("\n--- Testing Label-Noise Reference Methods ---")
    y_noisy = corrupt_labels(splits.y_train, splits.n_classes, noise_frac=0.30, random_state=0)
    for ref in ["deep_knn", "bitempered"]:
        t0 = time.time()
        res = run_all_combinations(
            x_pretrain=splits.x_train,
            x_train_ft=splits.x_train,
            y_train_ft=y_noisy,
            x_val=splits.x_val,
            y_val=splits.y_val,
            x_test=splits.x_test,
            y_test=splits.y_test,
            n_classes=splits.n_classes,
            input_dim=input_dim,
            x_unlabeled=None,
            pretrain_methods=["none", "scarf", "scarf_ae"],
            reference_methods=[ref],
            max_pretrain_epochs=150,
            max_finetune_epochs=100,
        )
        elapsed = time.time() - t0
        print(f"  {ref:15s}: {elapsed:.2f}s | accs: {res}")

if __name__ == "__main__":
    test_timing()
