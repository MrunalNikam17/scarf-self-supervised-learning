"""
Offline smoke test: exercises the full pipeline (preprocessing, SCARF
pre-training, fine-tuning, baselines, evaluation) on a few small datasets
bundled with scikit-learn (no internet required), so you can sanity-check
your installation before running the real OpenML-CC18 benchmark, which
needs network access.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_digits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scarf.baselines import pretrain_autoencoder, pretrain_scarf_discriminative
from scarf.data import preprocess_dataset
from scarf.evaluate import relative_gain, win_matrix
from scarf.trainer import evaluate_accuracy, finetune_classifier, pretrain_scarf


def sklearn_dataset_to_xy(loader):
    d = loader()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(d.target)
    categorical_mask = [False] * X.shape[1]  # all numeric
    return X, y, categorical_mask


DATASETS = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "digits": load_digits,
}


def main():
    n_trials = 3
    methods = ["control", "scarf", "scarf_ae", "no_noise_ae", "add_noise_ae", "scarf_disc"]
    max_pretrain_epochs = 30   # small, just to prove the loop + early stopping works
    max_finetune_epochs = 30
    device = "cpu"

    results = defaultdict(lambda: defaultdict(list))

    for name, loader in DATASETS.items():
        X, y, cat_mask = sklearn_dataset_to_xy(loader)
        print(f"\n=== {name}: {X.shape[0]} rows x {X.shape[1]} features, {y.nunique()} classes ===")
        for trial in range(n_trials):
            splits = preprocess_dataset(X, y, cat_mask, scale="zscore", random_state=trial)
            input_dim = splits.x_train.shape[1]

            for method in methods:
                t0 = time.time()
                if method == "control":
                    encoder = None
                elif method == "scarf":
                    encoder, hist = pretrain_scarf(splits.x_train, splits.x_val, input_dim,
                                                    max_epochs=max_pretrain_epochs, device=device)
                elif method == "scarf_ae":
                    encoder, hist = pretrain_autoencoder(splits.x_train, splits.x_val, input_dim,
                                                          noise_type="scarf", max_epochs=max_pretrain_epochs, device=device)
                elif method == "no_noise_ae":
                    encoder, hist = pretrain_autoencoder(splits.x_train, splits.x_val, input_dim,
                                                          noise_type="none", max_epochs=max_pretrain_epochs, device=device)
                elif method == "add_noise_ae":
                    encoder, hist = pretrain_autoencoder(splits.x_train, splits.x_val, input_dim,
                                                          noise_type="gaussian", max_epochs=max_pretrain_epochs, device=device)
                elif method == "scarf_disc":
                    encoder, hist = pretrain_scarf_discriminative(splits.x_train, splits.x_val, input_dim,
                                                                   max_epochs=max_pretrain_epochs, device=device)

                encoder, head, _ = finetune_classifier(
                    splits.x_train, splits.y_train, splits.x_val, splits.y_val,
                    n_classes=splits.n_classes, encoder=encoder, input_dim=input_dim,
                    max_epochs=max_finetune_epochs, device=device,
                )
                acc = evaluate_accuracy(encoder, head, splits.x_test, splits.y_test, device=device)
                results[method][name].append(acc)
                print(f"  trial {trial+1}/{n_trials} | {method:12s} | acc={acc:.4f} ({time.time()-t0:.1f}s)")

    results_np = {m: {d: np.array(v) for d, v in dd.items()} for m, dd in results.items()}

    print("\n=== Summary (mean +/- std test accuracy) ===")
    for method, dd in results_np.items():
        for dataset, accs in dd.items():
            print(f"{method:12s} {dataset:15s} {accs.mean():.4f} +/- {accs.std():.4f}")

    print("\n=== Win matrix ===")
    print(win_matrix(results_np))

    print("\n=== Relative gain vs control ===")
    for method in methods:
        if method == "control":
            continue
        gains = relative_gain(results_np[method], results_np["control"])
        print(f"{method}: {gains.to_dict()}  (avg={gains.mean() if len(gains) else float('nan'):.2f}%)")

    print("\nSmoke test completed successfully.")


if __name__ == "__main__":
    main()
