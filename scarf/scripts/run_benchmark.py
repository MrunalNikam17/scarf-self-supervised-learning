"""
Run the SCARF benchmark on a small set of OpenML-CC18 datasets.

Compares supervised-only ("control") training against SCARF-pretrained
fine-tuning (and, optionally, the autoencoder / discriminative-SCARF
baselines), repeated over several train/val/test splits, and reports:
  - per-dataset mean +/- std test accuracy for each method
  - a win matrix (Welch's t-test, p<0.05) between methods
  - average relative gain of each method vs. control

Requires internet access to openml.org (not available inside this sandbox,
but works in a normal environment). Install deps first:
    pip install torch scikit-learn pandas numpy scipy openml

Example:
    python scripts/run_benchmark.py \
        --dataset-ids 11 37 54 1510 1494 15 \
        --n-trials 10 \
        --methods control scarf \
        --output-dir results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scarf.baselines import pretrain_autoencoder, pretrain_scarf_discriminative
from scarf.data import load_openml_dataset, preprocess_dataset
from scarf.evaluate import win_matrix, relative_gain
from scarf.trainer import evaluate_accuracy, finetune_classifier, pretrain_scarf

# A handful of small/medium OpenML-CC18 datasets that are fast to iterate on.
# (name is informational only; the numeric id is what's used)
DEFAULT_DATASET_IDS = {
    11: "balance-scale",
    37: "diabetes",
    54: "vehicle",
    1510: "wdbc",
    1494: "qsar-biodeg",
    15: "breast-w",
}

ALL_METHODS = ["control", "scarf", "scarf_ae", "no_noise_ae", "add_noise_ae", "scarf_disc"]


def run_one_trial(splits, method: str, device: str, max_pretrain_epochs: int, max_finetune_epochs: int, seed: int):
    input_dim = splits.x_train.shape[1]

    encoder = None
    if method == "control":
        encoder = None
    elif method == "scarf":
        encoder, _ = pretrain_scarf(
            splits.x_train, splits.x_val, input_dim,
            max_epochs=max_pretrain_epochs, device=device,
        )
    elif method == "scarf_ae":
        encoder, _ = pretrain_autoencoder(
            splits.x_train, splits.x_val, input_dim, noise_type="scarf",
            max_epochs=max_pretrain_epochs, device=device,
        )
    elif method == "no_noise_ae":
        encoder, _ = pretrain_autoencoder(
            splits.x_train, splits.x_val, input_dim, noise_type="none",
            max_epochs=max_pretrain_epochs, device=device,
        )
    elif method == "add_noise_ae":
        encoder, _ = pretrain_autoencoder(
            splits.x_train, splits.x_val, input_dim, noise_type="gaussian",
            max_epochs=max_pretrain_epochs, device=device,
        )
    elif method == "scarf_disc":
        encoder, _ = pretrain_scarf_discriminative(
            splits.x_train, splits.x_val, input_dim,
            max_epochs=max_pretrain_epochs, device=device,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    encoder, head, _ = finetune_classifier(
        splits.x_train, splits.y_train, splits.x_val, splits.y_val,
        n_classes=splits.n_classes, encoder=encoder, input_dim=input_dim,
        max_epochs=max_finetune_epochs, device=device,
    )
    test_acc = evaluate_accuracy(encoder, head, splits.x_test, splits.y_test, device=device)
    return test_acc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-ids", type=int, nargs="+", default=list(DEFAULT_DATASET_IDS.keys()))
    parser.add_argument("--methods", nargs="+", default=["control", "scarf"], choices=ALL_METHODS)
    parser.add_argument("--n-trials", type=int, default=10, help="paper uses 30; reduce for speed")
    parser.add_argument("--max-pretrain-epochs", type=int, default=200, help="paper uses 1000 (patience=3 makes this a ceiling, not a target)")
    parser.add_argument("--max-finetune-epochs", type=int, default=100, help="paper uses 200")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # results[method][dataset_id] = np.array of per-trial test accuracies
    results = defaultdict(lambda: defaultdict(list))

    for dataset_id in args.dataset_ids:
        name = DEFAULT_DATASET_IDS.get(dataset_id, str(dataset_id))
        print(f"\n=== Dataset {dataset_id} ({name}) ===")
        t0 = time.time()
        X, y, categorical_mask, _ = load_openml_dataset(dataset_id)
        print(f"  loaded: {X.shape[0]} rows x {X.shape[1]} raw features, {y.nunique()} classes "
              f"({time.time() - t0:.1f}s)")

        for trial in range(args.n_trials):
            splits = preprocess_dataset(X, y, categorical_mask, scale="zscore", random_state=args.seed + trial)
            for method in args.methods:
                t1 = time.time()
                acc = run_one_trial(
                    splits, method, args.device,
                    args.max_pretrain_epochs, args.max_finetune_epochs, args.seed + trial,
                )
                results[method][dataset_id].append(acc)
                print(f"  trial {trial+1}/{args.n_trials} | {method:12s} | test_acc={acc:.4f} ({time.time()-t1:.1f}s)")

    # Convert to numpy arrays and save raw results.
    results_np = {m: {d: np.array(v) for d, v in dd.items()} for m, dd in results.items()}
    raw_path = os.path.join(args.output_dir, "raw_results.json")
    with open(raw_path, "w") as f:
        json.dump({m: {str(d): v.tolist() for d, v in dd.items()} for m, dd in results_np.items()}, f, indent=2)
    print(f"\nSaved raw per-trial results to {raw_path}")

    # Summary table: mean +/- std accuracy per (dataset, method).
    rows = []
    for method, dd in results_np.items():
        for dataset_id, accs in dd.items():
            rows.append({
                "dataset_id": dataset_id,
                "dataset_name": DEFAULT_DATASET_IDS.get(dataset_id, str(dataset_id)),
                "method": method,
                "mean_acc": accs.mean(),
                "std_acc": accs.std(),
                "n_trials": len(accs),
            })
    summary = pd.DataFrame(rows).sort_values(["dataset_id", "method"])
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary table to {summary_path}")
    print("\n" + summary.to_string(index=False))

    # Win matrix across methods (if >= 2 methods were run).
    if len(args.methods) >= 2:
        wm = win_matrix(results_np)
        wm_path = os.path.join(args.output_dir, "win_matrix.csv")
        wm.to_csv(wm_path)
        print(f"\nWin matrix saved to {wm_path}")
        print(wm)

    # Relative gain of every non-control method vs control, if control was run.
    if "control" in results_np:
        for method in args.methods:
            if method == "control":
                continue
            gains = relative_gain(results_np[method], results_np["control"])
            print(f"\nRelative gain of {method} over control (datasets with p<0.20 diff):")
            print(gains)
            if len(gains):
                print(f"  average: {gains.mean():.2f}%")


if __name__ == "__main__":
    main()
