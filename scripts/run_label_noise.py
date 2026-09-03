"""
Label-noise experiment (paper Section 4.2): `NOISE_FRAC` of TRAINING labels
are corrupted to a uniformly random class (including, possibly, the true
class); validation and test labels stay clean. SCARF / SCARF-AE pre-train on
the training features (labels aren't used for pre-training anyway, so noise
doesn't affect that stage) and are then fine-tuned on the noisy-labeled data,
comparing standard fine-tuning against robust label noise baselines (Deep k-NN,
Bi-tempered loss, distillation, etc.).

Usage:
    python scripts/run_label_noise.py --n-trials 10
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scarf"))

from scarf.data import corrupt_labels, load_csv_dataset, preprocess_dataset
from scarf.evaluate import average_relative_gain_table, win_matrix
from scarf.experiment_runner import (
    ALL_PRETRAIN_METHODS,
    DEFAULT_PRETRAIN_METHODS,
    DEFAULT_REFERENCE_METHODS,
    LABEL_NOISE_REFERENCE_METHODS,
    run_all_combinations,
)

DATA_DIR = os.path.join(os.path.dirname(HERE), "data_cache")

DATASETS = {
    "pima-diabetes": "pima-indians-diabetes.csv",
    "sonar": "sonar.csv",
    "ionosphere": "ionosphere.csv",
    "glass": "glass.csv",
    "wine": "wine.csv",
    "wheat-seeds": "wheat-seeds.csv",
}

DEFAULT_NOISE_FRAC = 0.30  # matches the paper's 30% label-noise setting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--noise-frac", type=float, default=DEFAULT_NOISE_FRAC)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--pretrain-methods", nargs="+", default=DEFAULT_PRETRAIN_METHODS, choices=ALL_PRETRAIN_METHODS)
    parser.add_argument("--reference-methods", nargs="+", default=None, choices=LABEL_NOISE_REFERENCE_METHODS)
    parser.add_argument("--max-pretrain-epochs", type=int, default=300)
    parser.add_argument("--max-finetune-epochs", type=int, default=150)
    parser.add_argument("--output-dir", default="results_label_noise")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ref_methods = args.reference_methods or LABEL_NOISE_REFERENCE_METHODS
    os.makedirs(args.output_dir, exist_ok=True)

    results = defaultdict(lambda: defaultdict(list))

    selected_datasets = {k: DATASETS[k] for k in args.datasets if k in DATASETS}

    for name, fname in selected_datasets.items():
        path = os.path.join(DATA_DIR, fname)
        X, y = load_csv_dataset(path)
        print(f"\n=== {name}: {X.shape[0]} rows x {X.shape[1]} raw features, "
              f"{y.nunique()} classes, {args.noise_frac:.0%} label noise ===")
        t_dataset = time.time()

        for trial in range(args.n_trials):
            splits = preprocess_dataset(X, y, scale="zscore", random_state=trial)
            input_dim = splits.x_train.shape[1]

            # Only training labels are corrupted; validation/test stay clean
            y_train_noisy = corrupt_labels(
                splits.y_train, splits.n_classes, noise_frac=args.noise_frac, random_state=trial
            )
            n_flipped = int(np.sum(y_train_noisy != splits.y_train))

            t0 = time.time()
            trial_results = run_all_combinations(
                x_pretrain=splits.x_train,        # pre-training never sees labels anyway
                x_train_ft=splits.x_train,        # fine-tuning sees full noisy training set
                y_train_ft=y_train_noisy,
                x_val=splits.x_val, y_val=splits.y_val,       # clean
                x_test=splits.x_test, y_test=splits.y_test,   # clean
                n_classes=splits.n_classes, input_dim=input_dim,
                device=args.device,
                max_pretrain_epochs=args.max_pretrain_epochs,
                max_finetune_epochs=args.max_finetune_epochs,
                reference_methods=ref_methods,
                pretrain_methods=args.pretrain_methods,
            )
            for key, acc in trial_results.items():
                results[key][name].append(acc)

            elapsed = time.time() - t0
            summary_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(trial_results.items()))
            print(f"  trial {trial+1}/{args.n_trials} ({elapsed:.1f}s, "
                  f"{n_flipped}/{len(y_train_noisy)} labels flipped): {summary_str}")

        print(f"  dataset total: {time.time()-t_dataset:.1f}s")

    _save_and_report(results, args.output_dir, args.pretrain_methods, ref_methods)


def _save_and_report(results, out_dir, pretrain_methods, reference_methods):
    results_np = {m: {d: np.array(v) for d, v in dd.items()} for m, dd in results.items()}

    with open(os.path.join(out_dir, "raw_results.json"), "w") as f:
        json.dump(
            {m: {d: v.tolist() for d, v in dd.items()} for m, dd in results_np.items()},
            f, indent=2,
        )

    rows = []
    for method, dd in results_np.items():
        for dataset, accs in dd.items():
            rows.append({
                "dataset": dataset, "method": method,
                "mean_acc": accs.mean(), "std_acc": accs.std(), "n_trials": len(accs),
            })
    summary = pd.DataFrame(rows).sort_values(["dataset", "method"])
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print("\n" + summary.to_string(index=False))

    wm = win_matrix(results_np)
    wm.to_csv(os.path.join(out_dir, "win_matrix.csv"))
    print("\n=== Win matrix ===")
    print(wm)

    table = average_relative_gain_table(
        results_np,
        pretrain_methods=[m for m in pretrain_methods if m != "none"],
        reference_methods=reference_methods,
    )
    table.to_csv(os.path.join(out_dir, "relative_gain_table.csv"))
    print("\n=== Average relative gain (%) vs. reference alone -- Table 1 style ===")
    print(table)


if __name__ == "__main__":
    main()
