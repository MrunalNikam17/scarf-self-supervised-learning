"""
Semi-supervised experiment (paper Section 4.3): only `LABELED_FRAC` of the
training split retains labels; the rest is unlabeled. SCARF / SCARF-AE
pre-train on the FULL training split (labels ignored), while every
fine-tuning recipe (control, mixup, label smoothing) only ever sees the
labeled subset. This mirrors the paper's claim that pre-training methods
can leverage the unlabeled remainder that plain supervised baselines can't.

Usage:
    python scripts/run_semi_supervised.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scarf.data import load_csv_dataset, make_semi_supervised, preprocess_dataset
from scarf.evaluate import average_relative_gain_table, win_matrix
from scarf.experiment_runner import (
    DEFAULT_PRETRAIN_METHODS,
    DEFAULT_REFERENCE_METHODS,
    run_all_combinations,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data_cache")

DATASETS = {
    "pima-diabetes": "pima-indians-diabetes.csv",
    "sonar": "sonar.csv",
    "ionosphere": "ionosphere.csv",
    "glass": "glass.csv",
    "wine": "wine.csv",
    "wheat-seeds": "wheat-seeds.csv",
}

LABELED_FRAC = 0.25  # matches the paper's 25%-labeled semi-supervised setting


def main():
    n_trials = 10
    max_pretrain_epochs = 300
    max_finetune_epochs = 150
    device = "cpu"
    out_dir = os.path.join(os.path.dirname(HERE), "results_semi_supervised")
    os.makedirs(out_dir, exist_ok=True)

    results = defaultdict(lambda: defaultdict(list))

    for name, fname in DATASETS.items():
        path = os.path.join(DATA_DIR, fname)
        X, y = load_csv_dataset(path)
        print(f"\n=== {name}: {X.shape[0]} rows x {X.shape[1]} raw features, "
              f"{y.nunique()} classes, {LABELED_FRAC:.0%} labeled ===")
        t_dataset = time.time()

        for trial in range(n_trials):
            splits = preprocess_dataset(X, y, scale="zscore", random_state=trial)
            input_dim = splits.x_train.shape[1]

            labeled_idx, unlabeled_idx = make_semi_supervised(
                splits, labeled_frac=LABELED_FRAC, random_state=trial
            )
            x_labeled = splits.x_train[labeled_idx]
            y_labeled = splits.y_train[labeled_idx]

            t0 = time.time()
            trial_results = run_all_combinations(
                x_pretrain=splits.x_train,   # full split (labeled + unlabeled), labels ignored
                x_train_ft=x_labeled,        # fine-tuning only ever sees the labeled subset
                y_train_ft=y_labeled,
                x_val=splits.x_val, y_val=splits.y_val,
                x_test=splits.x_test, y_test=splits.y_test,
                n_classes=splits.n_classes, input_dim=input_dim,
                device=device,
                max_pretrain_epochs=max_pretrain_epochs,
                max_finetune_epochs=max_finetune_epochs,
            )
            for key, acc in trial_results.items():
                results[key][name].append(acc)

            elapsed = time.time() - t0
            summary_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(trial_results.items()))
            print(f"  trial {trial+1}/{n_trials} ({elapsed:.1f}s, "
                  f"{len(labeled_idx)} labeled / {len(unlabeled_idx)} unlabeled): {summary_str}")

        print(f"  dataset total: {time.time()-t_dataset:.1f}s")

    _save_and_report(results, out_dir)


def _save_and_report(results, out_dir):
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
        pretrain_methods=[m for m in DEFAULT_PRETRAIN_METHODS if m != "none"],
        reference_methods=list(DEFAULT_REFERENCE_METHODS.keys()),
    )
    table.to_csv(os.path.join(out_dir, "relative_gain_table.csv"))
    print("\n=== Average relative gain (%) vs. reference alone -- Table 1 style ===")
    print(table)


if __name__ == "__main__":
    main()
