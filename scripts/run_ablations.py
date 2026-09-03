"""
Script to run the SCARF Appendix ablation experiments:
  1. Corruption strategies (marginal, none, mean, gaussian, joint, missing_feature, feature_dropout)
     under both z-score and min-max feature scaling.
  2. Batch size sweep: {4, 16, 64, 128, 256, 512}.
  3. Corruption rate sweep: 10% to 90% in steps of 10%.
  4. Softmax temperature sweep: {0.01, 0.1, 1.0, 10.0}.
  5. Alternative contrastive losses: InfoNCE vs. Barlow Twins (5e-3) vs. Alignment+Uniformity (equal-weighted, t=2).
  6. Pre-training vs. Co-training: lambda in {0.01, 0.1, 1.0}.
  7. Pre-training vs. Data Augmentation.
  8. Validation metric for early stopping: InfoNCE loss vs. InfoNCE error (off-diagonal argmax).

Usage:
    python scripts/run_ablations.py --n-trials 5 --output-dir results_ablations/
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

from scarf.ablations import (
    pretrain_ablation,
    train_co_training,
    train_data_augmentation,
)
from scarf.data import load_csv_dataset, preprocess_dataset
from scarf.evaluate import relative_gain, win_matrix
from scarf.trainer import evaluate_accuracy, finetune_classifier, pretrain_scarf

DATA_DIR = os.path.join(os.path.dirname(HERE), "data_cache")
DEFAULT_DATASETS = {
    "wine": "wine.csv",
    "glass": "glass.csv",
    "sonar": "sonar.csv",
    "pima-diabetes": "pima-indians-diabetes.csv",
}


def run_corruption_ablation(splits_dict, input_dim, n_classes, n_trials, device, max_pt, max_ft):
    print("\n=======================================================")
    print(" Ablation 1: Corruption Strategies (z-score & min-max)")
    print("=======================================================")
    strategies = ["marginal", "none", "mean", "gaussian", "joint", "missing_feature", "feature_dropout"]
    scalings = ["zscore", "minmax"]
    results = defaultdict(lambda: defaultdict(list))

    for scaling in scalings:
        print(f"\n--- Scaling: {scaling} ---")
        for dname, splits in splits_dict[scaling].items():
            for trial in range(n_trials):
                trial_splits = splits[trial]
                inp_d = trial_splits.x_train.shape[1]
                for strat in strategies:
                    method_key = f"{strat}_{scaling}"
                    enc, _ = pretrain_ablation(
                        trial_splits.x_train, trial_splits.x_val, inp_d,
                        corruption_strategy=strat, max_epochs=max_pt, device=device,
                    )
                    enc, head, _ = finetune_classifier(
                        trial_splits.x_train, trial_splits.y_train, trial_splits.x_val, trial_splits.y_val,
                        n_classes=trial_splits.n_classes, encoder=enc, input_dim=inp_d,
                        max_epochs=max_ft, device=device,
                    )
                    acc = evaluate_accuracy(enc, head, trial_splits.x_test, trial_splits.y_test, device=device)
                    results[method_key][dname].append(acc)
    return results


def run_batch_size_ablation(splits_dict, n_trials, device, max_pt, max_ft):
    print("\n=======================================================")
    print(" Ablation 2: Batch Size Sweep {4, 16, 64, 128, 256, 512}")
    print("=======================================================")
    batch_sizes = [4, 16, 64, 128, 256, 512]
    results = defaultdict(lambda: defaultdict(list))

    for dname, splits in splits_dict["zscore"].items():
        for trial in range(n_trials):
            trial_splits = splits[trial]
            inp_d = trial_splits.x_train.shape[1]
            for bs in batch_sizes:
                key = f"batch_{bs}"
                enc, _ = pretrain_ablation(
                    trial_splits.x_train, trial_splits.x_val, inp_d,
                    corruption_strategy="marginal", batch_size=bs, max_epochs=max_pt, device=device,
                )
                enc, head, _ = finetune_classifier(
                    trial_splits.x_train, trial_splits.y_train, trial_splits.x_val, trial_splits.y_val,
                    n_classes=trial_splits.n_classes, encoder=enc, input_dim=inp_d,
                    max_epochs=max_ft, device=device,
                )
                acc = evaluate_accuracy(enc, head, trial_splits.x_test, trial_splits.y_test, device=device)
                results[key][dname].append(acc)
    return results


def run_corruption_rate_ablation(splits_dict, n_trials, device, max_pt, max_ft):
    print("\n=======================================================")
    print(" Ablation 3: Corruption Rate Sweep (0.1 to 0.9)")
    print("=======================================================")
    rates = [round(r, 1) for r in np.arange(0.1, 1.0, 0.1)]
    results = defaultdict(lambda: defaultdict(list))

    for dname, splits in splits_dict["zscore"].items():
        for trial in range(n_trials):
            trial_splits = splits[trial]
            inp_d = trial_splits.x_train.shape[1]
            for cr in rates:
                key = f"rate_{int(cr*100)}"
                enc, _ = pretrain_ablation(
                    trial_splits.x_train, trial_splits.x_val, inp_d,
                    corruption_strategy="marginal", corruption_rate=cr, max_epochs=max_pt, device=device,
                )
                enc, head, _ = finetune_classifier(
                    trial_splits.x_train, trial_splits.y_train, trial_splits.x_val, trial_splits.y_val,
                    n_classes=trial_splits.n_classes, encoder=enc, input_dim=inp_d,
                    max_epochs=max_ft, device=device,
                )
                acc = evaluate_accuracy(enc, head, trial_splits.x_test, trial_splits.y_test, device=device)
                results[key][dname].append(acc)
    return results


def run_temperature_ablation(splits_dict, n_trials, device, max_pt, max_ft):
    print("\n=======================================================")
    print(" Ablation 4: Softmax Temperature Sweep {0.01, 0.1, 1.0, 10.0}")
    print("=======================================================")
    temps = [0.01, 0.1, 1.0, 10.0]
    results = defaultdict(lambda: defaultdict(list))

    for dname, splits in splits_dict["zscore"].items():
        for trial in range(n_trials):
            trial_splits = splits[trial]
            inp_d = trial_splits.x_train.shape[1]
            for tau in temps:
                key = f"temp_{tau}"
                enc, _ = pretrain_ablation(
                    trial_splits.x_train, trial_splits.x_val, inp_d,
                    corruption_strategy="marginal", temperature=tau, max_epochs=max_pt, device=device,
                )
                enc, head, _ = finetune_classifier(
                    trial_splits.x_train, trial_splits.y_train, trial_splits.x_val, trial_splits.y_val,
                    n_classes=trial_splits.n_classes, encoder=enc, input_dim=inp_d,
                    max_epochs=max_ft, device=device,
                )
                acc = evaluate_accuracy(enc, head, trial_splits.x_test, trial_splits.y_test, device=device)
                results[key][dname].append(acc)
    return results


def run_losses_ablation(splits_dict, n_trials, device, max_pt, max_ft):
    print("\n=======================================================")
    print(" Ablation 5: Alternative Losses (InfoNCE, Barlow, Align+Uniform)")
    print("=======================================================")
    loss_types = ["infonce", "barlow", "align_uniform"]
    results = defaultdict(lambda: defaultdict(list))

    for dname, splits in splits_dict["zscore"].items():
        for trial in range(n_trials):
            trial_splits = splits[trial]
            inp_d = trial_splits.x_train.shape[1]
            for lt in loss_types:
                enc, _ = pretrain_ablation(
                    trial_splits.x_train, trial_splits.x_val, inp_d,
                    corruption_strategy="marginal", loss_type=lt, max_epochs=max_pt, device=device,
                )
                enc, head, _ = finetune_classifier(
                    trial_splits.x_train, trial_splits.y_train, trial_splits.x_val, trial_splits.y_val,
                    n_classes=trial_splits.n_classes, encoder=enc, input_dim=inp_d,
                    max_epochs=max_ft, device=device,
                )
                acc = evaluate_accuracy(enc, head, trial_splits.x_test, trial_splits.y_test, device=device)
                results[lt][dname].append(acc)
    return results


def run_cotrain_and_aug_ablation(splits_dict, n_trials, device, max_pt, max_ft):
    print("\n=======================================================")
    print(" Ablation 6-7: Pre-train vs Co-train vs Data Augmentation")
    print("=======================================================")
    results = defaultdict(lambda: defaultdict(list))

    for dname, splits in splits_dict["zscore"].items():
        for trial in range(n_trials):
            trial_splits = splits[trial]
            inp_d = trial_splits.x_train.shape[1]

            # 1. SCARF Pretraining
            enc, _ = pretrain_scarf(
                trial_splits.x_train, trial_splits.x_val, inp_d,
                max_epochs=max_pt, device=device,
            )
            enc, head, _ = finetune_classifier(
                trial_splits.x_train, trial_splits.y_train, trial_splits.x_val, trial_splits.y_val,
                n_classes=trial_splits.n_classes, encoder=enc, input_dim=inp_d,
                max_epochs=max_ft, device=device,
            )
            acc = evaluate_accuracy(enc, head, trial_splits.x_test, trial_splits.y_test, device=device)
            results["pretrain_scarf"][dname].append(acc)

            # 2. Co-training with lambda in {0.01, 0.1, 1.0} (matching Figure 9)
            for lam in [0.01, 0.1, 1.0]:
                key = f"cotrain_lam_{lam}"
                enc, head, _ = train_co_training(
                    trial_splits.x_train, trial_splits.y_train, trial_splits.x_val, trial_splits.y_val,
                    n_classes=trial_splits.n_classes, input_dim=inp_d, lambda_cont=lam,
                    max_epochs=max_ft, device=device,
                )
                acc = evaluate_accuracy(enc, head, trial_splits.x_test, trial_splits.y_test, device=device)
                results[key][dname].append(acc)

            # 3. Data Augmentation
            enc, head, _ = train_data_augmentation(
                trial_splits.x_train, trial_splits.y_train, trial_splits.x_val, trial_splits.y_val,
                n_classes=trial_splits.n_classes, input_dim=inp_d,
                max_epochs=max_ft, device=device,
            )
            acc = evaluate_accuracy(enc, head, trial_splits.x_test, trial_splits.y_test, device=device)
            results["data_augmentation"][dname].append(acc)

            # 4. Supervised Control
            enc, head, _ = finetune_classifier(
                trial_splits.x_train, trial_splits.y_train, trial_splits.x_val, trial_splits.y_val,
                n_classes=trial_splits.n_classes, encoder=None, input_dim=inp_d,
                max_epochs=max_ft, device=device,
            )
            acc = evaluate_accuracy(enc, head, trial_splits.x_test, trial_splits.y_test, device=device)
            results["control"][dname].append(acc)

    return results


def run_val_metric_ablation(splits_dict, n_trials, device, max_pt, max_ft):
    print("\n=======================================================")
    print(" Ablation 8: Validation Metric (InfoNCE Loss vs. Error)")
    print("=======================================================")
    metrics = ["loss", "error"]
    results = defaultdict(lambda: defaultdict(list))

    for dname, splits in splits_dict["zscore"].items():
        for trial in range(n_trials):
            trial_splits = splits[trial]
            inp_d = trial_splits.x_train.shape[1]
            for met in metrics:
                key = f"earlystop_{met}"
                enc, _ = pretrain_ablation(
                    trial_splits.x_train, trial_splits.x_val, inp_d,
                    corruption_strategy="marginal", early_stop_metric=met, max_epochs=max_pt, device=device,
                )
                enc, head, _ = finetune_classifier(
                    trial_splits.x_train, trial_splits.y_train, trial_splits.x_val, trial_splits.y_val,
                    n_classes=trial_splits.n_classes, encoder=enc, input_dim=inp_d,
                    max_epochs=max_ft, device=device,
                )
                acc = evaluate_accuracy(enc, head, trial_splits.x_test, trial_splits.y_test, device=device)
                results[key][dname].append(acc)
    return results


def _summarize_and_save(results_dict, out_path, name):
    results_np = {m: {d: np.array(v) for d, v in dd.items()} for m, dd in results_dict.items()}
    rows = []
    for method, dd in results_np.items():
        for dataset, accs in dd.items():
            rows.append({
                "ablation": name,
                "dataset": dataset,
                "variant": method,
                "mean_acc": accs.mean(),
                "std_acc": accs.std(),
                "n_trials": len(accs),
            })
    df = pd.DataFrame(rows).sort_values(["variant", "dataset"])
    df.to_csv(out_path, index=False)
    print(f"\nSaved {name} summary to {out_path}")
    print(df.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--max-pretrain-epochs", type=int, default=150)
    parser.add_argument("--max-finetune-epochs", type=int, default=100)
    parser.add_argument("--output-dir", default="results_ablations")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(os.path.dirname(HERE), args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Pre-generate dataset splits across scalings and trials
    print("Pre-generating dataset splits...")
    splits_dict = {"zscore": defaultdict(list), "minmax": defaultdict(list)}
    for name, fname in DEFAULT_DATASETS.items():
        fpath = os.path.join(DATA_DIR, fname)
        X, y = load_csv_dataset(fpath)
        for trial in range(args.n_trials):
            splits_dict["zscore"][name].append(preprocess_dataset(X, y, scale="zscore", random_state=trial))
            splits_dict["minmax"][name].append(preprocess_dataset(X, y, scale="minmax", random_state=trial))

    all_dfs = []

    # 1. Corruption strategies
    res1 = run_corruption_ablation(splits_dict, 0, 0, args.n_trials, args.device, args.max_pretrain_epochs, args.max_finetune_epochs)
    all_dfs.append(_summarize_and_save(res1, os.path.join(args.output_dir, "ablation_corruptions.csv"), "corruption_strategies"))

    # 2. Batch size
    res2 = run_batch_size_ablation(splits_dict, args.n_trials, args.device, args.max_pretrain_epochs, args.max_finetune_epochs)
    all_dfs.append(_summarize_and_save(res2, os.path.join(args.output_dir, "ablation_batch_size.csv"), "batch_size"))

    # 3. Corruption rate
    res3 = run_corruption_rate_ablation(splits_dict, args.n_trials, args.device, args.max_pretrain_epochs, args.max_finetune_epochs)
    all_dfs.append(_summarize_and_save(res3, os.path.join(args.output_dir, "ablation_corruption_rate.csv"), "corruption_rate"))

    # 4. Temperature
    res4 = run_temperature_ablation(splits_dict, args.n_trials, args.device, args.max_pretrain_epochs, args.max_finetune_epochs)
    all_dfs.append(_summarize_and_save(res4, os.path.join(args.output_dir, "ablation_temperature.csv"), "temperature"))

    # 5. Losses
    res5 = run_losses_ablation(splits_dict, args.n_trials, args.device, args.max_pretrain_epochs, args.max_finetune_epochs)
    all_dfs.append(_summarize_and_save(res5, os.path.join(args.output_dir, "ablation_losses.csv"), "alternative_losses"))

    # 6. Co-train and data augmentation
    res6 = run_cotrain_and_aug_ablation(splits_dict, args.n_trials, args.device, args.max_pretrain_epochs, args.max_finetune_epochs)
    all_dfs.append(_summarize_and_save(res6, os.path.join(args.output_dir, "ablation_cotrain_aug.csv"), "cotrain_and_augmentation"))

    # 7. Validation metric
    res7 = run_val_metric_ablation(splits_dict, args.n_trials, args.device, args.max_pretrain_epochs, args.max_finetune_epochs)
    all_dfs.append(_summarize_and_save(res7, os.path.join(args.output_dir, "ablation_val_metric.csv"), "validation_metric"))

    # Unified master summary
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_path = os.path.join(args.output_dir, "all_ablations_summary.csv")
    master_df.to_csv(master_path, index=False)
    print(f"\nAll ablations completed! Saved unified summary to {master_path}")


if __name__ == "__main__":
    main()
