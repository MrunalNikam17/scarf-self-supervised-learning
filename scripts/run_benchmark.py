"""
Run the full SCARF benchmark on OpenML-CC18 datasets matching the paper's experiments:
  - Setting 1: Fully-supervised (100% labeled training data)
  - Setting 2: Semi-supervised (25% labeled training data, 75% unlabeled)
  - Setting 3: Label-noise (30% uniform label noise on training data)

Evaluates all pre-training methods (none, SCARF, SCARF-AE, no-noise AE, add-noise AE, SCARF-disc)
combined with all reference recipes (control, dropout, mixup, label_smooth, distill,
self_train, tri_train, deep_knn, bitempered) and produces:
  - raw_results.json
  - summary.csv (per-dataset mean +/- std test accuracy)
  - win_matrix.csv (pairwise Welch's t-test p<0.05 win ratio)
  - relative_gain_table.csv (Table 1 reproduction: relative gain % with p<0.20 filter)

Usage:
    python scripts/run_benchmark.py --setting supervised --dataset-ids 11 37 54 1510 1494 15 --n-trials 10
    python scripts/run_benchmark.py --setting semi_supervised --n-trials 10
    python scripts/run_benchmark.py --setting label_noise --n-trials 10
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

from scarf.data import (
    corrupt_labels,
    load_csv_dataset,
    load_openml_dataset,
    make_semi_supervised,
    preprocess_dataset,
)
from scarf.evaluate import average_relative_gain_table, win_matrix
from scarf.experiment_runner import (
    ALL_PRETRAIN_METHODS,
    DEFAULT_PRETRAIN_METHODS,
    LABEL_NOISE_REFERENCE_METHODS,
    SEMI_SUPERVISED_REFERENCE_METHODS,
    SUPERVISED_REFERENCE_METHODS,
    run_all_combinations,
)

DEFAULT_DATASET_IDS = {
    11: "balance-scale",
    37: "diabetes",
    54: "vehicle",
    1510: "wdbc",
    1494: "qsar-biodeg",
    15: "breast-w",
}

LOCAL_FALLBACKS = {
    11: "wheat-seeds.csv",  # balance-scale fallback if offline
    37: "pima-indians-diabetes.csv",
    54: "vehicle.csv",
    1510: "breast-cancer-wisconsin.csv",
    1494: "sonar.csv",
    15: "breast-cancer-wisconsin.csv",
}

DATA_DIR = os.path.join(os.path.dirname(HERE), "data_cache")


def load_dataset_auto(dataset_id: int):
    """Load dataset from OpenML, falling back to cached CSV if necessary."""
    try:
        X, y, cat_mask, feat_names = load_openml_dataset(dataset_id)
        name = DEFAULT_DATASET_IDS.get(dataset_id, str(dataset_id))
        return X, y, cat_mask, f"openml_{dataset_id}_{name}"
    except Exception as e:
        print(f"  [Notice] OpenML load failed for {dataset_id} ({e}); trying local fallback...")
        fallback_fname = LOCAL_FALLBACKS.get(dataset_id, "wine.csv")
        fpath = os.path.join(DATA_DIR, fallback_fname)
        X, y = load_csv_dataset(fpath)
        return X, y, None, f"local_{fallback_fname.split('.')[0]}"


def run_benchmark_for_setting(
    setting: str,
    dataset_ids: list[int],
    n_trials: int,
    pretrain_methods: list[str],
    reference_methods: list[str],
    labeled_frac: float,
    noise_frac: float,
    max_pretrain_epochs: int,
    max_finetune_epochs: int,
    output_dir: str,
    device: str,
    seed: int = 0,
):
    print(f"\n=================================================================")
    print(f" Running Benchmark Setting: {setting.upper()}")
    print(f" Pretrain methods: {pretrain_methods}")
    print(f" Reference methods: {reference_methods}")
    print(f"=================================================================")

    setting_dir = os.path.join(output_dir, setting)
    os.makedirs(setting_dir, exist_ok=True)

    results = defaultdict(lambda: defaultdict(list))

    for did in dataset_ids:
        t_d = time.time()
        X, y, cat_mask, dname = load_dataset_auto(did)
        print(f"\n=== Dataset {did} ({dname}): {X.shape[0]} rows x {X.shape[1]} cols, {y.nunique()} classes ===")

        for trial in range(n_trials):
            trial_seed = seed + trial
            splits = preprocess_dataset(X, y, cat_mask, scale="zscore", random_state=trial_seed)
            input_dim = splits.x_train.shape[1]

            if setting == "supervised":
                x_pre = splits.x_train
                x_ft = splits.x_train
                y_ft = splits.y_train
                x_unlab = None
            elif setting == "semi_supervised":
                x_pre = splits.x_train
                lab_idx, unlab_idx = make_semi_supervised(splits, labeled_frac=labeled_frac, random_state=trial_seed)
                x_ft = splits.x_train[lab_idx]
                y_ft = splits.y_train[lab_idx]
                x_unlab = splits.x_train[unlab_idx]
            elif setting == "label_noise":
                x_pre = splits.x_train
                x_ft = splits.x_train
                y_ft = corrupt_labels(splits.y_train, splits.n_classes, noise_frac=noise_frac, random_state=trial_seed)
                x_unlab = None
            else:
                raise ValueError(f"Unknown setting: {setting}")

            t0 = time.time()
            trial_results = run_all_combinations(
                x_pretrain=x_pre,
                x_train_ft=x_ft,
                y_train_ft=y_ft,
                x_val=splits.x_val,
                y_val=splits.y_val,
                x_test=splits.x_test,
                y_test=splits.y_test,
                n_classes=splits.n_classes,
                input_dim=input_dim,
                x_unlabeled=x_unlab,
                device=device,
                max_pretrain_epochs=max_pretrain_epochs,
                max_finetune_epochs=max_finetune_epochs,
                reference_methods=reference_methods,
                pretrain_methods=pretrain_methods,
            )
            for k, acc in trial_results.items():
                results[k][dname].append(acc)

            elapsed = time.time() - t0
            summary_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(trial_results.items())[:4])
            print(f"  trial {trial+1}/{n_trials} ({elapsed:.1f}s): {summary_str}...")

        print(f"  Dataset total: {time.time()-t_d:.1f}s")

    _save_and_report_setting(results, setting_dir, pretrain_methods, reference_methods, root_dir=output_dir)
    return results


def _save_and_report_setting(results, out_dir, pretrain_methods, reference_methods, root_dir=None):
    results_np = {m: {d: np.array(v) for d, v in dd.items()} for m, dd in results.items()}

    dirs_to_save = [out_dir]
    if root_dir and os.path.abspath(root_dir) != os.path.abspath(out_dir):
        dirs_to_save.append(root_dir)

    for d in dirs_to_save:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "raw_results.json"), "w") as f:
            json.dump(
                {m: {ds: v.tolist() for ds, v in dd.items()} for m, dd in results_np.items()},
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
    for d in dirs_to_save:
        summary.to_csv(os.path.join(d, "summary.csv"), index=False)
    print("\n" + summary.to_string(index=False))

    wm = win_matrix(results_np)
    for d in dirs_to_save:
        wm.to_csv(os.path.join(d, "win_matrix.csv"))
    print("\n=== Win matrix ===")
    print(wm)

    active_pretrain = [m for m in pretrain_methods if m != "none"]
    table = average_relative_gain_table(
        results_np,
        pretrain_methods=active_pretrain,
        reference_methods=reference_methods,
    )
    for d in dirs_to_save:
        table.to_csv(os.path.join(d, "relative_gain_table.csv"))
    print("\n=== Average relative gain (%) vs. reference alone -- Table 1 style ===")
    print(table)

    from scipy import stats
    print("\n=== Detailed Relative Gain Breakdown (alpha=0.20) ===")
    for ref in reference_methods:
        for pt in active_pretrain:
            combo_key = f"{ref}+{pt}"
            if ref not in results_np or combo_key not in results_np:
                continue
            print(f"\n--- {combo_key} vs {ref} ---")
            gains = []
            for ds in sorted(results_np[ref].keys()):
                acc_m = results_np[combo_key][ds]
                acc_r = results_np[ref][ds]
                t, p = stats.ttest_ind(acc_m, acc_r, equal_var=False)
                diff = 100.0 * (acc_m.mean() - acc_r.mean()) / acc_r.mean() if acc_r.mean() > 0 else 0.0
                status = "INCLUDED (p < 0.20)" if p < 0.20 else "EXCLUDED (p >= 0.20)"
                if p < 0.20:
                    gains.append(diff)
                print(f"  {ds:25s}: ref={acc_r.mean()*100:5.2f}%, combo={acc_m.mean()*100:5.2f}%, diff={diff:+6.2f}%, p={p:6.4f} -> {status}")
            avg_g = np.mean(gains) if gains else float('nan')
            print(f"  Average Relative Gain: {avg_g:.2f}% ({len(gains)}/{len(results_np[ref])} datasets included)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=["supervised", "semi_supervised", "label_noise", "all"], default="supervised")
    parser.add_argument("--dataset-ids", type=int, nargs="+", default=list(DEFAULT_DATASET_IDS.keys()))
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--labeled-frac", type=float, default=0.25)
    parser.add_argument("--noise-frac", type=float, default=0.30)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Combined list of methods, e.g. control scarf scarf_ae mixup label_smooth")
    parser.add_argument("--pretrain-methods", nargs="+", default=None)
    parser.add_argument("--reference-methods", nargs="+", default=None)
    parser.add_argument("--max-pretrain-epochs", type=int, default=200)
    parser.add_argument("--max-finetune-epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Parse methods if provided via --methods
    pretrain_methods = args.pretrain_methods
    reference_methods = args.reference_methods

    if args.methods:
        known_pretrain = {"none", "scarf", "scarf_ae", "no_noise_ae", "add_noise_ae", "scarf_disc"}
        known_reference = {
            "control", "dropout", "mixup", "label_smooth", "distill",
            "self_train", "tri_train", "deep_knn", "bitempered",
        }
        parsed_pretrain = [m for m in args.methods if m in known_pretrain]
        parsed_ref = [m for m in args.methods if m in known_reference]
        if parsed_pretrain:
            if "none" not in parsed_pretrain:
                parsed_pretrain = ["none"] + parsed_pretrain
            pretrain_methods = parsed_pretrain
        if parsed_ref:
            reference_methods = parsed_ref

    if pretrain_methods is None:
        pretrain_methods = ALL_PRETRAIN_METHODS

    settings_to_run = [args.setting] if args.setting != "all" else ["supervised", "semi_supervised", "label_noise"]

    for stg in settings_to_run:
        if reference_methods:
            ref_methods = reference_methods
        elif stg == "supervised":
            ref_methods = SUPERVISED_REFERENCE_METHODS
        elif stg == "semi_supervised":
            ref_methods = SEMI_SUPERVISED_REFERENCE_METHODS
        elif stg == "label_noise":
            ref_methods = LABEL_NOISE_REFERENCE_METHODS

        run_benchmark_for_setting(
            setting=stg,
            dataset_ids=args.dataset_ids,
            n_trials=args.n_trials,
            pretrain_methods=pretrain_methods,
            reference_methods=ref_methods,
            labeled_frac=args.labeled_frac,
            noise_frac=args.noise_frac,
            max_pretrain_epochs=args.max_pretrain_epochs,
            max_finetune_epochs=args.max_finetune_epochs,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
