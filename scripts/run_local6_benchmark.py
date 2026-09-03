"""
Run SCARF vs. control (and SCARF-AE) on 6 real small/medium tabular
classification datasets, using local CSV copies (useful when OpenML's API
isn't reachable). For the real OpenML-CC18 benchmark, use run_benchmark.py
instead once you have network access to openml.org.
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

from scarf.baselines import pretrain_autoencoder
from scarf.data import load_csv_dataset, preprocess_dataset
from scarf.evaluate import relative_gain, win_matrix
from scarf.trainer import evaluate_accuracy, finetune_classifier, pretrain_scarf

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

METHODS = ["control", "scarf", "scarf_ae"]


def run_one(splits, method, device, max_pretrain_epochs, max_finetune_epochs):
    input_dim = splits.x_train.shape[1]
    if method == "control":
        encoder = None
    elif method == "scarf":
        encoder, _ = pretrain_scarf(splits.x_train, splits.x_val, input_dim,
                                     max_epochs=max_pretrain_epochs, device=device)
    elif method == "scarf_ae":
        encoder, _ = pretrain_autoencoder(splits.x_train, splits.x_val, input_dim,
                                           noise_type="scarf", max_epochs=max_pretrain_epochs, device=device)
    else:
        raise ValueError(method)

    encoder, head, _ = finetune_classifier(
        splits.x_train, splits.y_train, splits.x_val, splits.y_val,
        n_classes=splits.n_classes, encoder=encoder, input_dim=input_dim,
        max_epochs=max_finetune_epochs, device=device,
    )
    return evaluate_accuracy(encoder, head, splits.x_test, splits.y_test, device=device)


def main():
    n_trials = 10
    max_pretrain_epochs = 300
    max_finetune_epochs = 150
    device = "cpu"
    out_dir = os.path.join(os.path.dirname(HERE), "results_local6")
    os.makedirs(out_dir, exist_ok=True)

    results = defaultdict(lambda: defaultdict(list))

    for name, fname in DATASETS.items():
        path = os.path.join(DATA_DIR, fname)
        X, y = load_csv_dataset(path)
        print(f"\n=== {name}: {X.shape[0]} rows x {X.shape[1]} raw features, {y.nunique()} classes ===")
        t_dataset = time.time()
        for trial in range(n_trials):
            splits = preprocess_dataset(X, y, scale="zscore", random_state=trial)
            for method in METHODS:
                t0 = time.time()
                acc = run_one(splits, method, device, max_pretrain_epochs, max_finetune_epochs)
                results[method][name].append(acc)
                print(f"  trial {trial+1}/{n_trials} | {method:9s} | acc={acc:.4f} ({time.time()-t0:.1f}s)")
        print(f"  dataset total: {time.time()-t_dataset:.1f}s")

    results_np = {m: {d: np.array(v) for d, v in dd.items()} for m, dd in results.items()}

    with open(os.path.join(out_dir, "raw_results.json"), "w") as f:
        json.dump({m: {d: v.tolist() for d, v in dd.items()} for m, dd in results_np.items()}, f, indent=2)

    rows = []
    for method, dd in results_np.items():
        for dataset, accs in dd.items():
            rows.append({"dataset": dataset, "method": method, "mean_acc": accs.mean(),
                         "std_acc": accs.std(), "n_trials": len(accs)})
    summary = pd.DataFrame(rows).sort_values(["dataset", "method"])
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print("\n" + summary.to_string(index=False))

    wm = win_matrix(results_np)
    wm.to_csv(os.path.join(out_dir, "win_matrix.csv"))
    print("\n=== Win matrix ===")
    print(wm)

    print("\n=== Relative gain vs control ===")
    for method in METHODS:
        if method == "control":
            continue
        gains = relative_gain(results_np[method], results_np["control"])
        print(f"{method}: {gains.to_dict()}")
        if len(gains):
            print(f"  average: {gains.mean():.2f}%")


if __name__ == "__main__":
    main()
