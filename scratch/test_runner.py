"""
Unit test for extended experiment_runner.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scarf"))

import pandas as pd
from sklearn.datasets import load_wine
from scarf.data import preprocess_dataset, make_semi_supervised
from scarf.experiment_runner import run_all_combinations

d = load_wine()
X = pd.DataFrame(d.data, columns=d.feature_names)
y = pd.Series(d.target)
splits = preprocess_dataset(X, y, scale="zscore", random_state=42)
lab_idx, unlab_idx = make_semi_supervised(splits, labeled_frac=0.25, random_state=42)

res = run_all_combinations(
    x_pretrain=splits.x_train,
    x_train_ft=splits.x_train[lab_idx],
    y_train_ft=splits.y_train[lab_idx],
    x_val=splits.x_val,
    y_val=splits.y_val,
    x_test=splits.x_test,
    y_test=splits.y_test,
    n_classes=splits.n_classes,
    input_dim=splits.x_train.shape[1],
    x_unlabeled=splits.x_train[unlab_idx],
    pretrain_methods=["none", "scarf", "scarf_ae"],
    reference_methods=["control", "dropout", "self_train"],
    max_pretrain_epochs=3,
    max_finetune_epochs=3,
)

print("Runner executed successfully!")
for k, v in sorted(res.items()):
    print(f"  {k:25s}: {v:.4f}")
