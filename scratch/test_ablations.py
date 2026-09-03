"""
Unit test for ablations:
  - corruption strategies: marginal, none, mean, gaussian, joint, missing_feature, feature_dropout
  - alternative losses: Barlow Twins, Alignment + Uniformity
  - validation metrics: loss vs error
  - co-training
  - data augmentation
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scarf"))

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from scarf.data import preprocess_dataset
from scarf.trainer import finetune_classifier, evaluate_accuracy
from scarf.ablations import (
    pretrain_ablation,
    train_co_training,
    train_data_augmentation,
)

def main():
    print("Testing ablations on Wine dataset...")
    d = load_wine()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(d.target)
    splits = preprocess_dataset(X, y, scale="zscore", random_state=42)
    input_dim = splits.x_train.shape[1]
    n_classes = splits.n_classes

    # 1. Corruption strategies
    strategies = ["marginal", "none", "mean", "gaussian", "joint", "missing_feature", "feature_dropout"]
    print("1. Testing corruption strategies...")
    for strat in strategies:
        enc, _ = pretrain_ablation(splits.x_train, splits.x_val, input_dim, corruption_strategy=strat, max_epochs=3)
        enc, head, _ = finetune_classifier(splits.x_train, splits.y_train, splits.x_val, splits.y_val, n_classes, encoder=enc, max_epochs=3)
        acc = evaluate_accuracy(enc, head, splits.x_test, splits.y_test)
        print(f"   Strategy {strat:16s} -> acc = {acc:.4f}")

    # 2. Alternative losses
    print("2. Testing alternative losses...")
    for loss_name in ["infonce", "barlow", "align_uniform"]:
        enc, _ = pretrain_ablation(splits.x_train, splits.x_val, input_dim, loss_type=loss_name, max_epochs=3)
        enc, head, _ = finetune_classifier(splits.x_train, splits.y_train, splits.x_val, splits.y_val, n_classes, encoder=enc, max_epochs=3)
        acc = evaluate_accuracy(enc, head, splits.x_test, splits.y_test)
        print(f"   Loss {loss_name:16s} -> acc = {acc:.4f}")

    # 3. Validation metric: error vs loss
    print("3. Testing validation metrics (error vs loss)...")
    for metric in ["loss", "error"]:
        enc, _ = pretrain_ablation(splits.x_train, splits.x_val, input_dim, early_stop_metric=metric, max_epochs=3)
        enc, head, _ = finetune_classifier(splits.x_train, splits.y_train, splits.x_val, splits.y_val, n_classes, encoder=enc, max_epochs=3)
        acc = evaluate_accuracy(enc, head, splits.x_test, splits.y_test)
        print(f"   Metric {metric:16s} -> acc = {acc:.4f}")

    # 4. Co-training
    print("4. Testing Co-training (lambda=0.1)...")
    enc, head, _ = train_co_training(splits.x_train, splits.y_train, splits.x_val, splits.y_val, n_classes, input_dim, lambda_cont=0.1, max_epochs=3)
    acc = evaluate_accuracy(enc, head, splits.x_test, splits.y_test)
    print(f"   Co-training acc = {acc:.4f}")

    # 5. Data augmentation
    print("5. Testing Data augmentation...")
    enc, head, _ = train_data_augmentation(splits.x_train, splits.y_train, splits.x_val, splits.y_val, n_classes, input_dim, max_epochs=3)
    acc = evaluate_accuracy(enc, head, splits.x_test, splits.y_test)
    print(f"   Data augmentation acc = {acc:.4f}")

    print("\nALL ABLATION UNIT TESTS PASSED!")

if __name__ == "__main__":
    main()
