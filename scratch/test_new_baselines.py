"""
Unit test for new baselines:
  - train_bitempered
  - train_deep_knn
  - train_self_distillation
  - train_self_training
  - train_tri_training
  - dropout fine-tuning
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scarf"))

import numpy as np
from sklearn.datasets import load_wine
from scarf.data import preprocess_dataset, make_semi_supervised, corrupt_labels
from scarf.trainer import pretrain_scarf, finetune_classifier, evaluate_accuracy
from scarf.semisup_baselines import (
    BiTemperedLogisticLoss,
    train_bitempered,
    train_deep_knn,
    train_self_distillation,
    train_self_training,
    train_tri_training,
)

def main():
    print("Testing new baselines on Wine dataset...")
    d = load_wine()
    import pandas as pd
    X = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(d.target)
    splits = preprocess_dataset(X, y, scale="zscore", random_state=42)
    input_dim = splits.x_train.shape[1]
    n_classes = splits.n_classes

    # Pre-train a quick encoder to test pre-trained interaction
    print("Pretraining a test SCARF encoder (5 epochs)...")
    enc_pretrained, _ = pretrain_scarf(splits.x_train, splits.x_val, input_dim, max_epochs=5)

    # 1. Test Dropout fine-tuning
    print("1. Testing Dropout fine-tuning (dropout_rate=0.04)...")
    enc, head, _ = finetune_classifier(
        splits.x_train, splits.y_train, splits.x_val, splits.y_val,
        n_classes=n_classes, encoder=enc_pretrained, dropout_rate=0.04, max_epochs=5,
    )
    acc = evaluate_accuracy(enc, head, splits.x_test, splits.y_test)
    print(f"   Dropout accuracy: {acc:.4f}")

    # 2. Test Bi-Tempered Loss
    print("2. Testing Bi-Tempered Logistic Loss...")
    enc, head, _ = train_bitempered(
        splits.x_train, splits.y_train, splits.x_val, splits.y_val,
        n_classes=n_classes, encoder=enc_pretrained, max_epochs=5,
    )
    acc = evaluate_accuracy(enc, head, splits.x_test, splits.y_test)
    print(f"   Bi-tempered accuracy: {acc:.4f}")

    # 3. Test Deep k-NN (with label noise)
    print("3. Testing Deep k-NN (filtering)...")
    y_noisy = corrupt_labels(splits.y_train, n_classes, noise_frac=0.3, random_state=42)
    enc, head, _ = train_deep_knn(
        splits.x_train, y_noisy, splits.x_val, splits.y_val,
        n_classes=n_classes, encoder=enc_pretrained, k=5, max_epochs=5,
    )
    acc = evaluate_accuracy(enc, head, splits.x_test, splits.y_test)
    print(f"   Deep k-NN accuracy: {acc:.4f}")

    # 4. Test Self-Distillation (with semi-supervised data)
    print("4. Testing Self-Distillation...")
    lab_idx, unlab_idx = make_semi_supervised(splits, labeled_frac=0.25, random_state=42)
    enc, head, _ = train_self_distillation(
        splits.x_train[lab_idx], splits.y_train[lab_idx], splits.x_val, splits.y_val,
        n_classes=n_classes, x_unlabeled=splits.x_train[unlab_idx],
        encoder=enc_pretrained, max_epochs=5,
    )
    acc = evaluate_accuracy(enc, head, splits.x_test, splits.y_test)
    print(f"   Self-Distillation accuracy: {acc:.4f}")

    # 5. Test Self-Training
    print("5. Testing Self-Training...")
    enc, head, _ = train_self_training(
        splits.x_train[lab_idx], splits.y_train[lab_idx], splits.x_val, splits.y_val,
        n_classes=n_classes, x_unlabeled=splits.x_train[unlab_idx],
        encoder=enc_pretrained, threshold=0.75, n_iterations=3, max_epochs=5,
    )
    acc = evaluate_accuracy(enc, head, splits.x_test, splits.y_test)
    print(f"   Self-Training accuracy: {acc:.4f}")

    # 6. Test Tri-Training
    print("6. Testing Tri-Training...")
    enc, head, _ = train_tri_training(
        splits.x_train[lab_idx], splits.y_train[lab_idx], splits.x_val, splits.y_val,
        n_classes=n_classes, x_unlabeled=splits.x_train[unlab_idx],
        encoder=enc_pretrained, n_iterations=3, max_epochs=5,
    )
    acc = evaluate_accuracy(enc, head, splits.x_test, splits.y_test)
    print(f"   Tri-Training accuracy: {acc:.4f}")

    print("\nALL 6 NEW BASELINE TESTS PASSED!")

if __name__ == "__main__":
    main()
