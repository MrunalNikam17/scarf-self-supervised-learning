"""
Data loading and pre-processing for OpenML-CC18-style tabular classification
datasets, following the paper's description:

  - categorical features -> one-hot encoded
  - missing categorical entries -> filled with the mode
  - missing numerical entries -> filled with the mean
  - columns that are always missing -> dropped
  - numerical features -> z-score normalized (default; paper found z-score
    best for all but 3 datasets, which we allow overriding via `scale=None`)
  - 70% / 10% / 20% train / validation / test split
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


@dataclass
class DatasetSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    feature_names: list


class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


def load_openml_dataset(dataset_id: int):
    """Fetch a raw (X, y) pandas DataFrame/Series pair from OpenML by dataset id.

    Requires the `openml` package and network access to openml.org (not
    available inside this sandbox, but works in a normal environment with
    internet access). Install with: pip install openml
    """
    import openml

    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_mask, feature_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    return X, y, categorical_mask, feature_names


def load_csv_dataset(path: str, label_col: int = -1, header: Optional[int] = None, na_values=("?",)):
    """Load a local CSV dataset into (X, y) for preprocess_dataset(). Used for
    datasets fetched from a plain CSV mirror (e.g. when OpenML's API isn't
    reachable). Assumes the label is a single column (default: the last one)
    and all other columns are features; dtypes are inferred by pandas so
    numeric vs. categorical columns are detected automatically.
    """
    df = pd.read_csv(path, header=header, na_values=list(na_values))
    df.columns = [str(c) for c in df.columns]
    label_name = df.columns[label_col]
    y = df[label_name]
    X = df.drop(columns=[label_name])
    return X, y


def preprocess_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_mask: Optional[list] = None,
    scale: Optional[str] = "zscore",
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 0,
) -> DatasetSplits:
    """Turn a raw (X, y) pair into train/val/test numpy splits, following the
    paper's preprocessing recipe.

    scale: "zscore", "minmax", "mean", or None.
    """
    X = X.copy()

    if categorical_mask is None:
        categorical_mask = [not pd.api.types.is_numeric_dtype(X[c]) for c in X.columns]

    # Drop columns that are always missing.
    always_missing = [c for c in X.columns if X[c].isna().all()]
    if always_missing:
        X = X.drop(columns=always_missing)
        categorical_mask = [m for c, m in zip(X.columns, categorical_mask) if c not in always_missing]

    cat_cols = [c for c, is_cat in zip(X.columns, categorical_mask) if is_cat]
    num_cols = [c for c, is_cat in zip(X.columns, categorical_mask) if not is_cat]

    # Impute missing values: mode for categorical, mean for numerical.
    for c in cat_cols:
        if X[c].isna().any():
            mode = X[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "missing"
            X[c] = X[c].fillna(fill)
    for c in num_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].mean())

    # One-hot encode categoricals.
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)

    feature_names = list(X.columns)
    X_arr = X.to_numpy(dtype=np.float64)

    # Encode labels to 0..K-1.
    le = LabelEncoder()
    y_arr = le.fit_transform(y)
    n_classes = len(le.classes_)

    # 70/10/20 train/val/test split.
    x_train, x_temp, y_train, y_temp = train_test_split(
        X_arr, y_arr, test_size=(val_size + test_size), random_state=random_state, stratify=y_arr
    )
    relative_test = test_size / (val_size + test_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=relative_test, random_state=random_state, stratify=y_temp
    )

    # Scale numeric (now one-hot expanded) features using train statistics only.
    # We scale all columns since one-hot columns are already in {0, 1} and
    # z-score/min-max leave near-binary columns reasonably well-behaved,
    # matching the paper's practice of scaling the full one-hot representation.
    if scale is not None:
        if scale == "zscore":
            mean = x_train.mean(axis=0, keepdims=True)
            std = x_train.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            x_train = (x_train - mean) / std
            x_val = (x_val - mean) / std
            x_test = (x_test - mean) / std
        elif scale == "minmax":
            mn = x_train.min(axis=0, keepdims=True)
            mx = x_train.max(axis=0, keepdims=True)
            rng = mx - mn
            rng[rng == 0] = 1.0
            x_train = (x_train - mn) / rng
            x_val = (x_val - mn) / rng
            x_test = (x_test - mn) / rng
        elif scale == "mean":
            mean = x_train.mean(axis=0, keepdims=True)
            mn = x_train.min(axis=0, keepdims=True)
            mx = x_train.max(axis=0, keepdims=True)
            rng = mx - mn
            rng[rng == 0] = 1.0
            x_train = (x_train - mean) / rng
            x_val = (x_val - mean) / rng
            x_test = (x_test - mean) / rng
        else:
            raise ValueError(f"Unknown scale option: {scale}")

    return DatasetSplits(
        x_train=x_train.astype(np.float32),
        y_train=y_train,
        x_val=x_val.astype(np.float32),
        y_val=y_val,
        x_test=x_test.astype(np.float32),
        y_test=y_test,
        n_classes=n_classes,
        feature_names=feature_names,
    )


def make_semi_supervised(splits: DatasetSplits, labeled_frac: float, random_state: int = 0):
    """Return (labeled_idx, unlabeled_idx) into x_train for the semi-supervised
    setting where only `labeled_frac` of the training data retains labels.
    """
    n = splits.x_train.shape[0]
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(n)
    n_labeled = max(1, int(round(labeled_frac * n)))
    labeled_idx = idx[:n_labeled]
    unlabeled_idx = idx[n_labeled:]
    return labeled_idx, unlabeled_idx


def corrupt_labels(y: np.ndarray, n_classes: int, noise_frac: float, random_state: int = 0) -> np.ndarray:
    """Replace `noise_frac` of labels with a uniformly random class (including,
    possibly, the true class), matching the paper's label-noise protocol.
    """
    rng = np.random.RandomState(random_state)
    y_noisy = y.copy()
    n = len(y)
    n_corrupt = int(round(noise_frac * n))
    idx = rng.choice(n, size=n_corrupt, replace=False)
    y_noisy[idx] = rng.randint(0, n_classes, size=n_corrupt)
    return y_noisy
