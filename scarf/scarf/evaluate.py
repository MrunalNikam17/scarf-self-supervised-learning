"""
Evaluation utilities matching the paper's reporting methodology:

  - Win matrix: for each pair of methods (i, j), the fraction of datasets on
    which i's mean accuracy significantly (Welch's t-test, p<0.05) beats j's,
    counted only among dataset-pairs where a significant difference exists.
  - Relative gain: percent improvement of a method over a reference method,
    filtered to datasets where the means differ with p<0.20 (used for the
    paper's box plots / Table 1 summaries).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def welch_significant_diff(acc_i: np.ndarray, acc_j: np.ndarray, alpha: float = 0.05):
    """Welch's t-test (unequal variance) between two arrays of per-trial
    accuracies for the same dataset. Returns (beats, loses) booleans for
    method i relative to method j.
    """
    if len(acc_i) < 2 or len(acc_j) < 2:
        return False, False
    t_stat, p_val = stats.ttest_ind(acc_i, acc_j, equal_var=False)
    if p_val >= alpha:
        return False, False
    return (t_stat > 0), (t_stat < 0)


def win_matrix(results: Dict[str, Dict[str, np.ndarray]], alpha: float = 0.05) -> pd.DataFrame:
    """results: {method_name: {dataset_id: array_of_per_trial_accuracies}}.
    All methods must share the same set of dataset ids.

    Returns an MxM DataFrame W where W.loc[i, j] = wins / (wins + losses) of
    method i vs method j across all datasets, using Welch's t-test to define
    "beats"/"loses" per dataset (NaN on the diagonal).
    """
    methods = list(results.keys())
    dataset_ids = list(next(iter(results.values())).keys())

    W = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for i in methods:
        for j in methods:
            if i == j:
                W.loc[i, j] = np.nan
                continue
            wins, losses = 0, 0
            for d in dataset_ids:
                acc_i = results[i][d]
                acc_j = results[j][d]
                beats, loses = welch_significant_diff(acc_i, acc_j, alpha)
                wins += int(beats)
                losses += int(loses)
            denom = wins + losses
            W.loc[i, j] = (wins / denom) if denom > 0 else np.nan
    return W


def relative_gain(
    method_acc: Dict[str, np.ndarray],
    reference_acc: Dict[str, np.ndarray],
    alpha: float = 0.20,
) -> pd.Series:
    """Per-dataset relative percent gain of `method_acc` over `reference_acc`,
    filtered to datasets where the two differ with p < alpha (Welch's t-test).
    Both dicts map dataset_id -> array of per-trial accuracies.
    """
    gains = {}
    for d in method_acc:
        if d not in reference_acc:
            continue
        acc_m = method_acc[d]
        acc_r = reference_acc[d]
        if len(acc_m) < 2 or len(acc_r) < 2:
            continue
        _, p_val = stats.ttest_ind(acc_m, acc_r, equal_var=False)
        if p_val >= alpha:
            continue
        mean_m = np.mean(acc_m)
        mean_r = np.mean(acc_r)
        if mean_r == 0:
            continue
        gains[d] = 100.0 * (mean_m - mean_r) / mean_r
    return pd.Series(gains, name="relative_gain_pct")


def average_relative_gain_table(
    all_results: Dict[str, Dict[str, np.ndarray]],
    pretrain_methods: List[str],
    reference_methods: List[str],
    alpha: float = 0.20,
) -> pd.DataFrame:
    """Reproduce Table 1 style summary: average relative gain in accuracy
    when adding each pre-training method (columns) to each reference/baseline
    method (rows), e.g. reference_methods=["control", "dropout", "mixup", ...].

    `all_results` keys are like "control", "control+SCARF", "dropout+SCARF_AE",
    etc. -- callers are responsible for the naming convention; this function
    just needs, for each (reference, pretrain) pair, the accuracy dict for
    reference alone and for reference+pretrain combined.
    """
    table = pd.DataFrame(index=reference_methods, columns=pretrain_methods, dtype=float)
    for ref in reference_methods:
        for pt in pretrain_methods:
            combo_key = f"{ref}+{pt}"
            if ref not in all_results or combo_key not in all_results:
                table.loc[ref, pt] = np.nan
                continue
            gains = relative_gain(all_results[combo_key], all_results[ref], alpha=alpha)
            table.loc[ref, pt] = gains.mean() if len(gains) else np.nan
    return table
