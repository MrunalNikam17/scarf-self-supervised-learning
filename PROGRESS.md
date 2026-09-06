# SCARF Replication Progress Log

This document tracks all experimental runs, timestamps, configurations, trial counts, and honest verdicts across the reproduction effort.

---

## Stage Summary & Status

| Stage | Benchmark / Experiment | Config / Datasets | Trials | Status | Honest Verdict |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Stage 0** | Fully-Supervised Benchmark (`results_openml6`) | 6 OpenML datasets (`11, 37, 54, 1510, 1494, 15`) | 15 | Completed | Inconclusive signal: Control+SCARF vs Control delta is small (-1.22% relative gain driven by WDBC) and within symmetric noise bounds across 6 datasets. Matches paper's small fully-supervised effect size. |
| **Stage 0** | Semi-Supervised Benchmark (`results_openml6_semisup`) | 6 OpenML datasets, 25% labeled fraction | 15 | Completed | Highly polarized by dataset: QSAR-Biodeg shows strong positive transfer (+11.89%, p=0.0011), while Vehicle exhibits severe, reproducible negative transfer (-12% to -29%, p < 0.01). Small-validation checkpoint sensitivity contributes, but low p-values confirm real negative transfer. |
| **Stage 0** | Vehicle Diagnosis & Early-Stopping Investigation | OpenML ID 54 (`vehicle`), 100% and 25% labeled | 10 | Completed | `min_epochs=15` lifts accuracy for both methods (+7.6 to +7.9 pp) by eliminating 4-epoch collapses, but does not close the gap. `scarf_ae` consistently hurts Vehicle worse than plain `scarf` across almost all reference baselines (e.g. `dropout+scarf_ae`: -29.10%, p=0.0001; `control+scarf_ae`: -16.17%, p=0.0071). |
| **Stage 1** | Label-Noise Benchmark (`results_openml6_labelnoise`) | OpenML 6 datasets, 30% label noise, 7 reference x 3 pretrain methods | 15 | Completed (2026-09-05 18:01:54) | Mostly inconclusive: 12 of 14 Table-1 cells are NaN (0/6 datasets pass p < 0.20 filter). Only `label_smooth+scarf` (+1.50%, driven solely by qsar-biodeg) and `distill+scarf` (+3.38%, driven solely by vehicle) pass the filter; no evidence of harm, but weak evidence of general benefit at 6-dataset scale. |
| **Stage 2** | Extended Baselines Benchmark (`results_openml6_semisup`) | Semi-supervised extended: 7 reference (`control, dropout, mixup, label_smooth, distill, self_train, tri_train`) x 3 pretrain methods on 6 OpenML datasets | 15 (5 for tri_train\*) | Completed (2026-09-05 18:17:08) | Highly polarized by dataset: Strong positive transfer on QSAR-biodeg (+10% to +21%, p < 0.01) and modest gains on diabetes/balance-scale, but Vehicle consistently suffers severe negative transfer (-12% to -29%, p < 0.01). Tri-training confirmed to plateau by iteration 4. |
| **Stage 3** | Ablation Suite (`results_ablations`) | Phonemes (OpenML ID 1489, 5,404 rows), Appendix Figures 5–11 | 3 | Ablation 1 completed (696.5s); Sweeps 2–8 pending Colab GPU | Ablation 1 on Phonemes confirms marginal sampling outperforms mean, joint, and no-corruption, and z-score beats min-max. Sweeps 2–8 pending execution on Colab GPU. |
| **Stage 4/5** | Final Documentation & README Rewrite | End-to-end rewrite of `README.md` | - | Completed | Full codebase, baselines, empirical tables, corrected vehicle framing, limitations, and Google Colab execution guide documented. |

\* *Caveat: `tri_train` runs are based on n=5 trials due to 3-model iteration costs, while all other baselines are based on n=15 trials.*

---

## Detailed Experimental Logs & Raw Tables

### 1. Semi-Supervised Relative Gain Table (OpenML-CC18 6 Datasets, 25% Labeled, 15 Trials)

Aggregated relative gains filtered at the paper's $p < 0.20$ significance threshold:

| Reference Baseline | SCARF Gain | SCARF-AE Gain | Notes |
| :--- | :---: | :---: | :--- |
| **control** | +0.29% | -2.64% | Modest positive for SCARF; negative for SCARF-AE |
| **dropout** | +1.82% | -7.89% | SCARF aids dropout; SCARF-AE severely degrades |
| **mixup** | +0.93% | +2.14% | Positive transfer across both pre-trainers |
| **label_smooth** | -2.91% | -4.43% | Mixed / slightly negative |
| **distill** | +8.87% | -4.95% | Strong positive synergy with SCARF |
| **self_train** | +1.03% | -2.29% | Moderate positive transfer with SCARF |
| **tri_train** (n=5) | +2.01% | +9.56% | Notable positive gains (higher sampling variance) |

#### Corrected Framing for Vehicle Silhouettes vs. QSAR-Biodeg

- **QSAR-Biodeg (OpenML 1494, 41 features)**:
  - `control+scarf` vs. `control`: 0.8343 vs. 0.7456 (+11.89% relative gain, **p = 0.0011**).
  - `dropout+scarf` vs. `dropout`: 0.8292 vs. 0.7340 (+12.98% relative gain, **p = 0.0003**).
  - `tri_train+scarf` vs. `tri_train`: 0.8472 vs. 0.6981 (+21.35% relative gain, **p = 0.0016**).
  - *Conclusion*: Represents the strongest evidence that SCARF functions as described in the paper when tabular features provide meaningful subspace structure.
- **Vehicle Silhouettes (OpenML 54, 18 features)**:
  - `dropout+scarf_ae` vs. `dropout`: 0.4157 vs. 0.5863 (**-29.10% relative gain, p = 0.0001**).
  - `control+scarf_ae` vs. `control`: 0.4941 vs. 0.5894 (**-16.17% relative gain, p = 0.0071**).
  - `distill+scarf_ae` vs. `distill`: 0.4329 vs. 0.5545 (**-21.93% relative gain, p = 0.0073**).
  - `label_smooth+scarf` vs. `label_smooth`: 0.5259 vs. 0.6306 (**-10.47% raw drop, p = 0.0094**).
  - *Conclusion*: With p-values well below 0.01, this is a real, reproducible negative transfer effect, not mere checkpoint noise. Furthermore, `scarf_ae` consistently inflicts worse damage than plain `scarf`, showing that reconstruction pretext objectives are especially counterproductive on this dataset.

---

### 2. Stage C Ablation Study Results

#### Ablation 1: Corruption Strategies on Phonemes (OpenML 1489, 3 Trials, 696.5s)

Executed on 2026-09-06 (11:28:11 to 11:39:47):

| Variant | Scaling | Mean Accuracy | Std Accuracy | Paper Comparison |
| :--- | :---: | :---: | :---: | :--- |
| `missing_feature_zscore` | z-score | 0.868762 | 0.004713 | Top performer; learnable embedding works well |
| `marginal_zscore` (SCARF) | z-score | 0.864449 | 0.006764 | **Matches**: Outperforms none, mean, joint, and Gaussian |
| `feature_dropout_zscore` | z-score | 0.863216 | 0.004590 | Competitive with marginal sampling |
| `gaussian_zscore` | z-score | 0.860136 | 0.005354 | Slightly lower than marginal sampling |
| `none_zscore` | z-score | 0.859211 | 0.010245 | **Matches**: Contrastive corruption beats identity view |
| `mean_zscore` | z-score | 0.857671 | 0.010565 | Lower than marginal sampling |
| `joint_zscore` | z-score | 0.853974 | 0.006579 | **Matches**: Sampling whole joint rows degrades performance |
| `missing_feature_minmax` | min-max | 0.845656 | 0.011860 | Min-max is systematically lower across all variants |
| `mean_minmax` | min-max | 0.843808 | 0.005894 | Min-max degradation |
| `feature_dropout_minmax` | min-max | 0.842884 | 0.008504 | Min-max degradation |
| `gaussian_minmax` | min-max | 0.838571 | 0.010273 | Min-max degradation |
| `marginal_minmax` | min-max | 0.837030 | 0.018795 | Min-max degradation |
| `joint_minmax` | min-max | 0.826864 | 0.002426 | Lowest among corruptions |
| `none_minmax` | min-max | 0.823167 | 0.004549 | Lowest overall |

---

### 3. Ablations 2–8: Pending Execution on Google Colab GPU

The remaining ablation sweeps on Phonemes (OpenML ID 1489, 5,404 rows, 3 trials) require significant compute and are **pending execution on Google Colab GPU**:

- **Ablation 2: Batch Size Sweep** $\{4, 16, 64, 128, 256, 512\}$ — *Pending Colab run*
- **Ablation 3: Corruption Rate Sweep** (10% to 90% in steps of 10%) — *Pending Colab run*
- **Ablation 4: Softmax Temperature Sweep** $\{0.01, 0.1, 1.0, 10.0\}$ — *Pending Colab run*
- **Ablation 5: Alternative Loss Objectives** (InfoNCE vs. Barlow Twins vs. Alignment & Uniformity) — *Pending Colab run*
- **Ablations 6 & 7: Pre-Training vs. Co-Training vs. Data Augmentation** — *Pending Colab run*
- **Ablation 8: Validation Metric for Early Stopping** (InfoNCE Loss vs. InfoNCE Error) — *Pending Colab run*

*Note: Previous CSV files in `results_ablations/` for sweeps 2–8 were historical test artifacts on small local CSVs (Glass, Diabetes, Sonar, Wine) dated 2026-09-03, not the instructed Phonemes (OpenML 1489) benchmark. All numbers have been removed to maintain documentation integrity until the real Colab GPU run is performed.*
