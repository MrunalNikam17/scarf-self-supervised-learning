# SCARF Replication Progress Log

This document tracks all experimental runs, timestamps, configurations, trial counts, and honest verdicts across the reproduction effort.

---

## Stage Summary & Status

| Stage | Benchmark / Experiment | Config / Datasets | Trials | Status | Honest Verdict |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Stage 0** | Fully-Supervised Benchmark (`results_openml6`) | 6 OpenML datasets (`11, 37, 54, 1510, 1494, 15`) | 15 | Completed | Inconclusive signal: Control+SCARF vs Control delta is small (-1.22% relative gain driven by WDBC) and within symmetric noise bounds. |
| **Stage 0** | Semi-Supervised Benchmark (`results_openml6_semisup`) | 6 OpenML datasets, 25% labeled fraction | 15 | Completed | Mixed: WDBC and Breast-W show modest gains, while Vehicle shows systematic negative transfer (-2.5 to -2.8 pp) driven by small-validation-set (85 examples) checkpoint selection. |
| **Stage 0** | Vehicle Diagnosis & Early-Stopping Investigation | OpenML ID 54 (`vehicle`), 100% and 25% labeled | 10 | Completed | `min_epochs=15` lifts accuracy for both methods (+7.6 to +7.9 pp) by eliminating 4-epoch collapses, but does not close the 2.5 pp gap (negative transfer is dataset-intrinsic, not a bug). |
| **Stage 1** | Label-Noise Benchmark (`results_openml6_labelnoise`) | OpenML 6 datasets, 30% label noise, 7 reference x 3 pretrain methods | 15 | Completed (2026-09-05 18:01:54) | Mostly inconclusive: 12 of 14 Table-1 cells are NaN (0/6 datasets pass p < 0.20 filter). Only label_smooth+scarf (+1.50%, driven solely by qsar-biodeg) and distill+scarf (+3.38%, driven solely by vehicle) pass the filter; no consistent pretraining advantage on this 6-dataset sample. |
| **Stage 2** | Extended Baselines Benchmark (`results_openml6_semisup`) | Semi-supervised extended: 7 reference (`control, dropout, mixup, label_smooth, distill, self_train, tri_train`) x 3 pretrain methods on 6 OpenML datasets | 15 (5 for tri_train) | Completed (2026-09-05 18:17:08) | Highly polarized by dataset: Strong positive transfer on QSAR-biodeg (+10% to +21%, p < 0.01) and modest gains on diabetes/balance-scale, but Vehicle consistently suffers severe negative transfer (-12% to -29%, p < 0.01). Tri-training empirically confirmed to plateau by iteration 4. |
| **Stage 3** | Ablation Suite (`results_ablations`) | Phonemes (OpenML ID 1489), Appendix Figures 5-11 | 3-5 | On Hold (Hardware Break) | Script verified; awaiting user go-ahead after cooling. |
| **Stage 5** | README Rewrite | Entire repository documentation | - | On Hold (Hardware Break) | Awaiting completion of Stage 3. |

---
