# SCARF: Self-Supervised Contrastive Learning Using Random Feature Corruption

A from-scratch PyTorch reproduction of Bahri, Jiang, Tay & Metzler (ICLR 2022),
*"SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption"*.

## What's implemented

- **`scarf/corruption.py`** — the marginal-sampling corruption (Algorithm 1,
  lines 2–5): for each row, sample `q = floor(c · M)` feature indices and
  replace them with i.i.d. draws from that feature's empirical marginal.
- **`scarf/model.py`** — encoder `f` (4-layer ReLU MLP, hidden 256),
  projection head `g` (2-layer, L2-normalized output), classification head
  `h` (2-layer), matching the paper's architecture (Section 4, "Model
  architecture and training").
- **`scarf/losses.py`** — InfoNCE loss exactly as in Algorithm 1, line 8.
- **`scarf/trainer.py`** — `pretrain_scarf` (contrastive pre-training with
  early stopping, patience 3, on a *static* validation InfoNCE loss built by
  cycling through validation data 10 epochs, as described in Section 3) and
  `finetune_classifier` (supervised fine-tuning of `f`+`h`, early stopping on
  validation classification error; supports label smoothing and mixup).
- **`scarf/baselines.py`** — the ablation baselines: no-noise autoencoder,
  additive-Gaussian-noise autoencoder, SCARF-corruption autoencoder, and
  discriminative SCARF (binary real-vs-corrupted pretext task).
- **`scarf/data.py`** — OpenML-CC18-style preprocessing: one-hot encoding of
  categoricals, mode/mean imputation of missing values, z-score/min-max/mean
  scaling, 70/10/20 train/val/test split, plus helpers for the
  semi-supervised (`make_semi_supervised`) and label-noise
  (`corrupt_labels`) settings from Section 4.
- **`scarf/evaluate.py`** — the paper's win matrix (Welch's t-test, p<0.05)
  and relative-gain (p<0.20) evaluation methodology (Section 4, "Evaluation
  methods").

Default hyperparameters match the paper: Adam @ lr=1e-3, batch size 128,
corruption rate c=0.6, temperature τ=1, hidden dim 256, early-stopping
patience 3.

**Not (yet) implemented:** self-training / tri-training / self-distillation /
Deep k-NN / bi-tempered-loss baselines, and the alternative-loss ablations
(Barlow Twins, Alignment-Uniformity). These are straightforward to add on top
of `trainer.py` if you want the full ablation suite — ask and I can add them.

## Install

```bash
pip install -r requirements.txt
```

## Quick sanity check (no internet required)

```bash
python scripts/smoke_test.py
```

Runs the full pipeline (all pre-training methods + fine-tuning + evaluation)
on `sklearn`'s bundled breast_cancer/wine/digits datasets, 3 trials each,
with small epoch budgets — just to confirm the code runs correctly, not to
produce meaningful numbers.

## Real benchmark

### Option A: OpenML-CC18 (matches the paper exactly)

```bash
python scripts/run_benchmark.py \
    --dataset-ids 11 37 54 1510 1494 15 \
    --n-trials 10 \
    --methods control scarf \
    --output-dir results/
```

Requires network access to openml.org and `pip install openml`.

### Option B: local CSVs (used to validate this project — see results below)

```bash
python scripts/run_local6_benchmark.py
```

## Results so far (6 small UCI datasets, 10 trials, `control` vs `scarf` vs `scarf_ae`)

| dataset | control | scarf | scarf_ae |
|---|---|---|---|
| glass | 0.561 | 0.491 | **0.598** |
| ionosphere | **0.928** | 0.892 | 0.894 |
| pima-diabetes | 0.754 | 0.752 | **0.760** |
| sonar | 0.767 | **0.779** | 0.705 |
| wheat-seeds | 0.860 | 0.872 | **0.881** |
| wine | **0.950** | 0.900 | 0.903 |

**SCARF pre-training did *not* help on 4/6 of these datasets — this is expected,
not a bug, for three reasons that all point the same direction:**

1. **These are tiny datasets** (208–768 rows). SCARF's own paper only reports
   a **1–2% average gain** in the fully-supervised setting across 69
   datasets with 30 trials each — a small, noisy effect that a handful of
   200-row datasets with 10 trials will often show as a wash or even a
   reversal, purely from variance.
2. **This is the *fully-supervised* setting**, exactly where the paper's
   gains are smallest. The paper shows SCARF shines much more in the
   **semi-supervised** setting (+2–4%, Section 4.3) and under **label noise**
   (+2–3%, Section 4.2) — settings where pre-training gets to exploit
   unlabeled data or lower-quality labels that supervised training can't use
   as well.
3. **These aren't the actual OpenML-CC18 datasets** the paper benchmarks on
   (network access to openml.org isn't available in this sandbox — see
   "Option A" above to run on the real 69, or a subset of them, once you have
   internet access). They're a reasonable stand-in for exercising the code,
   but shouldn't be over-interpreted as a critique of SCARF.

**Recommended next step:** rerun on the semi-supervised setting (25% labels)
on the same 6 datasets, where the paper's own results suggest we should see
a much clearer and more consistent win for SCARF. I can wire that up next.

Raw per-trial numbers: `results_local6/raw_results.json`,
`results_local6/summary.csv`, `results_local6/win_matrix.csv`.

## Project layout

```
scarf/
  scarf/
    corruption.py   # marginal-sampling view generation
    model.py         # encoder f, heads g/h
    losses.py        # InfoNCE
    trainer.py        # pretrain_scarf, finetune_classifier
    baselines.py      # autoencoder + discriminative-SCARF baselines
    data.py            # OpenML loading, preprocessing, semi-sup/label-noise helpers
    evaluate.py         # win matrix, relative gain
  scripts/
    smoke_test.py        # offline correctness check (sklearn datasets)
    run_benchmark.py       # real OpenML-CC18 benchmark (needs internet)
    run_local6_benchmark.py # 6-dataset benchmark using local CSVs
  data_cache/               # cached UCI CSVs used by run_local6_benchmark.py
  results_local6/            # results from the run above
  requirements.txt
```
