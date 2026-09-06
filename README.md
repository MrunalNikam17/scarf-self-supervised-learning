# SCARF: Self-Supervised Contrastive Learning Using Random Feature Corruption

A PyTorch implementation of SCARF (self-supervised contrastive learning for tabular data via random feature corruption), along with supporting reference baselines, ablation tooling, and benchmark evaluation scripts.

---

## What Is Implemented

### 1. Core Architecture & Pre-training
- [**`scarf/corruption.py`**](file:///c:/Users/mruna/Downloads/scarf_project/scarf/scarf/corruption.py): Feature corruption algorithms. Given a batch of input rows, it samples $q = \lfloor c \cdot M \rfloor$ feature indices per row and replaces them with independent draws from that feature's empirical marginal distribution built from the training set. Also supports alternative corruption strategies: constant empirical mean replacement, additive Gaussian noise, empirical joint row sampling, learnable missing-feature embeddings, and feature dropout.
- [**`scarf/model.py`**](file:///c:/Users/mruna/Downloads/scarf_project/scarf/scarf/model.py): Neural network architectures:
  - **Encoder $f$**: 4-layer MLP with ReLU activations and hidden dimension 256.
  - **Projection Head $g$**: 2-layer MLP (hidden 256, output 256) with $L_2$ normalization on the output embedding.
  - **Classification Head $h$**: 2-layer MLP (hidden 256) mapping representation to class logits.
- [**`scarf/losses.py`**](file:///c:/Users/mruna/Downloads/scarf_project/scarf/scarf/losses.py): Loss functions:
  - **InfoNCE**: Contrastive loss using cosine similarity with softmax temperature $\tau=1.0$.
  - **Barlow Twins**: Cross-correlation criterion with off-diagonal penalty weight $\lambda=5\times 10^{-3}$.
  - **Alignment and Uniformity**: Equal-weighted representation alignment and uniformity loss ($t=2.0$).
- [**`scarf/trainer.py`**](file:///c:/Users/mruna/Downloads/scarf_project/scarf/scarf/trainer.py): Training and evaluation loops:
  - `pretrain_scarf`: Self-supervised contrastive pre-training with Adam ($\text{lr}=10^{-3}$, batch size 128, patience 3) evaluated on a static validation set constructed by cycling through validation data across 10 passes.
  - `finetune_classifier`: Supervised fine-tuning of the encoder and classification head with cross-entropy and early stopping on validation classification error. Includes an opt-in `min_epochs` parameter to prevent premature stopping on initial training plateaus.

### 2. Pre-training Variants & Reference Baselines
- **Pre-training Variants** ([`scarf/baselines.py`](file:///c:/Users/mruna/Downloads/scarf_project/scarf/scarf/baselines.py)):
  - **SCARF**: InfoNCE contrastive representation learning with marginal feature corruption.
  - **SCARF-AE** (`scarf_ae`): Denoising autoencoder trained with SCARF marginal corruption to reconstruct original uncorrupted features using mean squared error.
  - **Vanilla Autoencoder**: Standard reconstruction autoencoder without corruption.
  - **Gaussian Autoencoder**: Denoising autoencoder with additive Gaussian noise.
  - **Discriminative SCARF**: Binary classification model trained to predict whether a sample is real or corrupted.
- **Reference Methods** ([`scarf/experiment_runner.py`](file:///c:/Users/mruna/Downloads/scarf_project/scarf/scarf/experiment_runner.py)):
  - **Control**: Supervised MLP trained from random initialization.
  - **Dropout**: Supervised training with feature and activation dropout ($p=0.04$).
  - **Mixup**: Supervised training with convex linear interpolation ($\alpha=0.20$) of input features and targets.
  - **Label Smoothing**: Cross-entropy with uniform target smoothing ($\epsilon=0.10$).
  - **Self-Distillation**: Knowledge distillation from an initial supervised teacher model ($T=2.0$, $\alpha=0.5$).
  - **Self-Training**: Confidence-thresholded pseudo-labeling on unlabeled data ($\tau_{\text{conf}}=0.75$, 10 iterations).
  - **Tri-Training**: Semi-supervised learning using three classifiers trained on bootstrap samples that iteratively pseudo-label unlabeled samples based on two-model agreement (10 iterations).
  - **Deep k-NN**: Non-parametric k-nearest neighbors classification ($k=50$) over frozen encoder embeddings.
  - **Bi-Tempered Loss**: Heavy-tailed loss robust to label noise ($t_1=0.8, t_2=1.2$).

### 3. Ablation Tooling & Evaluation
- [**`scarf/ablations.py`**](file:///c:/Users/mruna/Downloads/scarf_project/scarf/scarf/ablations.py) & [**`scripts/run_ablations.py`**](file:///c:/Users/mruna/Downloads/scarf_project/scripts/run_ablations.py):
  - **Corruption Strategies & Scaling**: 7 corruption strategies under z-score and min-max feature scaling.
  - **Hyperparameter Sweeps**: Batch sizes $\{4, 16, 64, 128, 256, 512\}$, corruption rates 10% to 90%, softmax temperatures $\{0.01, 0.1, 1.0, 10.0\}$.
  - **Alternative Losses**: InfoNCE vs. Barlow Twins vs. Alignment & Uniformity.
  - **Training Paradigms**: Pre-training vs. Co-training ($\lambda \in \{0.01, 0.1, 1.0\}$) vs. direct Data Augmentation.
  - **Validation Metrics**: Early stopping on InfoNCE validation loss vs. off-diagonal InfoNCE error.
- [**`scarf/data.py`**](file:///c:/Users/mruna/Downloads/scarf_project/scarf/scarf/data.py): Data loading from OpenML or local CSVs, categorical one-hot encoding, missing value imputation, scaling, 70/10/20 train/val/test splits, semi-supervised masking, and symmetric label noise injection.
- [**`scarf/evaluate.py`**](file:///c:/Users/mruna/Downloads/scarf_project/scarf/scarf/evaluate.py):
  - **Win Matrix**: Pairwise comparison measuring the fraction of datasets where method $i$ statistically outperforms method $j$ using Welch's two-sample t-test ($p < 0.05$).
  - **Relative Gain Table**: Relative percent gain of candidate method $M$ over reference method $M_0$, filtered to datasets where the two methods differ with $p < 0.20$:
$$\text{RelGain}(M, M_0) = 100 \times \frac{\text{Acc}(M) - \text{Acc}(M_0)}{\text{Acc}(M_0)}$$

---

## How to Run

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Sanity Check

Run the offline smoke test on bundled datasets:

```bash
python scripts/smoke_test.py
```

### 3. Benchmarks

Run the benchmark runner across OpenML datasets:

```bash
# Semi-supervised benchmark (25% labeled data)
python scripts/run_benchmark.py --dataset-ids 11 37 54 1510 1494 15 --n-trials 15 --semi-supervised --labeled-fraction 0.25 --output-dir results_semisup/

# Fully-supervised benchmark (100% labeled data)
python scripts/run_benchmark.py --dataset-ids 11 37 54 1510 1494 15 --n-trials 15 --output-dir results_supervised/

# Label-noise benchmark (30% label corruption)
python scripts/run_benchmark.py --dataset-ids 11 37 54 1510 1494 15 --n-trials 15 --label-noise 0.30 --output-dir results_labelnoise/
```

### 4. Ablation Sweeps on Google Colab / GPU

`scripts/run_ablations.py` automatically detects CUDA if available:

```python
# In Google Colab
!git clone https://github.com/MrunalNikam17/scarf-self-supervised-learning.git
%cd scarf-self-supervised-learning
!pip install -r requirements.txt

# Run remaining sweeps on Phonemes (OpenML 1489)
!python scripts/run_ablations.py --dataset-id 1489 --n-trials 3 --sweep "batch_size,corruption_rate,temperature,losses,cotrain_aug,val_metric" --output-dir colab_ablations/
```

---

## Experimental Results

Experiments were conducted on 6 benchmark datasets from OpenML:
- **Balance-Scale** (ID 11): 625 samples, 4 features, 3 classes
- **QSAR-Biodegradation** (ID 1494): 1,055 samples, 41 features, 2 classes
- **WDBC** (ID 1510): 569 samples, 30 features, 2 classes
- **Breast-Cancer Wisconsin** (ID 15): 699 samples, 9 features, 2 classes
- **Diabetes** (ID 37): 768 samples, 8 features, 2 classes
- **Vehicle Silhouettes** (ID 54): 846 samples, 18 features, 4 classes

---

### 1. Semi-Supervised Learning (25% Labeled Data, 15 Trials)

In this setting, 75% of training labels are masked and used only as unlabeled data during pre-training.

#### Relative Gain Table ($p < 0.20$ significance filter):

| Reference Baseline | SCARF Relative Gain | SCARF-AE Relative Gain |
| :--- | :---: | :---: |
| **Control** | +0.29% | -2.64% |
| **Dropout** | +1.82% | -7.89% |
| **Mixup** | +0.93% | +2.14% |
| **Label Smoothing** | -2.91% | -4.43% |
| **Self-Distillation** | +8.87% | -4.95% |
| **Self-Training** | +1.03% | -2.29% |
| **Tri-Training**\* | +2.01% | +9.56% |

\* *`tri_train` evaluated with $n=5$ trials per configuration due to iteration overhead; all other baselines use $n=15$.*

#### Dataset-Level Split:

1. **QSAR-Biodeg (OpenML 1494)**: Adding SCARF pre-training improved classification accuracy across every reference method:
   - `control+scarf` vs. `control`: **0.8343 vs. 0.7456** (+11.89% relative gain, $p = 0.0011$).
   - `dropout+scarf` vs. `dropout`: **0.8292 vs. 0.7340** (+12.98% relative gain, $p = 0.0003$).
   - `tri_train+scarf` vs. `tri_train`: **0.8472 vs. 0.6981** (+21.35% relative gain, $p = 0.0016$).
   - `self_train+scarf` vs. `self_train`: **0.8377 vs. 0.7531** (+11.23% relative gain, $p = 0.0022$).
   - `mixup+scarf` vs. `mixup`: **0.8016 vs. 0.7346** (+9.12% relative gain, $p = 0.0084$).

2. **Vehicle (OpenML 54)**: Adding pre-training consistently degraded classification accuracy across reference methods:
   - `control+scarf_ae` vs. `control`: **0.4941 vs. 0.5894** (-16.17% relative gain, $p = 0.0071$).
   - `dropout+scarf_ae` vs. `dropout`: **0.4157 vs. 0.5863** (-29.10% relative gain, $p = 0.0001$).
   - `distill+scarf_ae` vs. `distill`: **0.4329 vs. 0.5545** (-21.93% relative gain, $p = 0.0073$).
   - `label_smooth+scarf` vs. `label_smooth`: **0.5259 vs. 0.6306** (-10.47% raw drop, $p = 0.0094$).
   - Across almost every reference baseline on Vehicle, `scarf_ae` caused a larger drop than `scarf` (e.g., dropout+scarf at 0.5161 vs. dropout+scarf_ae at 0.4157).

3. **Other Datasets**: Diabetes, Breast-W, and Balance-Scale showed smaller, consistent shifts (e.g., `control+scarf` reached 0.7225 vs. 0.7065 on Diabetes, 0.8725 vs. 0.8608 on Balance-Scale).

---

### 2. Fully-Supervised Learning (100% Labeled Data, 15 Trials)

#### Relative Gain Table ($p < 0.20$ significance filter):

| Reference Baseline | SCARF Relative Gain | SCARF-AE Relative Gain |
| :--- | :---: | :---: |
| **Control** | -1.22% | -1.16% |
| **Mixup** | +1.99% | +2.73% |
| **Label Smoothing** | +0.86% | +0.96% |

- `control+scarf` vs. `control`: Only WDBC met the $p < 0.20$ threshold (accuracy shifted from 0.957 to 0.950), yielding a -1.22% relative gain.
- In the win matrix ($p < 0.05$), pairwise comparisons between `control` and `control+scarf` recorded 0 significant wins and 0 significant losses across all 6 datasets.

---

### 3. Label-Noise Benchmark (30% Symmetric Noise, 15 Trials)

Under 30% uniform random label noise:
- 12 of 14 relative gain table cells were **NaN**, as differences failed the $p < 0.20$ filter on 6 of 6 datasets.
- Only two combinations cleared the threshold:
  - `label_smooth+scarf` vs. `label_smooth`: **+1.50%** relative gain (cleared on QSAR-Biodeg).
  - `distill+scarf` vs. `distill`: **+3.38%** relative gain (cleared on Vehicle).
- No method combinations showed statistically significant negative transfer under label noise at this sample size.

---

### 4. Ablation Study Results (Phonemes, OpenML 1489, 3 Trials)

#### Ablation 1: Corruption Strategies & Feature Scaling

Evaluated on Phonemes (5,404 rows, 5 features, 3 trials):

| Strategy | Scaling | Test Accuracy (Mean ± Std) |
| :--- | :---: | :---: |
| **Missing Feature Embedding** | z-score | **0.8688 ± 0.0047** |
| **Marginal Sampling (SCARF)** | z-score | **0.8644 ± 0.0068** |
| **Feature Dropout** | z-score | 0.8632 ± 0.0046 |
| **Gaussian Noise** | z-score | 0.8601 ± 0.0054 |
| **No Corruption (`none`)** | z-score | 0.8592 ± 0.0102 |
| **Marginal Mean** | z-score | 0.8577 ± 0.0106 |
| **Joint Sampling** | z-score | 0.8540 ± 0.0066 |
| **Missing Feature Embedding** | min-max | 0.8457 ± 0.0119 |
| **Marginal Mean** | min-max | 0.8438 ± 0.0059 |
| **Feature Dropout** | min-max | 0.8429 ± 0.0085 |
| **Gaussian Noise** | min-max | 0.8386 ± 0.0103 |
| **Marginal Sampling (SCARF)** | min-max | 0.8370 ± 0.0188 |
| **Joint Sampling** | min-max | 0.8269 ± 0.0024 |
| **No Corruption (`none`)** | min-max | 0.8232 ± 0.0045 |

**Findings**:
- **Scaling impact**: z-score scaling achieved higher accuracy than min-max scaling across all strategies (+1.5 to +3.6 percentage points).
- **Strategy performance**: Under z-score, learnable missing-feature embeddings and marginal sampling scored highest (0.8688 and 0.8644), followed by feature dropout (0.8632), Gaussian noise (0.8601), uncorrupted input (0.8592), empirical mean replacement (0.8577), and joint empirical sampling (0.8540).

#### Ablations 2–8: Pending Colab GPU Execution

The remaining ablation sweeps on Phonemes (OpenML ID 1489, 3 trials) require dedicated GPU compute and are pending execution:
- **Ablation 2**: Batch Size Sweep $\{4, 16, 64, 128, 256, 512\}$
- **Ablation 3**: Corruption Rate Sweep (10% to 90%)
- **Ablation 4**: Softmax Temperature Sweep $\{0.01, 0.1, 1.0, 10.0\}$
- **Ablation 5**: Alternative Losses (InfoNCE vs. Barlow Twins vs. Alignment & Uniformity)
- **Ablations 6 & 7**: Pre-Training vs. Co-Training vs. Data Augmentation
- **Ablation 8**: Validation Metric for Early Stopping (InfoNCE Loss vs. InfoNCE Error)

---

## Known Limitations

1. **Benchmark Scale**: The benchmark was evaluated on 6 OpenML datasets across 15 trials (and 5 trials for `tri_train`). At this sample size, small percentage shifts in fully-supervised or noisy settings remain within sampling noise.
2. **Tri-Training Trial Budget**: `tri_train` trains three separate classifiers per iteration across 10 iterations. Due to execution time, it was evaluated at $n=5$ trials per configuration, which carries higher sampling variance than the $n=15$ runs.
3. **Dataset-Specific Negative Transfer**: Vehicle Silhouettes exhibited consistent, statistically significant negative transfer with SCARF ($p < 0.01$), and autoencoder pre-training degraded accuracy further. Pre-training representations via random feature corruption does not benefit all tabular geometries.
4. **Early Stopping Sensitivity**: Early stopping with patience 3 can terminate training prematurely during initial plateaus on smaller datasets. The `min_epochs` parameter mitigates early cutoffs, but does not alter performance when pre-training is fundamentally unsuited to a dataset.

---

## Project Structure

```
scarf_project/
├── scarf/
│   ├── corruption.py         # Marginal sampling and alternative corruptions
│   ├── model.py              # Encoder, ProjectionHead, ClassificationHead
│   ├── losses.py             # InfoNCE, Barlow Twins, Alignment & Uniformity
│   ├── trainer.py            # Pre-training, fine-tuning, early stopping
│   ├── baselines.py          # Autoencoders (vanilla, gaussian, scarf_ae) & discriminative SCARF
│   ├── experiment_runner.py  # Reference methods (dropout, mixup, self/tri-training, etc.)
│   ├── ablations.py          # Co-training, data augmentation, alternative losses
│   ├── data.py               # Preprocessing, OpenML loading, dataset splitting
│   └── evaluate.py           # Win matrix (t-test p<0.05) & relative gain (p<0.20)
├── scripts/
│   ├── run_benchmark.py      # Benchmark runner (supervised, semi-supervised, label noise)
│   ├── run_ablations.py      # Ablation runner with automatic CUDA detection
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation and empirical findings
```
