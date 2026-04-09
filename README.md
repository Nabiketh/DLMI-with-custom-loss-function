# Chrono-Aware Embryo Phase Classification
### A 16-Phase Ablation Study with Custom Hybrid Loss

---

## Overview

This project investigates **chronologically-aware deep learning classification** for human embryo development using time-lapse imaging. Standard classifiers treat all class confusions equally — confusing phase 1 with phase 2 is penalized the same as confusing phase 1 with phase 16. This project addresses that blind spot.

**Task:** Classify time-lapse embryo images into **16 ordered developmental phases**: `tPB2 → … → tHB`

**Core Idea:** Introduce an ordinal/temporal notion of distance via a **hybrid loss function** that blends:
- **Cross-Entropy (CE)** — for exact phase classification
- **Expected-value regression penalty (MSE)** — for chronological proximity

---

## Custom Loss Function

Let $K = 16$ be the number of phases, indexed $i \in \{0, \dots, 15\}$. For a sample with true class index $y$ and model logits $\mathbf{z} \in \mathbb{R}^K$:

**1. Cross-Entropy (baseline)**
$$\mathcal{L}_{\text{CE}} = -\log p_y, \quad \text{where } \mathbf{p} = \text{softmax}(\mathbf{z})$$

**2. Expected phase index**
$$\mathbb{E}[\hat{y}] = \sum_{i=0}^{15} p_i \cdot i$$

**3. Chronological distance penalty (MSE)**
$$\mathcal{L}_{\text{MSE}} = \big(\mathbb{E}[\hat{y}] - y\big)^2$$

This creates a **rubber-band effect** — an error of 1 phase yields penalty 1, while an error of 10 phases yields penalty 100.

**4. Final hybrid loss**
$$\mathcal{L}_{\text{Hybrid}} = \alpha\,\mathcal{L}_{\text{CE}} + (1-\alpha)\,\mathcal{L}_{\text{MSE}}$$

| Configuration | α value |
|---|---|
| Baseline | `1.0` (pure CE) |
| Hybrid | `0.5` (50% CE + 50% MSE) |

---

## Experiment Setup

| Setting | Value |
|---|---|
| Hardware | RTX 3070 Ti Laptop GPU (8GB VRAM) |
| Input size | 299×299 (aligned with Inception v3) |
| Data split | 70 / 15 / 15 at **video level** |
| Classes | 16 ordered developmental phases |
| Backbones tested | MobileNet_v2, GoogLeNet, Inception_v3, VGG16, VGG19 |

Each backbone is trained twice — once with the **Baseline** objective and once with the **Hybrid** objective — for a total of **10 runs**.

**Evaluation Metrics:**
- **Exact Accuracy** — predicted phase matches true phase exactly
- **Tolerance Accuracy (±1)** — off-by-one predictions counted as correct

---

## Results

| Model | Run | Best Val Exact (%) | Best Val Tol ±1 (%) | Overfit Gap (pp) | Final Train Loss | Epochs |
|---|---|---:|---:|---:|---:|---:|
| MobileNet | Baseline (CE) | 64.65 | 87.21 | 1.60 | 0.91 | 6 |
| MobileNet | Hybrid (50/50 CE+MSE) | 63.40 | 87.25 | 1.72 | 0.86 | 9 |
| GoogLeNet | Baseline (CE) | 60.08 | 85.20 | 3.08 | 1.04 | 10 |
| GoogLeNet | Hybrid (50/50 CE+MSE) | 54.76 | 80.08 | -1.30 | 1.70 | 10 |
| InceptionV3 | Baseline (CE) | **67.00** | **89.42** | 0.45 | 0.89 | 6 |
| InceptionV3 | Hybrid (50/50 CE+MSE) | 63.60 | 87.61 | 0.15 | 1.01 | 8 |
| VGG16 | Baseline (CE) | 28.66 | 47.35 | -2.51 | 2.27 | 10 |
| VGG16 | Hybrid (50/50 CE+MSE) | 26.10 | 46.45 | -1.86 | 7.90 | 10 |
| VGG19 | Baseline (CE) | 49.99 | 73.72 | -1.06 | 1.47 | 7 |
| VGG19 | Hybrid (50/50 CE+MSE) | 49.25 | 72.26 | -1.93 | 2.09 | 7 |

---

## Key Findings

**🏆 Champion:** `InceptionV3 + Baseline CE` achieved the best overall performance — **89.42% tolerance accuracy** and **67.00% exact accuracy**.

**Regularization effect:** The Hybrid loss consistently suppressed overfitting, often shrinking or even flipping the train–val gap, suggesting it acts as an implicit regularizer.

**Gradient conflict / "lazy middle" behavior:** The MSE term can dominate optimization when the model assigns probability mass far from the true phase. This encourages safer, middle-ground predictions — typically reducing exact accuracy while inflating training loss compared to pure CE.

**Per-model observations:**
- **MobileNet** — consistent and efficient workhorse across both objectives
- **GoogLeNet** — degraded noticeably under the Hybrid objective
- **InceptionV3** — top performer overall; Hybrid variant showed minimal overfitting (gap: 0.15pp)
- **VGG16 / VGG19** — underperformed strongly; VGG16 + Hybrid showed severe loss inflation (final train loss: 7.90)


Each `*_results.png` contains training/validation loss and accuracy curves for that run. Checkpoints are saved as `.pth` files in the same directory.
## Report

See [`REPORT.md`](REPORT.md) for the full academic-style write-up covering methodology, analysis, and conclusions.
