# Brain Tumor Segmentation in MRI — BRISC 2025

> Systematic benchmark of five deep learning segmentation architectures
> on BRISC 2025, with an original proposed model in progress.
> All five of our reproductions surpass the Weighted mIoU of the
> classical baselines reported in the dataset paper. Our best result
> — U-Net at **80.94% W-mIoU** — exceeds the previous state-of-the-art
> (SaberNet, 80.60%) under identical experimental conditions.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![SMP](https://img.shields.io/badge/segmentation--models--pytorch-0.5.0-informational)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20A100%2080GB-76B900?logo=nvidia&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-BRISC%202025-blueviolet)
![JBook](https://img.shields.io/badge/Jupyter%20Book-published-orange)
![Lang](https://img.shields.io/badge/Notebook%20language-Spanish-yellow)

📖 **[Read the full project documentation (Jupyter Book)](https://luispenaranda08.github.io/brain-tumor-segmentation-brisc2025)**

---

## Overview

This project presents a pixel-wise segmentation study of brain tumors
in T1 MRI images using **BRISC 2025** — a radiologist-annotated dataset
covering glioma, meningioma, pituitary tumor, and non-tumor cases across
three anatomical planes (Axial, Coronal, Sagittal).

Five benchmark architectures were reproduced and evaluated under
homogeneous training conditions. Building on the benchmark findings,
we propose an original architecture — **Res2Net-SE(2) UNet** — designed
to address the two main sources of variability identified in the analysis:
inter-tumor morphological variability and anatomical-plane variability.
Results for the proposed model are in progress.

> **Note:** The Jupyter Book is currently written in Spanish.
> An English version will be available in a dedicated branch.

---

## Key Results

| Metric | Value | Model |
|---|---|---|
| **Weighted mIoU** | **80.94%** | U-Net (Ours) — new SOTA |
| Dice Score (global) | 87.43% | U-Net (Ours) |
| Best Meningioma IoU | 91.13% | UNet++ (Ours) |
| Previous SOTA | 80.60% | SaberNet — Fateh et al., 2026 |

---

## Dataset — BRISC 2025

| Split | Images |
|---|---|
| Train | 3,146 |
| Validation | 787 |
| Test | 860 |
| **Total** | **4,793** |

- **Modality:** MRI T1, grayscale, three anatomical planes
- **Classes:** Glioma · Meningioma · Pituitary tumor · No tumor
- **Annotated by:** certified radiologists
- **Reference:** Fateh et al. (2026). *BRISC: Annotated Dataset for
  Brain Tumor Segmentation and Classification.* Scientific Data, 13(1):361.
  https://doi.org/10.1038/s41597-026-06753-y

---

## Benchmark Results

### Weighted mIoU — Full Comparison vs. BRISC 2025 Paper

| # | Model | Glioma | Meningioma | Pituitary | W-mIoU | Source |
|---|---|---|---|---|---|---|
| 1 | **U-Net** | 68.7% | 90.8% | 81.3% | **80.94%** | **Ours** |
| 2 | **UNet++** | 68.6% | 91.1% | 80.7% | **80.85%** | **Ours** |
| 3 | SaberNet ★ | 74.0% | 82.4% | 84.3% | 80.60% | Fateh et al. |
| 4 | **Attention U-Net** | 68.2% | 90.9% | 80.4% | **80.52%** | **Ours** |
| 5 | **DeepLabV3+** | 68.4% | 90.6% | 80.2% | **80.41%** | **Ours** |
| 6 | **TransUNet** | 67.6% | 90.6% | 80.6% | **80.34%** | **Ours** |
| 7 | DAD | 75.2% | 80.4% | 82.3% | 79.50% | Fateh et al. |
| 8 | ABANet | 72.4% | 80.4% | 84.7% | 79.50% | Fateh et al. |
| 9 | BASNet | 74.0% | 77.5% | 81.7% | 77.90% | Fateh et al. |
| 10 | EINet | 73.6% | 78.4% | 80.3% | 77.70% | Fateh et al. |
| 11 | DeepLabV3+ (paper) | 72.0% | 77.5% | 78.7% | 76.30% | Fateh et al. |
| 12 | U-Net (paper) | 69.7% | 77.1% | 79.3% | 75.70% | Fateh et al. |

> ★ Previous SOTA. **Bold** = our reproductions.

### Original Contributions — Not Reported in the BRISC Paper

**mIoU by anatomical plane:**

| Model | Axial | Coronal | Sagittal |
|---|---|---|---|
| **U-Net** | 79.65% | 79.90% | **83.73%** |
| **UNet++** | 79.17% | 81.15% | 82.83% |
| **Attention U-Net** | 79.11% | 80.73% | 82.21% |
| **TransUNet** | 78.65% | 80.46% | 82.48% |
| **DeepLabV3+** | 78.58% | 81.02% | 82.27% |

**Finding:** The Sagittal plane is consistently the easiest across all
models; the Axial plane is the most challenging. The combination
**Glioma × Axial** concentrates the highest segmentation difficulty —
a pattern not reported in the original paper and a key motivation for
the proposed architecture.

---

## Proposed Model — In Progress

**Res2Net-SE(2) UNet** — a hybrid architecture combining:

- **Res2Net encoder** → multi-scale feature hierarchies within each
  residual block, improving discrimination between tumor morphologies.
- **SE(2) equivariance** → group convolutions equivariant to rotations
  and translations, producing consistent representations across
  anatomical planes.
- **Group Deconvolution decoder** → maintains geometric consistency
  during upsampling.
- **Ablation study** → three variants trained under identical conditions:
  Res2NetUNet · SE2UNet · Res2NetSE2UNet.

*Results will be added to this repository and the Jupyter Book upon
completion of training.*

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 512 × 512 |
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 1e-4 |
| Loss | ComboLoss (0.5·Dice + 0.5·BCE) |
| LR Scheduler | ReduceLROnPlateau (patience=20, factor=0.5) |
| Early stopping | patience=80 |
| Max epochs | 500 |
| Precision | Mixed (AMP) |
| Encoder | ResNet34 (ImageNet pretrained) |
| GPU | NVIDIA A100 SXM4 80GB |
| Seed | 42 |

---

## Repository Structure
