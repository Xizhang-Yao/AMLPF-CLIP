# AMLPF-CLIP

**AMLPF-CLIP** is an improved CLIP-based framework for histopathological image classification (HIC) that addresses key challenges in computer-aided diagnosis by integrating adaptive multi-level prompt fusion, class-balanced resampling, and cross-architecture knowledge distillation. This repository provides code and implementation details for the approach described in our upcoming paper.

## Overview

Histopathological image classification is critical for effective cancer screening, disease grading, and treatment planning. Despite advances with deep learning models, existing methods face limitations in integrating domain-specific pathology knowledge, handling class imbalance, and managing high computational costs. Our method, **AMLPF-CLIP**, leverages:

- **Adaptive Multi-Level Prompt Fusion (AMLPF):** Dynamically integrates multi-tiered, domain-specific textual cues to enhance the semantic alignment between visual and textual modalities.
- **Class-Balanced Resampling:** Adjusts sampling weights based on class-specific accuracy to mitigate bias in imbalanced datasets.
- **Cross-Architecture Knowledge Distillation:** Transfers predictive knowledge from a high-capacity ViT-based teacher model to a lightweight CLIP-based student model using L2 norm–based alignment, thereby preserving high performance while reducing computational cost.

## Repository Contents

- **/code:** Source code for implementing AMLPF-CLIP.
- **/configs:** Configuration files for training and evaluation.
- **/experiments:** Scripts and notebooks for reproducing our experimental results on benchmark datasets.
- **/docs:** Documentation and further details regarding architecture, training procedure, and hyper-parameter settings.
- **README.md:** This file.

## Status

The code for AMLPF-CLIP is currently under preparation and will be released shortly. We are finalizing our experiments and documentation to ensure reproducibility and ease of use.

<--
## How to Cite

If you use our code or refer to our work in your research, please cite our upcoming paper:

```
@article{yourpaper202X,
  title={AMLPF-CLIP: Advancing Histopathological Image Classification via Adaptive Prompt Fusion, Class-Balanced Resampling, and Cross-Architecture Knowledge Distillation},
  author={Your Name and Coauthors},
  journal={Journal/Conference Name},
  year={202X},
  publisher={Publisher}
}
```
-->

## Contact

For questions or further information, please contact [yaoxizhang2022@email.szu.edu.cn] or open an issue in this repository.

---

We hope this repository will facilitate further research and development in histopathological image analysis. Stay tuned for updates!
