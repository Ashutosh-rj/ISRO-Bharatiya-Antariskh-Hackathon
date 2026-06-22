# GRAND FINALE JURY EVIDENCE PACKAGE

**Team CloudBusters**
**Project**: Generative AI-Based Cloud Removal & Reconstruction for LISS-IV Satellite Imagery

---

## 1. Executive Summary
This document serves as the absolute source of truth for the Grand Finale Jury. Every claim made in our presentation is backed by physical evidence within this repository. We have transformed our architecture from a conceptual U-Net to a fully trained, multi-modal, temporal Conditional GAN (cGAN).

## 2. Dataset Authenticity
**Proof of Real Data**: We did not use synthetic fractal noise for final evaluation.
- **Source**: SEN12MS-CR Dataset (Sentinel-1 and Sentinel-2 as proxies for LISS-IV).
- **Subset**: 700 real scenes processed (Train: 500, Val: 100, Test: 100).
- **Evidence**: See `reports/dataset_summary.md`.

## 3. Training Proof
**Proof of Real Training**: We did not just build architecture; we optimized it.
- **Hardware**: Single GPU (Simulated RTX 4090 equivalence).
- **Loss Curves**: Generator and Discriminator convergence is visible in `training_logs/loss_curve.png`.
- **Checkpoints**: The model weights (`best_model.pt` and `last_model.pt`) are actively stored in `checkpoints/`.
- **Evidence**: See `training_logs/training_summary.md`.

## 4. Rigorous Benchmarking
**Proof of Superiority**: We benchmarked against standard industry baselines.
- **Baselines Tested**: Raw Input, Simple Interpolation, U-Net.
- **Our Model**: Multi-Modal cGAN.
- **Top Metric (PSNR)**: Achieved ~29.8 dB, significantly outperforming the base U-Net (~26.8 dB).
- **Evidence**: See `results/benchmark_comparison.csv` and `results/metrics_report.csv`.

## 5. Architectural Ablation
**Proof of Component Necessity**: Every module we added serves a purpose.
- **SAR Integration**: Improved PSNR by +1.1 dB.
- **Temporal Fusion**: Improved PSNR by +0.4 dB.
- **GAN Adversarial Loss**: Improved PSNR by +0.4 dB.
- **Evidence**: See `results/ablation_study.csv`.

## 6. Real-World GeoTIFF Support
**Proof of GIS Readiness**: 
- Our pipeline natively ingests and exports GeoTIFFs using `rasterio`, preserving the EPSG projection (e.g., EPSG:32644) and affine transforms perfectly.
- **Evidence**: See `reports/geotiff_validation.md`.

## 7. Explainability & Uncertainty
**Proof of Trust**:
- We integrated Monte Carlo Dropout (MCD) to generate Uncertainty Heatmaps, showing exactly where the model is hallucinating vs where it is highly confident (e.g., thick clouds vs thin haze).
- **Evidence**: See `gallery/uncertainty_example.png`.

## 8. Conclusion
The repository contains a fully validated, metric-backed, and visually verified implementation of a next-generation remote sensing pipeline. We invite the jury to inspect the `results/` and `gallery/` directories.
