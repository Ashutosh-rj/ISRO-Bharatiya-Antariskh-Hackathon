# JURY DEFENSE PACKAGE
## Team CloudBusters (Formerly Antriksh) - ISRO BAH 2026

This document serves as the master technical defense for the Grand Finale Jury.

### 1. Scientific Authenticity
- **Spectral Physics:** The pipeline explicitly preserves the true G-R-NIR 3-band structure of LISS-IV sensor data.
- **Metric Validity:** We discarded PSNR/SSIM as primary metrics and migrated to domain-specific measures: SAM (Spectral Angle Mapper), ERGAS, and NDVI-RMSE.
- **Dataset Realism:** Replaced toy synthetic fractals with SEN12MS-CR, utilizing real Sentinel-1 (SAR) and Sentinel-2 (Optical) physics to simulate LISS-IV cloud-cover dynamics.

### 2. Generative AI Architecture
- **cGAN + PatchGAN:** The deterministic U-Net was upgraded to a Conditional GAN. The Generator fuses SAR + Optical via a `CrossModalTransformer`, while the `PatchGANDiscriminator` enforces localized textural realism.
- **Diffusion Refinement:** A Conditional DDPM module iteratively refines the generated outputs to perfectly align with the high-frequency surface distribution of the target region.

### 3. Multi-Modal & Temporal Fusion
- **SAR Integration:** Sentinel-1 C-band VV/VH data penetrates clouds.
- **Cross-Modal Transformer:** Replaced basic concatenation with a Multi-Head Attention mechanism allowing Optical queries to extract features from SAR keys/values.
- **Temporal Transformer:** Implemented time-series persistence modeling (T-1 to T-4) to guarantee structural immutability (e.g., roads, buildings).

### 4. Cloud & Haze Understanding
- **Opacity Mapping:** Thin clouds are no longer binarized; they generate an opacity gradient (0-1).
- **Shadow Detection:** NIR morphological heuristics extract and map cloud shadows to prevent them from being reconstructed as lakes/waterbodies.
- **Haze Removal:** Dark Channel Prior applied to visible bands.

### 5. Explainable AI & Engineering
- **MCD Uncertainty Heatmaps:** Monte Carlo Dropout provides a variance map (uncertainty), critical for downstream GIS operations.
- **Mixed Precision (AMP) & W&B:** Deployed for rapid training and rigorous experiment tracking.

**Final Score Target:** 88-92 / 100.
