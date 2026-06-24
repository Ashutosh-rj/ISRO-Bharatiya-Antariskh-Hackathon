# JURY DEFENSE PACKAGE
## Team Antriksh - ISRO BAH 2026

This document serves as the master technical defense for the Grand Finale Jury.

### 1. Scientific Authenticity
- **Spectral Physics:** The pipeline explicitly preserves the true G-R-NIR 3-band structure of LISS-IV sensor data.
- **Metric Validity:** We discarded PSNR/SSIM as primary metrics and migrated to domain-specific measures: SAM (Spectral Angle Mapper), ERGAS, and NDVI-RMSE.
- **Dataset Realism:** Replaced toy synthetic fractals with SEN12MS-CR, utilizing real Sentinel-1 (SAR) and Sentinel-2 (Optical) physics to simulate LISS-IV cloud-cover dynamics.

### 2. Generative AI Architecture
- **cGAN + PatchGAN:** The deterministic U-Net was upgraded to a Conditional GAN. The Generator fuses SAR + Optical via a `CrossModalTransformer`, while the `PatchGANDiscriminator` enforces localized textural realism.

### 3. Multi-Modal Fusion
- **SAR Integration:** Sentinel-1 C-band VV/VH data penetrates clouds.
- **Cross-Modal Transformer:** Replaced basic concatenation with a Multi-Head Attention mechanism allowing Optical queries to extract features from SAR keys/values.

### 4. Cloud & Haze Understanding
- **Opacity Mapping:** Thin clouds are no longer binarized; they generate an opacity gradient (0-1).
- **Shadow Detection:** NIR morphological heuristics extract and map cloud shadows to prevent them from being reconstructed as lakes/waterbodies.
- **Haze Removal:** Dark Channel Prior applied to visible bands.

### 5. LISS-IV vs. Sentinel Domain Gap
While our architecture is designed for high-resolution multispectral data, we acknowledge the domain gap between our training dataset (SEN12MS-CR) and the target LISS-IV sensor. LISS-IV has different spectral response functions, spatial resolution (5.8m vs 10m/20m), and noise profiles compared to Sentinel-2. 
**Mitigation Plan:** Our solution is architected to support transfer learning. The current weights provide strong structural and cross-modal priors. For true LISS-IV operational deployment, we propose a fine-tuning phase using an adversarial domain adaptation loss (e.g., CycleGAN or a domain classifier) on unpaired clear LISS-IV images to adapt the Sentinel-trained generator to the specific radiometric properties of the LISS-IV sensor without requiring perfectly paired cloudy/clear LISS-IV datasets.
