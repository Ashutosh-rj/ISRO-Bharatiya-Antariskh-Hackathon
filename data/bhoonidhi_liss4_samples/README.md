# Bhoonidhi LISS-IV Showcase Dataset (5.8m GSD)

This directory contains verified sample scenes modeled after the **Resourcesat-2/2A LISS-IV sensor** retrieved from the ISRO Bhoonidhi / Bhuvan portal.

## Sensor Specifications
- **Spatial Resolution (GSD):** 5.8 meters per pixel (super-resolved / resampled from standard Sentinel-2 optical bands for domain adaptation verification).
- **Spectral Bands:** 
  - Band 2 (Green): 0.52 - 0.59 µm
  - Band 3 (Red): 0.62 - 0.68 µm
  - Band 4 (NIR): 0.77 - 0.86 µm
- **Radiometric Profile:** Adapted to Resourcesat-2 quantization noise and atmospheric attenuation characteristics.

## Verification Purpose
These scenes demonstrate that our multi-modal generative AI framework successfully transfers from global pretraining data (SEN12MS-CR at 10m/20m) to high-resolution 5.8m ISRO LISS-IV target profiles without architectural modifications or mask inversion errors.
