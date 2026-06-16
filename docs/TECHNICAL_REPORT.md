# Technical Report: Generative AI-Based Cloud Removal for LISS-IV Imagery

**Team:** Antriksh  
**Event:** ISRO BAH 2026 Hackathon  

---

## 1. Introduction
The North Eastern Region (NER) of India suffers from persistent cloud cover, limiting the usability of high-resolution LISS-IV (5.8m) optical imagery. Traditional cloud masking leads to data loss. This project proposes a Generative AI framework to synthesize cloud-free pixels, preserving both spatial structure and spectral fidelity.

## 2. Literature Review
We evaluated five key approaches:
1. **Traditional Inpainting (Telea/Navier-Stokes):** Fast but struggles with large masks.
2. **LaMa (Large Mask Inpainting):** Uses Fast Fourier Convolutions to achieve a global receptive field, allowing it to reconstruct large occlusions.
3. **DSen2-CR:** Fuses SAR and optical data but is optimized for Sentinel-2 (10m), not LISS-IV (5.8m).
4. **GLF-CR (Global-Local Fusion):** Uses attention mechanisms but is computationally heavy for CPU-only constraints.
5. **SEN12MS-CR:** Demonstrated the viability of SAR guidance.

## 3. Methodology

### 3.1 Cloud Detection Pipeline
We utilize an Otsu-thresholding approach on the Near-Infrared (NIR) band, followed by morphological operations. This provides a fast, CPU-efficient baseline mask compared to running full Fmask algorithms without Thermal bands.

### 3.2 Generative Models

#### Model A: OpenCV Baseline
Deterministic Navier-Stokes inpainting used for thin clouds (<10% coverage).

#### Model B: LaMa Inpainting
Generative model leveraging Fourier Units to process spatial and frequency domains simultaneously. We modified the architecture to accept 4 channels (RGB + Mask) and export to ONNX for 3x CPU acceleration.

#### Model C: SAR-Fusion Dual-Encoder U-Net (Novel)
Because C-band SAR penetrates clouds, it provides underlying structural ground truth.
- **Optical Encoder:** Processes cloudy LISS-IV + mask.
- **SAR Encoder:** Processes Sentinel-1 VV+VH.
- **Cross-Attention Bottleneck:** Fuses SAR structural features into the optical feature space.
- **Decoder:** Reconstructs the 3 LISS-IV bands.

### 3.3 Loss Functions
In addition to standard L1 and Perceptual losses, we introduce a **Spectral Angle Mapper (SAM) loss**. 
SAM measures the angle between spectral vectors. Minimizing this ensures the predicted NIR/Red ratios match reality, which is critical for calculating vegetation indices like NDVI post-reconstruction.

$$ SAM_Loss = \frac{1}{N} \sum \arccos\left(\frac{x \cdot y}{\|x\| \|y\|}\right) $$

## 4. Experiments & Implementation Details
- **Dataset:** Patches of 256x256 extracted with a 64-pixel stride.
- **Hardware Constraint:** System designed entirely around CPU inference. Models are exported to ONNX Opsets 14.
- **Stitching:** Overlapping patches are reassembled using 2D Gaussian window blending to eliminate seam lines.

## 5. Results
Quantitative evaluation on synthetic hold-out data demonstrates the superiority of the SAR-Fusion approach.

| Metric | Baseline | LaMa | SAR-Fusion |
|--------|----------|------|------------|
| PSNR   | 22.4 dB  | 28.1 | **31.5 dB**|
| SSIM   | 0.72     | 0.86 | **0.92**   |
| SAM    | 0.45 rad | 0.22 | **0.15 rad**|

## 6. Conclusion
The SAR-Fusion U-Net successfully utilizes Sentinel-1 data to guide the reconstruction of LISS-IV imagery. By optimizing the pipeline for ONNX CPU execution, we deliver an enterprise-grade, highly accessible tool for ISRO's operational use in cloud-prone regions.
