# Hackathon Submission Checklist

## Phase 1: Code & Environment 
- [x] Python 3.10+ environment specified
- [x] CPU-only PyTorch and ONNX configurations active
- [x] `requirements.txt` included with pinned versions
- [x] Dockerfile provided for reproducible builds

## Phase 2: Pipeline Implementation
- [x] Data download scripts (Bhuvan WMS, Copernicus)
- [x] Automated cloud masking (Otsu NIR)
- [x] 256x256 Patch extraction & Gaussian stitching
- [x] GeoTIFF CRS (EPSG:32644) preservation

## Phase 3: Models
- [x] Baseline OpenCV Inpainting
- [x] LaMa Inpainting (Generative)
- [x] SAR-Fusion Dual-Encoder U-Net (Generative + Multimodal)
- [x] ONNX Export pipeline for all deep models

## Phase 4: Evaluation & UI
- [x] PSNR, SSIM, SAM, RMSE metrics implemented
- [x] Interactive Streamlit App with 5 detailed pages
- [x] Before/After split-screen slider

## Phase 5: Documentation
- [x] `README.md` with installation/usage
- [x] `TECHNICAL_REPORT.md` outlining methodology
- [x] `architecture_diagram` included (via Mermaid in README)
