# LISS-IV Cloud Removal & Surface Reconstruction

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%20CPU-EE4C2C.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An enterprise-grade Generative AI platform for automated cloud removal and surface reconstruction in LISS-IV satellite imagery, specifically targeting the North Eastern Region (NER) of India. 

**Developed for ISRO's BAH 2026 Hackathon by Team Antriksh.**

## 🚀 Features

- **Multimodal SAR-Fusion (Novel)**: Fuses C-band Sentinel-1 SAR data with LISS-IV optical data to reconstruct occluded areas based on structural ground-truth.
- **CPU-Optimized Inference**: All models (LaMa, U-Net) are exported to ONNX for 3-5x faster inference on CPU-only hardware.
- **Geospatial Integrity**: Full GeoTIFF support. Processes data while preserving CRS (EPSG:32644) and metadata.
- **Interactive Web App**: A production-ready Streamlit dashboard for real-time visualization, metric comparison, and interactive dataset exploration.
- **Spectral Fidelity**: Uses a custom Spectral Angle Mapper (SAM) loss during training to ensure NIR band consistency for downstream LULC and NDVI tasks.

## 🏗️ Architecture

```mermaid
graph TD
    A[Bhuvan LISS-IV] --> C(Preprocessing & Tiling)
    B[Copernicus SAR] --> C
    C --> D{Cloud Detection}
    D -->|Mask| E[Model Router]
    
    E -->|<10% Cloud| F[OpenCV Baseline]
    E -->|10-60% Cloud| G[LaMa Inpainting]
    E -->|>60% Cloud| H[SAR-Fusion U-Net]
    
    F --> I(Gaussian Stitching)
    G --> I
    H --> I
    I --> J[Cloud-Free GeoTIFF]
```

## ⚙️ Installation

### 1. Local Environment
```bash
# Clone the repository
git clone https://github.com/team-antriksh/liss4_cloud_removal.git
cd liss4_cloud_removal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Docker (Recommended for exact reproducibility)
```bash
docker-compose up --build
# App will be available at http://localhost:8501
```

## 🔑 Configuration

1. Copy `.env.example` to `.env`
2. Add your ISRO Bhuvan and ESA Copernicus credentials:
```env
BHUVAN_USERNAME=your_username
BHUVAN_PASSWORD=your_password
COPERNICUS_USER=your_username
COPERNICUS_PASSWORD=your_password
DEVICE=cpu
```

## 💻 Usage

### Launching the Dashboard
```bash
streamlit run app/streamlit_app.py
```

### Running the Pipeline via CLI
```python
from pipeline.cloud_detection import CloudDetectionPipeline
from models.lama.inference import LaMaInference

# Initialize
detector = CloudDetectionPipeline()
model = LaMaInference('configs/lama_config.yaml')

# Process
mask, _ = detector.process(cloudy_image)
cloud_free = model.inpaint(cloudy_image, mask)
```

## 📊 Evaluation Metrics

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | SAM ↓ | CPU Inference (256px) |
|-------|--------|--------|---------|-------|-----------------------|
| OpenCV Baseline | 22.4 | 0.72 | 0.35 | 0.45 | **~0.01s** |
| LaMa Inpainting | 28.1 | 0.86 | 0.18 | 0.22 | ~0.4s |
| SAR-Fusion U-Net| **31.5** | **0.92** | **0.12** | **0.15** | ~0.6s |

## 📚 Documentation
- [Technical Report](docs/TECHNICAL_REPORT.md)
- [API Reference](docs/API_REFERENCE.md)

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements
- [LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://github.com/saic-mdal/lama)
- ISRO NRSC for providing the Bhuvan LISS-IV portal.
