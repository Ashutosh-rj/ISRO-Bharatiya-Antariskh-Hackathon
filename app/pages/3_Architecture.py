import streamlit as st

st.set_page_config(page_title="Architecture", page_icon="🏗️", layout="wide")
st.title("🏗️ System Architecture")

st.header("Overall Pipeline")
st.markdown("""
1. **Data Acquisition**: Bhuvan (LISS-IV) + Copernicus (Sentinel-1 SAR)
2. **Preprocessing**: Otsu Cloud Masking + 256x256 Patch Extraction
3. **Generative Inference**: CPU-optimized ONNX model execution
4. **Geospatial Assembly**: Gaussian blending and GeoTIFF CRS preservation
""")

st.header("Novel SAR-Fusion U-Net")
st.markdown("""
Our primary innovation is fusing C-band SAR data (which penetrates clouds) with LISS-IV optical data.
""")

col1, col2 = st.columns(2)
with col1:
    st.info("""
    **Optical Encoder (4-channel)**
    - Processes LISS-IV Green, Red, NIR + Cloud Mask
    - Extracts visible structural features
    """)
    st.info("""
    **SAR Encoder (2-channel)**
    - Processes Sentinel-1 VV and VH polarizations
    - Extracts terrain and infrastructure structure
    """)
with col2:
    st.warning("""
    **Cross-Attention Bottleneck**
    - Aligns SAR features with Optical features
    - Guides the reconstruction of masked regions
    """)
    st.success("""
    **Reconstruction Decoder**
    - Predicts cloud-free Green, Red, NIR
    - Uses SAM loss to preserve spectral fidelity
    """)
