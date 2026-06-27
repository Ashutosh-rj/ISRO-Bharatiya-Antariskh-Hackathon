import streamlit as st

st.set_page_config(page_title="Solution Architecture", page_icon="🏗️", layout="wide")
st.title("🏗️ Solution Architecture")

st.markdown("""
Our solution directly answers the problem statement's call for a **Generative AI-based framework** by implementing two advanced deep learning approaches:
1. A **Generative Adversarial Network (cGAN)** for spatial reconstruction (LaMa).
2. A **Multi-modal Fusion U-Net** utilizing auxiliary **Sentinel-1 SAR imagery**.
""")

st.header("1. Preserving Fine-Scale Spatial Details (cGAN)")
st.markdown("""
To preserve fine-scale spatial details, we utilize a Conditional GAN (cGAN) approach, specifically adapted from the LaMa (Large Mask Inpainting) architecture.
- **Why it works:** Generative Adversarial Networks excel at hallucinating realistic textures.
- **Application:** For regions where clouds obscure LISS-IV optical data and no valid SAR reference is available, the cGAN infers the missing ground structure based on the surrounding spatial context.
""")

st.header("2. Multi-Modal Fusion (Sentinel-1 SAR + LISS-IV)")
st.markdown("""
The problem statement notes that traditional masking leads to information loss. To reconstruct missing optical data, we fuse **LISS-IV optical imagery** with **Sentinel-1 SAR imagery** (C-band radar).
- **SAR Penetration:** Sentinel-1 C-band SAR radar backscatter (VV/VH) penetrates cloud cover, providing surface roughness and boundary geometry (buildings, water bodies, agricultural field edges) underneath the clouds. *(Note: Dedicated DEM elevation raster fusion is planned as an operational Phase 2 extension).*
- **Dual-Encoder U-Net:** Our custom architecture uses a Cross-Attention Bottleneck to align SAR features with Optical features, guiding the reconstruction of masked regions with ground-truth radar data.
""")

col1, col2 = st.columns(2)
with col1:
    st.info("""
    **Optical Encoder**
    - Extracts visible structural features from LISS-IV (Green, Red, NIR).
    - Identifies missing data using the Cloud Mask.
    """)
with col2:
    st.success("""
    **SAR Encoder**
    - Processes Sentinel-1 VV and VH polarizations.
    - Extracts unaffected terrain and infrastructure structure.
    """)

st.header("3. Spectral Consistency")
st.markdown("""
A key requirement is maintaining **spectral consistency** for downstream tasks like **land use–land cover mapping**. 
- **SAM Loss:** During the training of our Generative AI models, we employ a Spectral Angle Mapper (SAM) loss function.
- **Impact:** This ensures the predicted spectral signature (especially in the critical NIR band) exactly matches the true ground signature, making our cloud-free outputs perfectly valid for NDVI calculation.
""")
