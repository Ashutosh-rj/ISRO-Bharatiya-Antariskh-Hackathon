import streamlit as st
import yaml

st.set_page_config(
    page_title="ISRO BAH 2026: Cloud Removal Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config and style
def load_css():
    with open('app/assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Sidebar
with st.sidebar:
    st.markdown("---")
    st.title("Generative AI for LISS-IV")
    st.markdown("""
    **Team Antriksh**
    
    A Generative AI-based framework for automated cloud removal and surface reconstruction in LISS-IV imagery.
    """)
    st.markdown("---")
    st.info("Aligned with ISRO BAH 2026")

# Main Content
st.title("🛰️ LISS-IV Cloud Removal & Reconstruction")

st.markdown("""
### Generative AI-Based Framework

Persistent cloud cover is a major challenge in optical remote sensing, particularly over tropical and mountainous regions such as the North Eastern Region (NER) of India. Clouds and cloud shadows significantly reduce the usability of optical satellite imagery for applications such as **land use–land cover mapping**, **disaster monitoring**, and **environmental assessment**.

This platform demonstrates our solution to **automated cloud removal and surface reconstruction in LISS-IV imagery** while preserving fine-scale spatial details and spectral consistency.

#### Key Innovations:
1. **Generative AI (cGAN/LaMa):** A robust Conditional GAN for high-fidelity spatial reconstruction.
2. **Multi-Modal Fusion:** Fusing LISS-IV optical imagery with auxiliary **Sentinel-1 SAR imagery** to penetrate clouds and retrieve ground structure.
3. **Spectral Consistency:** Employing Spectral Angle Mapper (SAM) loss to preserve the integrity of the NIR band for downstream tasks (e.g., NDVI).

👈 **Select a page from the sidebar to explore the platform.**
""")

# Quick Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Primary Data", value="LISS-IV")
with col2:
    st.metric(label="Auxiliary Data", value="Sentinel-1 SAR")
with col3:
    st.metric(label="Target Region", value="NER India")

st.info("👈 Select **Live Demo** to run inference and explore the pipeline.")
