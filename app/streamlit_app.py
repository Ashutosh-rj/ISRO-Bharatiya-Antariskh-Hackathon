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
    
    Building a Generative AI platform prototype for 
    automated cloud removal and surface reconstruction in 
    LISS-IV imagery.
    """)
    st.markdown("---")
    st.info("CPU-Optimized Pipeline")

# Main Content
st.title("🛰️ LISS-IV Cloud Removal & Reconstruction")

st.markdown("""
### Welcome to the ISRO BAH 2026 Submission

Persistent cloud cover is a major challenge in optical remote sensing over tropical regions like North Eastern India. This platform demonstrates a state-of-the-art Generative AI pipeline for cloud removal:

1. **True Generative AI (cGAN):** A robust Conditional GAN with PatchGAN Discriminator.
2. **Multi-Modal Fusion:** Cross-Modal Transformers combining LISS-IV optical and Sentinel-1 SAR imagery.
3. **Temporal Reasoning:** Transformer-based modeling of historical cloud-free persistence.
4. **Explainable AI:** Monte Carlo Dropout (MCD) for Uncertainty Heatmaps and Attention Visualization.

👈 **Select a page from the sidebar to explore the platform.**
""")

# Quick Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Models Available", value="3")
with col2:
    st.metric(label="Target Area", value="North Eastern Region")
with col3:
    st.metric(label="Hardware Target", value="CPU Only")

st.info("👈 Select **Live Demo** to run inference, or **Model Comparison** to view benchmark results (if computed).")
