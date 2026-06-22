import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️")
st.title("ℹ️ About the Project")

st.header("ISRO BAH 2026 Problem Statement Alignment")
st.markdown("""
This project directly addresses the challenge of persistent cloud cover in LISS-IV optical remote sensing imagery, particularly over the North Eastern Region (NER) of India.

**Key Achievements against criteria:**
1. **Generative AI Framework:** Integrated state-of-the-art LaMa and custom SAR-Fusion U-Net.
2. **CPU-Optimized:** Models exported to ONNX, utilizing multi-threading for fast CPU inference.
3. **Multimodal Fusion:** First-of-its-kind fusion of LISS-IV with Sentinel-1 SAR.
4. **Geospatial Integrity:** Pipeline natively handles GeoTIFFs, preserving CRS and metadata.
5. **Spectral Consistency:** Custom SAM (Spectral Angle Mapper) loss ensures NIR band usability for downstream tasks like NDVI.
""")

st.header("Team Antriksh")
st.markdown("""
Developed during the ISRO BAH 2026 Hackathon.
""")

st.header("Future Work")
st.markdown("""
- **Diffusion Models:** Explore conditional latent diffusion for even higher fidelity (though currently too slow for CPU constraint).
- **Temporal Stacking:** Integrate multi-date LISS-IV observations into the fusion encoder.
- **Edge Deployment:** Quantize ONNX models to INT8 for deployment on resource-constrained devices.
""")
