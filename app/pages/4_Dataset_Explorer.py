import streamlit as st
import folium


st.set_page_config(page_title="Dataset Explorer", page_icon="🌍", layout="wide")
st.title("🌍 NER Dataset Explorer")

st.markdown("""
The models are trained and validated on data from the **North Eastern Region (NER)** of India, characterized by hilly terrain, dense vegetation, and persistent cloud cover.
""")

# Create Map
m = folium.Map(location=[26.14, 91.73], zoom_start=6) # Center around Assam/NER

# Add bounding box for sample region
folium.Rectangle(
    bounds=[[25.0, 90.0], [28.0, 95.0]],
    color='#FF6B35',
    fill=True,
    fill_color='#FF6B35',
    fill_opacity=0.2,
    popup="NER Training Region"
).add_to(m)

import streamlit.components.v1 as components
components.html(m._repr_html_(), height=500)

st.subheader("Data Modalities Used")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1. LISS-IV Optical")
    st.markdown("Source: Bhuvan\nResolution: 5.8m\nBands: Green, Red, NIR")
    
with col2:
    st.markdown("### 2. Sentinel-1 SAR")
    st.markdown("Source: Copernicus\nResolution: 10m\nBands: VV, VH")
    
with col3:
    st.markdown("### 3. Sentinel-2 (Reference)")
    st.markdown("Source: Copernicus\nResolution: 10m\nUsage: Cross-validation")
