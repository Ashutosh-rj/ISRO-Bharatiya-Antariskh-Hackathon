import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import plotly.express as px
from evaluation.visualize import create_interactive_metrics_chart

import os

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
st.title("📊 Model Comparison")

st.markdown("""
This page compares the performance of the three implemented models across a synthetic validation set.
* **Baseline**: OpenCV Navier-Stokes inpainting
* **LaMa**: Generative Large Mask inpainting
* **SAR-Fusion**: Novel Dual-Encoder U-Net fusing optical and SAR data
""")

metrics_path = os.path.join("results", "benchmark_comparison.csv")

if not os.path.exists(metrics_path):
    st.warning("⚠️ **Benchmark results not found.** The training and evaluation pipeline has not been executed yet.")
    st.info("To generate real metrics, run `python scripts/run_training.py` in your environment, then run the benchmark script.")
    st.stop()

# Load real data
df = pd.read_csv(metrics_path)

# Handle column naming mismatch from RVP generation scripts
if 'Method' in df.columns:
    df.rename(columns={'Method': 'Model'}, inplace=True)

st.subheader("Quantitative Metrics")

# Plotly chart
fig = create_interactive_metrics_chart(df)
st.plotly_chart(fig, use_container_width=True)

# Data table
metrics_to_highlight_max = [m for m in ['PSNR', 'SSIM'] if m in df.columns]
metrics_to_highlight_min = [m for m in ['SAM', 'RMSE_Total', 'LPIPS'] if m in df.columns]

style = df.style
if metrics_to_highlight_max:
    style = style.highlight_max(subset=metrics_to_highlight_max, color='lightgreen')
if metrics_to_highlight_min:
    style = style.highlight_min(subset=metrics_to_highlight_min, color='lightgreen')

st.dataframe(style, use_container_width=True)

st.subheader("Qualitative Analysis (Pending Results)")
st.info("Upload sample outputs here to show side-by-side visual comparisons of specific ROIs after running the models.")
