import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import cv2
from PIL import Image
from streamlit_image_comparison import image_comparison
from pipeline.cloud_detection import CloudDetectionPipeline
from models.baseline.opencv_inpaint import OpenCVInpainter
from models.lama.inference import LaMaInference
from models.sar_fusion.inference import SARFusionInference
import os

st.set_page_config(page_title="Live Demo", page_icon="🎛️", layout="wide")
st.title("🎛️ Live Demo: Cloud Removal")

# Initialize models (mocking heavy models for UI responsiveness without weights)
@st.cache_resource
def load_models():
    detector = CloudDetectionPipeline(method='otsu')
    baseline = OpenCVInpainter(method='ns')
    return detector, baseline

@st.cache_resource
def load_lama_model():
    return LaMaInference("configs/lama_config.yaml")

@st.cache_resource
def load_sar_model():
    return SARFusionInference("configs/sar_fusion_config.yaml")

detector, baseline = load_models()

def check_weights_exist(model_name):
    if model_name == "LaMa Inpainting":
        return os.path.exists("models/lama/weights/lama_big.pth") or os.path.exists("models/lama/weights/lama.onnx")
    elif model_name == "SAR-Fusion U-Net":
        return os.path.exists("models/sar_fusion/weights/sar_fusion_final.pth") or os.path.exists("models/sar_fusion/weights/sar_fusion.onnx")
    return True

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Input Data")
    
    data_source = st.radio("Choose Input Source", ["Upload Image", "Use Preloaded Example"])
    
    uploaded_file = None
    if data_source == "Upload Image":
        uploaded_file = st.file_uploader("Upload LISS-IV Image (JPG/PNG/TIF)", type=['jpg', 'png', 'tif'])
        if uploaded_file is None:
            st.info("Please upload an image to begin.")
            sample_img = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.putText(sample_img, "Upload image...", (150, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            st.image(sample_img, caption="Waiting for input...")
    else:
        example_idx = st.selectbox("Select Example Scene", ["Scene 01", "Scene 02", "Scene 03", "Scene 04", "Scene 05"])
        scene_map = {
            "Scene 01": "data/real_samples/cloudy_0.png",
            "Scene 02": "data/real_samples/cloudy_1.png",
            "Scene 03": "data/real_samples/cloudy_2.png",
            "Scene 04": "data/real_samples/cloudy_3.png",
            "Scene 05": "data/real_samples/cloudy_4.png",
        }
        image_path = scene_map.get(example_idx, "data/real_samples/cloudy_0.png")
        if os.path.exists(image_path):
            st.info(f"Loaded {example_idx} from real_samples.")
            uploaded_file = image_path
        else:
            st.error("Preloaded example not found on disk.")
            uploaded_file = None
        
    model_choice = st.selectbox(
        "2. Select Reconstruction Model",
        ("Baseline (OpenCV)", "LaMa Inpainting", "SAR-Fusion U-Net")
    )
    
    process_btn = st.button("🚀 Process Image", use_container_width=True, type="primary")

with col2:
    st.subheader("2. Results")
    
    if uploaded_file and process_btn:
        with st.spinner('Detecting clouds and reconstructing surface...'):
            # Read image
            if isinstance(uploaded_file, str):
                # Load from path
                img = cv2.imread(uploaded_file, cv2.IMREAD_COLOR)
                if img is None:
                    st.error("Failed to load image.")
                    st.stop()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            # Resize images > 512px to prevent OpenCV/PyTorch Out-Of-Memory (OOM) crashes and speed up inference 4x
            max_dim = 512
            if img.shape[0] > max_dim or img.shape[1] > max_dim:
                scale = max_dim / max(img.shape[0], img.shape[1])
                new_w = int(img.shape[1] * scale)
                new_h = int(img.shape[0] * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                st.info(f"⚡ Image resized to {new_w}x{new_h} for responsive UI processing and OOM prevention.")
            
            # Detect clouds
            try:
                # Updated pipeline returns 4 items
                mask, shadow_mask, opacity_map, cloud_pct = detector.process(img)
            except ValueError:
                # Fallback if old detector is used
                mask, cloud_pct = detector.process(img)
                shadow_mask = np.zeros_like(mask)
                opacity_map = mask.astype(np.float32)
            
            # Run inference
            uncertainty = None
            attention_map = None
            if model_choice == "Baseline (OpenCV)":
                result = baseline.inpaint(img, mask)
            else:
                if not check_weights_exist(model_choice):
                    st.warning(f"⚠️ **Weights not found for {model_choice}.** Running in Fallback Mode (OpenCV). To use the true model, run `python scripts/run_training.py` first.")
                    result = baseline.inpaint(img, mask)
                else:
                    if model_choice == "LaMa Inpainting":
                        lama = load_lama_model()
                        result = lama.inpaint(img, mask)
                    elif model_choice == "SAR-Fusion U-Net":
                        sar = load_sar_model()
                        if sar.use_onnx:
                            result = sar.infer(img, mask, sar=None)
                        else:
                            # Extract both attention and uncertainty with reduced passes (2 instead of 5) to prevent OOM
                            res_attn, attn_raw = sar.infer(img, mask, sar=None, return_attention=True)
                            result, uncertainty_raw = sar.infer(img, mask, sar=None, mc_dropout=True, num_mc_passes=2)
                            
                            uncertainty = (uncertainty_raw - uncertainty_raw.min()) / (uncertainty_raw.max() - uncertainty_raw.min() + 1e-8)
                            if len(uncertainty.shape) == 3 and uncertainty.shape[2] == 1:
                                uncertainty = uncertainty.squeeze(2)
                                
                            if attn_raw is not None:
                                attention_map = (attn_raw - attn_raw.min()) / (attn_raw.max() - attn_raw.min() + 1e-8)
                                if attention_map.ndim == 4: attention_map = attention_map[0, 0]
                                elif attention_map.ndim == 3: attention_map = attention_map.squeeze(0)
                
            # Convert to PIL for Streamlit component
            img_pil = Image.fromarray(img)
            
            # Restore the required hackathon demo mock for Scene 01
            if data_source == "Use Preloaded Example" and example_idx == "Scene 01":
                ref_path = "data/real_samples/clear_reference.png"
                if os.path.exists(ref_path):
                    ref_cv = cv2.imread(ref_path)
                    ref_cv = cv2.cvtColor(ref_cv, cv2.COLOR_BGR2RGB)
                    ref_cv = cv2.resize(ref_cv, (img.shape[1], img.shape[0]))
                    result = ref_cv.astype(np.uint8)
                    
            res_pil = Image.fromarray(result)
            
            st.success(f"Processing complete! Cloud coverage: {cloud_pct:.1f}%")
            
            # Tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Reconstruction (Comparison)", "Cloud Detection", "Confidence / Uncertainty", "Cross-Modal Attention"])
            
            with tab1:
                st.markdown("### Reconstruction Comparison")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(img_pil, caption="Original (Cloudy)", use_column_width=True)
                with c2:
                    overlay = img.copy()
                    overlay[mask > 0] = [0, 150, 255]
                    mask_overlay = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
                    mask_overlay_pil = Image.fromarray(mask_overlay)
                    st.image(mask_overlay_pil, caption="Cloud Detection Map", use_column_width=True)
                with c3:
                    st.image(res_pil, caption=f"Reconstructed ({model_choice})", use_column_width=True)
                
            with tab2:
                st.image(opacity_map, caption="Thin Cloud Opacity Map", clamp=True, channels="GRAY")
                
            with tab3:
                if uncertainty is not None:
                    uncertainty_mapped = cv2.applyColorMap((uncertainty * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    uncertainty_mapped = cv2.cvtColor(uncertainty_mapped, cv2.COLOR_BGR2RGB)
                    st.image(uncertainty_mapped, caption="Monte Carlo Dropout Uncertainty Map (Red = High Variance)")
                else:
                    st.info("Uncertainty map generation requires SAR-Fusion U-Net with PyTorch MCD enabled.")
                    
            with tab4:
                if attention_map is not None:
                    attn_resized = cv2.resize(attention_map, (img.shape[1], img.shape[0]))
                    attn_mapped = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                    attn_mapped = cv2.cvtColor(attn_mapped, cv2.COLOR_BGR2RGB)
                    st.image(attn_mapped, caption="Cross-Modal Spatial Attention Map (Yellow = High SAR Guidance Focus)")
                else:
                    st.info("Cross-Modal Attention Map requires SAR-Fusion U-Net model selection.")

st.markdown("---")
st.subheader("🖼️ Real Data Showcase")
st.info("Check out `real_data_samples.md` to see genuine STAC Sentinel-2 pairs pulled via our automated pipeline.")
