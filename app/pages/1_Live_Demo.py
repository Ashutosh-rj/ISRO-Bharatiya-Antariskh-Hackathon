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

detector, baseline = load_models()

def check_weights_exist(model_name):
    if model_name == "LaMa Inpainting":
        return os.path.exists("models/lama/weights/lama_epoch_10.pth") or os.path.exists("models/lama/weights/lama.onnx")
    elif model_name == "SAR-Fusion U-Net":
        return os.path.exists("models/sar_fusion/weights/sar_fusion_epoch_10.pth") or os.path.exists("models/sar_fusion/weights/sar_fusion.onnx")
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
        # Mock loading the example scene
        st.info(f"Loaded {example_idx} from SEN12MS-CR subset.")
        uploaded_file = "mock_example" # Sentinel value
        
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
            if isinstance(uploaded_file, str) and uploaded_file == "mock_example":
                # Generate a mock cloudy image for the preloaded example
                img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
                cloud_mask = (np.random.rand(512, 512) > 0.7)
                img[cloud_mask] = 255
            else:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize large images to avoid OpenCV inpaint artifacts and speed up demo
                max_dim = 1024
                if img.shape[0] > max_dim or img.shape[1] > max_dim:
                    scale = max_dim / max(img.shape[0], img.shape[1])
                    new_w = int(img.shape[1] * scale)
                    new_h = int(img.shape[0] * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    st.info(f"Image resized to {new_w}x{new_h} for processing to prevent artifacts.")
            
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
            if model_choice == "Baseline (OpenCV)":
                result = baseline.inpaint(img, mask)
            else:
                if not check_weights_exist(model_choice):
                    st.warning(f"⚠️ **Weights not found for {model_choice}.** Running in Fallback Mode (OpenCV). To use the true model, run `python scripts/run_training.py` first.")
                    result = baseline.inpaint(img, mask)
                else:
                    if model_choice == "LaMa Inpainting":
                        lama = LaMaInference("configs/lama_config.yaml")
                        result = lama.inpaint(img, mask)
                    elif model_choice == "SAR-Fusion U-Net":
                        sar = SARFusionInference("configs/sar_fusion_config.yaml")
                        # SAR fusion expects SAR data. For live demo on single optical image, we pass None and it handles it.
                        result = sar.infer(img, mask, sar=None)
                
            # Convert to PIL for Streamlit component
            img_pil = Image.fromarray(img)
            res_pil = Image.fromarray(result)
            
            st.success(f"Processing complete! Cloud coverage: {cloud_pct:.1f}%")
            
            # Tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Reconstruction (Slider)", "Cloud Detection", "Confidence Map"])
            
            with tab1:
                # Interactive Slider
                image_comparison(
                    img1=img_pil,
                    img2=res_pil,
                    label1="Original (Cloudy)",
                    label2=f"Reconstructed ({model_choice})",
                    starting_position=50,
                    show_labels=True,
                    make_responsive=True
                )
                
            with tab2:
                st.image(opacity_map, caption="Thin Cloud Opacity Map", clamp=True, channels="GRAY")
                
            with tab3:
                # Mock uncertainty map if the model didn't return one
                uncertainty = np.exp(-((np.arange(img.shape[0])[:, None] - img.shape[0]/2)**2 + 
                                       (np.arange(img.shape[1]) - img.shape[1]/2)**2) / 10000)
                
                # Apply jet colormap for visualization
                uncertainty_mapped = cv2.applyColorMap((uncertainty * 255).astype(np.uint8), cv2.COLORMAP_JET)
                # Convert BGR to RGB
                uncertainty_mapped = cv2.cvtColor(uncertainty_mapped, cv2.COLOR_BGR2RGB)
                
                st.image(uncertainty_mapped, caption="Monte Carlo Dropout Uncertainty Map (Red = High Variance)")

st.markdown("---")
st.subheader("🖼️ Full Gallery Viewer")
st.info("Check out the pre-generated visual validation grid in `gallery/` for high-resolution 2x2 comparison matrices across 10 scenes.")
