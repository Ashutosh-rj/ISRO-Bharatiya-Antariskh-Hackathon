import os
import sys
import logging
import numpy as np
import cv2

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from pipeline.data_download import PlanetaryComputerDownloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GenerateDataset")

def process_s2_to_training_pair(s2_npz_path: str, output_dir: str):
    """
    Extracts cloudy areas using the SCL (Scene Classification Layer) mask from Sentinel-2.
    Builds the (cloudy, mask, cloud_free) dataset pairs.
    Note: A true cloud removal dataset needs temporally matched cloud-free and cloudy images.
    For this hackathon sprint, we will use a partially cloudy image:
    1. The original image is 'cloudy'.
    2. We use SCL to find clouds -> 'mask'.
    3. We use cv2 inpainting to generate a naive 'cloud_free' ground truth if no clear image is available,
       OR we just take a clear image and apply a synthetic mask to create 'cloudy'.
    Given the constraints, taking a CLEAR image and adding synthetic clouds is the most robust way 
    to get perfect ground truth for PSNR/SSIM evaluation.
    """
    data = np.load(s2_npz_path)
    optical = data['optical']
    scl = data['scl']
    
    # SCL Classes: 3=Cloud Shadows, 8=Cloud Medium Prob, 9=Cloud High Prob, 10=Thin Cirrus
    is_cloudy = np.isin(scl, [3, 8, 9, 10])
    cloud_coverage = np.mean(is_cloudy)
    
    logger.info(f"Image {s2_npz_path} has {cloud_coverage*100:.1f}% cloud coverage.")
    
    if cloud_coverage > 0.05:
        # It has natural clouds. We can use it as input, but what is the ground truth?
        # For training, we need ground truth. Let's save it as a test image.
        out_path = os.path.join(output_dir, "test", os.path.basename(s2_npz_path).replace('s2_', 'real_cloud_'))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, cloudy=optical, mask=is_cloudy.astype(np.uint8))
        return None
    else:
        # It is mostly clear! We will use this as Ground Truth and add synthetic clouds.
        # This guarantees perfect pairs for training and validation.
        return create_synthetic_cloud_pair(optical, output_dir, prefix=os.path.basename(s2_npz_path))


def generate_fractal_noise(shape, octaves=4, persistence=0.5):
    """Generates fractal noise for realistic cloud simulation."""
    h, w = shape
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    frequency = 4  # Start with a low frequency
    
    for _ in range(octaves):
        # Generate random noise at current frequency
        base_noise = np.random.rand(h // frequency, w // frequency).astype(np.float32)
        # Resize to full image
        scaled_noise = cv2.resize(base_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        noise += scaled_noise * amplitude
        
        amplitude *= persistence
        frequency = max(1, frequency // 2)
        
    # Normalize
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

def create_synthetic_cloud_pair(clear_img: np.ndarray, output_dir: str, prefix: str):
    """Adds synthetic clouds to a clear image to create training triplets."""
    h, w, c = clear_img.shape
    
    # Generate realistic fractal noise
    noise = generate_fractal_noise((h, w), octaves=5, persistence=0.5)
    
    # Threshold to create cloud mask
    threshold = np.random.uniform(0.4, 0.7)
    mask = (noise > threshold).astype(np.float32)
    
    # Smooth the mask edges
    mask_blurred = cv2.GaussianBlur(mask, (21, 21), 0)
    
    # Create cloud layer (white/gray)
    cloud_color = np.random.uniform(200, 255, size=(1, 1, 3))
    cloud_layer = np.ones_like(clear_img) * cloud_color
    
    # Blend
    mask_3d = np.expand_dims(mask_blurred, axis=-1)
    cloudy_img = clear_img * (1 - mask_3d) + cloud_layer * mask_3d
    cloudy_img = np.clip(cloudy_img, 0, 255).astype(np.uint8)
    
    out_dict = {
        'cloud_free': clear_img,
        'cloudy': cloudy_img,
        'mask': (mask_blurred > 0.1).astype(np.uint8) # Binary mask for loss
    }
    
    out_path = os.path.join(output_dir, "train", prefix)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **out_dict)
    logger.info(f"Created synthetic training pair: {out_path}")
    return out_dict


def main():
    logger.info("Starting Dataset Generation Pipeline...")
    downloader = PlanetaryComputerDownloader()
    
    # Bounding box for a region in India (e.g., near Bengaluru for varied terrain)
    bbox = [77.5, 12.9, 77.7, 13.1]
    
    # 1. Search for a clear Sentinel-2 image to serve as ground truth base
    logger.info("Searching for clear Optical data...")
    s2_items = downloader.search_sentinel2(bbox, "2023-01-01", "2023-03-31", max_cloud_cover=10.0)
    
    # 2. Search for Sentinel-1 SAR data for the same region/time
    logger.info("Searching for SAR data...")
    s1_items = downloader.search_sentinel1(bbox, "2023-01-01", "2023-03-31")
    
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Download 2 items to form a small dataset
    for i in range(min(2, len(s2_items))):
        s2_item = s2_items[i]
        success = downloader.download_s2_patch(s2_item, bbox, raw_dir)
        if success:
            process_s2_to_training_pair(os.path.join(raw_dir, f"s2_{s2_item.id}.npz"), processed_dir)
            
    for i in range(min(2, len(s1_items))):
        s1_item = s1_items[i]
        downloader.download_s1_patch(s1_item, raw_dir)
        
    logger.info("Dataset generation complete. Models can now be trained on real satellite structures.")

if __name__ == "__main__":
    main()
