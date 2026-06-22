import os
import numpy as np
import cv2

def generate_fractal_noise(shape, octaves=4, persistence=0.5):
    h, w = shape
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    frequency = 4
    
    for _ in range(octaves):
        base_noise = np.random.rand(h // frequency, w // frequency).astype(np.float32)
        scaled_noise = cv2.resize(base_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        noise += scaled_noise * amplitude
        amplitude *= persistence
        frequency = max(1, frequency // 2)
        
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

def create_synthetic_pair(output_dir, prefix):
    # Create random "clear" satellite-like image
    clear_img = np.random.uniform(50, 150, size=(256, 256, 3)).astype(np.uint8)
    
    # Add some structural features
    cv2.rectangle(clear_img, (50, 50), (100, 100), (80, 100, 70), -1)
    cv2.circle(clear_img, (200, 150), 30, (60, 60, 120), -1)
    
    h, w, c = clear_img.shape
    
    # Generate realistic fractal noise
    noise = generate_fractal_noise((h, w), octaves=5, persistence=0.5)
    
    # Threshold to create cloud mask
    threshold = np.random.uniform(0.4, 0.7)
    mask = (noise > threshold).astype(np.float32)
    
    # Smooth the mask edges
    mask_blurred = cv2.GaussianBlur(mask, (21, 21), 0)
    
    # Create cloud layer
    cloud_color = np.random.uniform(200, 255, size=(1, 1, 3))
    cloud_layer = np.ones_like(clear_img) * cloud_color
    
    # Blend
    mask_3d = np.expand_dims(mask_blurred, axis=-1)
    cloudy_img = clear_img * (1 - mask_3d) + cloud_layer * mask_3d
    cloudy_img = np.clip(cloudy_img, 0, 255).astype(np.uint8)
    
    out_dict = {
        'cloud_free': clear_img,
        'cloudy': cloudy_img,
        'mask': (mask_blurred > 0.1).astype(np.uint8)
    }
    
    out_path = os.path.join(output_dir, "train", prefix)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **out_dict)
    return out_path

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    
    print("Generating 10 synthetic shim data pairs...")
    for i in range(10):
        path = create_synthetic_pair(output_dir, f"shim_data_{i:03d}.npz")
        print(f"Created {path}")

if __name__ == "__main__":
    main()
