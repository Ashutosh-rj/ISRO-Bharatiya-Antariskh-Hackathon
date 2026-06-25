import os
import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.lama.model import LaMaGenerator
from models.sar_fusion.model import SARFusionUNet

def save_image(img, path, title):
    # Ensure it's in 0-255 uint8 format
    if img.max() <= 1.0:
        img = (img * 255.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # plt.imsave expects RGB, OpenCV expects BGR if using cv2.imwrite. 
    # Let's just use matplotlib since it handles RGB naturally.
    plt.imsave(path, img)

def main():
    # Load first patch
    test_dir = os.path.join(project_root, "data", "processed", "train")
    files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.npz')]
    if not files:
        print("No NPZ files found.")
        return
        
    data = np.load(files[0])
    cloudy = data.get('cloudy')
    mask = data.get('mask')
    sar = data.get('sar')
    if sar is None:
        sar = np.zeros((cloudy.shape[0], cloudy.shape[1], 2), dtype=cloudy.dtype)

    # Convert to torch
    device = torch.device('cpu')
    c_t = torch.from_numpy(cloudy).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    m_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    s_t = torch.from_numpy(sar).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    print("Loading models...")
    lama = LaMaGenerator(in_channels=4, out_channels=3).to(device)
    lama.load_state_dict(torch.load(os.path.join(project_root, "models/lama/weights/lama_big.pth"), map_location=device))
    lama.eval()

    sar_fusion = SARFusionUNet(optical_channels=4, sar_channels=2, out_channels=3, base_filters=64).to(device)
    sar_fusion.load_state_dict(torch.load(os.path.join(project_root, "models/sar_fusion/weights/sar_fusion_final.pth"), map_location=device), strict=False)
    sar_fusion.eval()

    print("Running inference...")
    with torch.no_grad():
        lama_out_t = lama(c_t, m_t)
        sar_out_t = sar_fusion(c_t, m_t, s_t)
        if isinstance(sar_out_t, tuple):
            sar_out_t = sar_out_t[0]

    lama_out = (lama_out_t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    sar_out = (sar_out_t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

    diff = np.abs(lama_out.astype(float) - sar_out.astype(float)).mean()
    print(f"Mean absolute difference between LaMa and SAR-Fusion outputs: {diff:.4f}")

    if diff < 1.0:
        print("RED FLAG: Outputs are virtually identical!")
    else:
        print("SUCCESS: Outputs are distinct.")

    # Save images
    res_dir = os.path.join(project_root, "results")
    save_image(cloudy, os.path.join(res_dir, "verify_cloudy.png"), "Cloudy Input")
    save_image(lama_out, os.path.join(res_dir, "verify_lama.png"), "LaMa Output")
    save_image(sar_out, os.path.join(res_dir, "verify_sar_fusion.png"), "SAR-Fusion Output")
    print(f"Saved PNGs to {res_dir}")

if __name__ == '__main__':
    main()
