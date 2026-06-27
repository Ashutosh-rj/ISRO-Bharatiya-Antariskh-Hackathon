import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from models.lama.inference import LaMaInference

lama = LaMaInference("configs/lama_config.yaml")

# Create a test image with a small mask
img = np.ones((1024, 1024, 3), dtype=np.uint8) * 128
mask = np.zeros((1024, 1024), dtype=np.uint8)
mask[500:600, 500:600] = 1

res = lama.inpaint(img, mask)
print("Output min:", res.min(), "max:", res.max(), "mean:", res.mean())
print("Output dtype:", res.dtype)
