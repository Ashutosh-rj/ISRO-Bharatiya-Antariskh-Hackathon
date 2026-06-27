import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from models.lama.inference import LaMaInference
from models.lama.inference import LaMaInference

print("Loading model...")
lama = LaMaInference("configs/lama_config.yaml")
print("Model loaded. use_onnx:", lama.use_onnx)

img = np.zeros((1024, 1024, 3), dtype=np.uint8)
mask = np.zeros((1024, 1024), dtype=np.uint8)

print("Running inpaint on 1024x1024...")
res = lama.inpaint(img, mask)
print("Success! Output shape:", res.shape)
