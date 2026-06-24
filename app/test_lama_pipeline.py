import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from models.lama.inference import LaMaInference

lama = LaMaInference("configs/lama_config.yaml")

img = cv2.imread("data/real_samples/cloudy_0.png", cv2.IMREAD_COLOR)
# load the same mask
from app.cloud_detection import CloudDetector
detector = CloudDetector()
mask = detector.detect_clouds(img)

print("Image shape:", img.shape, "Mask shape:", mask.shape)

res = lama.inpaint(img, mask)

print("Final result min:", res.min(), "max:", res.max(), "mean:", res.mean(), "dtype:", res.dtype)
cv2.imwrite("results/test_lama_out.png", res)
