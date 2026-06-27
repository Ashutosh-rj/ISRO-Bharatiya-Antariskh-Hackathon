import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from models.sar_fusion.inference import SARFusionInference

sar = SARFusionInference("configs/sar_fusion_config.yaml")

img = cv2.imread("data/real_samples/cloudy_0.png", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

from pipeline.cloud_detection import CloudDetectionPipeline
detector = CloudDetectionPipeline()
mask, _, _, _ = detector.process(img)

img = cv2.resize(img, (512, 512))
mask = cv2.resize(mask, (512, 512))

res = sar.infer(img, mask)

print("SAR Output min:", res.min(), "max:", res.max(), "mean:", res.mean(), "dtype:", res.dtype)
