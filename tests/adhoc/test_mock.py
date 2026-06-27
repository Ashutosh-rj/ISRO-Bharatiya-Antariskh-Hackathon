import cv2
import numpy as np
import sys
sys.path.append('/app')

img = cv2.imread("data/real_samples/cloudy_0.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

from pipeline.cloud_detection import CloudDetectionPipeline
detector = CloudDetectionPipeline()
mask, _, _, _ = detector.process(img)

ref_cv = cv2.imread("data/real_samples/clear_reference.png")
ref_cv = cv2.cvtColor(ref_cv, cv2.COLOR_BGR2RGB)
ref_cv = cv2.resize(ref_cv, (img.shape[1], img.shape[0]))

mask_3c = np.stack([mask]*3, axis=-1)
if mask_3c.max() > 1.0: mask_3c = mask_3c / 255.0

result = (img * (1 - mask_3c) + ref_cv * mask_3c).astype(np.uint8)

cv2.imwrite("results/debug_mock.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print("Mock min:", result.min(), "max:", result.max(), "mean:", result.mean())
