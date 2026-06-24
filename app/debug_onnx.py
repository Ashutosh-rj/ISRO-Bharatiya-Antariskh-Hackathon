import cv2
import numpy as np
import sys
import os
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess = ort.InferenceSession("models/lama/weights/lama.onnx", sess_options, providers=['CPUExecutionProvider'])

img = cv2.imread("data/real_samples/cloudy_0.png", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

from pipeline.cloud_detection import CloudDetectionPipeline
detector = CloudDetectionPipeline()
mask, _, _ = detector.detector.detect(img)

img = cv2.resize(img, (512, 512))
mask = cv2.resize(mask, (512, 512))

img_t = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
img_t = np.expand_dims(img_t, 0)

mask_t = np.expand_dims(mask.astype(np.float32), 0)
mask_t = np.expand_dims(mask_t, 0)

print(img_t.shape, mask_t.shape)

ort_inputs = {'image': img_t, 'mask': mask_t}
ort_outs = sess.run(None, ort_inputs)

pred = ort_outs[0][0]
print("Out shape:", pred.shape, "min:", pred.min(), "max:", pred.max(), "mean:", pred.mean())

pred = np.transpose(pred, (1, 2, 0))
pred = np.clip(pred, 0, 255).astype(np.uint8)
pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
cv2.imwrite("results/debug_onnx.png", pred)
print("Saved debug_onnx.png")
