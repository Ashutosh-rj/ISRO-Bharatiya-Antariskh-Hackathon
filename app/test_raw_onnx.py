import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from models.lama.inference import LaMaInference

lama = LaMaInference("configs/lama_config.yaml")

img = np.ones((512, 512, 3), dtype=np.uint8) * 128
mask = np.zeros((512, 512), dtype=np.uint8)

img_f = img.astype(np.float32) / 255.0
mask_f = mask.astype(np.float32)

img_t = np.expand_dims(np.transpose(img_f, (2, 0, 1)), 0)
mask_t = np.expand_dims(np.expand_dims(mask_f, 0), 0)

ort_inputs = {'image': img_t, 'mask': mask_t}
ort_outs = lama.ort_session.run(None, ort_inputs)

pred_t = ort_outs[0]
print("Raw ONNX output min:", pred_t.min(), "max:", pred_t.max(), "mean:", pred_t.mean())
