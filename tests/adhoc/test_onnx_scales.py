import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

from models.lama.inference import LaMaInference

lama = LaMaInference("configs/lama_config.yaml")

# Test 1: input [0, 1]
img = np.ones((512, 512, 3), dtype=np.uint8) * 128
mask = np.zeros((512, 512), dtype=np.uint8)

img_f_1 = img.astype(np.float32) / 255.0
mask_f = mask.astype(np.float32)

img_t_1 = np.expand_dims(np.transpose(img_f_1, (2, 0, 1)), 0)
mask_t = np.expand_dims(np.expand_dims(mask_f, 0), 0)

ort_inputs_1 = {'image': img_t_1, 'mask': mask_t}
ort_outs_1 = lama.ort_session.run(None, ort_inputs_1)[0]

# Test 2: input [0, 255]
img_f_2 = img.astype(np.float32)
img_t_2 = np.expand_dims(np.transpose(img_f_2, (2, 0, 1)), 0)
ort_inputs_2 = {'image': img_t_2, 'mask': mask_t}
ort_outs_2 = lama.ort_session.run(None, ort_inputs_2)[0]

print("Test 1 (input 0-1) output mean:", ort_outs_1.mean())
print("Test 2 (input 0-255) output mean:", ort_outs_2.mean())

