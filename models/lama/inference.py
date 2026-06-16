import os
import torch
import numpy as np
import logging
import yaml
import onnxruntime as ort
from models.lama.model import LaMaGenerator

logger = logging.getLogger(__name__)

class LaMaInference:
    """
    Inference wrapper for LaMa. Supports both PyTorch (.pth) and ONNX (.onnx) backends.
    ONNX is highly recommended for CPU-only execution.
    """
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.use_onnx = self.config['inference']['use_onnx']
        
        if self.use_onnx:
            onnx_path = self.config['inference']['onnx_path']
            if not os.path.exists(onnx_path):
                logger.warning(f"ONNX weights not found at {onnx_path}. Attempting to fallback to PyTorch.")
                self.use_onnx = False
            else:
                logger.info(f"Loading LaMa ONNX session from {onnx_path}")
                # Set CPU provider options for max performance
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 4
                self.ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
                
        if not self.use_onnx:
            pth_path = self.config['inference']['pytorch_path']
            logger.info(f"Loading LaMa PyTorch weights from {pth_path}")
            self.model = LaMaGenerator()
            if os.path.exists(pth_path):
                self.model.load_state_dict(torch.load(pth_path, map_location='cpu'))
            else:
                logger.warning(f"PyTorch weights not found at {pth_path}. Using random initialization!")
            self.model.eval()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        image: (H, W, 3) float32 [0-1] or uint8 [0-255]
        mask: (H, W) binary mask 0 or 1
        Returns inpainted image of same type and shape.
        """
        is_uint8 = image.dtype == np.uint8
        
        # Preprocess
        img_f = image.astype(np.float32) / 255.0 if is_uint8 else image.astype(np.float32)
        mask_f = mask.astype(np.float32)
        
        if mask_f.max() > 1.0:
            mask_f = mask_f / 255.0
            
        # Ensure dimensions [1, C, H, W]
        img_t = np.expand_dims(np.transpose(img_f, (2, 0, 1)), 0)
        mask_t = np.expand_dims(np.expand_dims(mask_f, 0), 0)
        
        # Pad to multiple of 8 (required by FFT ops)
        pad_mult = self.config['inference']['pad_to_multiple']
        h, w = img_t.shape[2], img_t.shape[3]
        pad_h = (pad_mult - h % pad_mult) % pad_mult
        pad_w = (pad_mult - w % pad_mult) % pad_mult
        
        if pad_h > 0 or pad_w > 0:
            img_t = np.pad(img_t, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='reflect')
            mask_t = np.pad(mask_t, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='reflect')

        # Run Inference
        if self.use_onnx:
            ort_inputs = {'image': img_t, 'mask': mask_t}
            ort_outs = self.ort_session.run(None, ort_inputs)
            pred_t = ort_outs[0]
        else:
            with torch.no_grad():
                img_ts = torch.from_numpy(img_t)
                mask_ts = torch.from_numpy(mask_t)
                pred_ts = self.model(img_ts, mask_ts)
                pred_t = pred_ts.numpy()
                
        # Postprocess
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            pred_t = pred_t[:, :, :h, :w]
            
        pred = np.transpose(pred_t[0], (1, 2, 0))
        
        if is_uint8:
            pred = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
        else:
            pred = np.clip(pred, 0.0, 1.0)
            
        return pred
