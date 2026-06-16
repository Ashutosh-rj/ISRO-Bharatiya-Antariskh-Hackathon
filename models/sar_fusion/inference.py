import os
import torch
import numpy as np
import logging
import yaml
import onnxruntime as ort
from models.sar_fusion.model import SARFusionUNet

logger = logging.getLogger(__name__)

class SARFusionInference:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.use_onnx = self.config['inference']['use_onnx']
        
        if self.use_onnx:
            onnx_path = self.config['inference']['onnx_path']
            if not os.path.exists(onnx_path):
                logger.warning("ONNX weights not found. Fallback to PyTorch.")
                self.use_onnx = False
            else:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 4
                self.ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
                
        if not self.use_onnx:
            pth_path = self.config['inference']['pytorch_path']
            self.model = SARFusionUNet(
                base_filters=self.config['model']['architecture']['base_filters']
            )
            if os.path.exists(pth_path):
                self.model.load_state_dict(torch.load(pth_path, map_location='cpu'))
            self.model.eval()

    def infer(self, optical: np.ndarray, mask: np.ndarray, sar: np.ndarray = None) -> np.ndarray:
        is_uint8 = optical.dtype == np.uint8
        
        opt_f = optical.astype(np.float32) / 255.0 if is_uint8 else optical.astype(np.float32)
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0: mask_f /= 255.0
            
        # Handle missing SAR by passing zeros
        if sar is None:
            sar_f = np.zeros((opt_f.shape[0], opt_f.shape[1], 2), dtype=np.float32)
        else:
            sar_f = sar.astype(np.float32) / 255.0 if is_uint8 else sar.astype(np.float32)
            
        opt_t = np.expand_dims(np.transpose(opt_f, (2, 0, 1)), 0)
        mask_t = np.expand_dims(np.expand_dims(mask_f, 0), 0)
        sar_t = np.expand_dims(np.transpose(sar_f, (2, 0, 1)), 0)
        
        # Pad to multiple of 16 for U-Net
        pad_mult = 16
        h, w = opt_t.shape[2], opt_t.shape[3]
        pad_h = (pad_mult - h % pad_mult) % pad_mult
        pad_w = (pad_mult - w % pad_mult) % pad_mult
        
        if pad_h > 0 or pad_w > 0:
            opt_t = np.pad(opt_t, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='reflect')
            mask_t = np.pad(mask_t, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='reflect')
            sar_t = np.pad(sar_t, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='reflect')

        if self.use_onnx:
            ort_inputs = {'optical': opt_t, 'mask': mask_t, 'sar': sar_t}
            pred_t = self.ort_session.run(None, ort_inputs)[0]
        else:
            with torch.no_grad():
                pred_t = self.model(
                    torch.from_numpy(opt_t), 
                    torch.from_numpy(mask_t), 
                    torch.from_numpy(sar_t)
                ).numpy()
                
        if pad_h > 0 or pad_w > 0:
            pred_t = pred_t[:, :, :h, :w]
            
        pred = np.transpose(pred_t[0], (1, 2, 0))
        
        if is_uint8:
            pred = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
        else:
            pred = np.clip(pred, 0.0, 1.0)
            
        return pred
