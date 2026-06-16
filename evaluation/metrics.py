import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute PSNR between 0-255 uint8 arrays."""
    psnr_metric = PeakSignalNoiseRatio(data_range=255.0)
    pred_t = torch.from_numpy(pred).unsqueeze(0).float()
    target_t = torch.from_numpy(target).unsqueeze(0).float()
    return psnr_metric(pred_t, target_t).item()

def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute SSIM between 0-255 uint8 arrays."""
    # Ensure channel first for torchmetrics: [B, C, H, W]
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float()
    target_t = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=255.0)
    return ssim_metric(pred_t, target_t).item()

def compute_lpips(pred: np.ndarray, target: np.ndarray, device='cpu') -> float:
    """Compute LPIPS perceptual loss."""
    # LPIPS expects input in range [-1, 1] and [B, C, H, W]
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    target_t = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    
    loss_fn = lpips.LPIPS(net='vgg', verbose=False).to(device)
    with torch.no_grad():
        score = loss_fn(pred_t, target_t).item()
    return score

def compute_sam(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Spectral Angle Mapper (SAM) in radians.
    SAM measures the spectral similarity between two images.
    """
    # Flatten spatial dimensions
    pred_flat = pred.astype(np.float32).reshape(-1, pred.shape[2])
    target_flat = target.astype(np.float32).reshape(-1, target.shape[2])
    
    # Compute dot product
    dot_product = np.sum(pred_flat * target_flat, axis=1)
    
    # Compute magnitudes
    pred_norm = np.linalg.norm(pred_flat, axis=1)
    target_norm = np.linalg.norm(target_flat, axis=1)
    
    # Avoid division by zero
    norms = pred_norm * target_norm
    norms[norms == 0] = 1e-10
    
    cos_theta = np.clip(dot_product / norms, -1.0, 1.0)
    sam_angles = np.arccos(cos_theta)
    
    return np.mean(sam_angles)

def compute_rmse(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute Root Mean Square Error globally and per-band."""
    rmse_total = np.sqrt(np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2))
    
    band_rmse = {}
    band_names = ['Green', 'Red', 'NIR']
    for i in range(min(pred.shape[2], len(band_names))):
        rmse_b = np.sqrt(np.mean((pred[:,:,i].astype(np.float32) - target[:,:,i].astype(np.float32)) ** 2))
        band_rmse[f'RMSE_{band_names[i]}'] = float(rmse_b)
        
    return {'RMSE_Total': float(rmse_total), **band_rmse}

class MetricsCalculator:
    def evaluate(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """Runs all metrics on the prediction and target pairs."""
        logger.info("Computing evaluation metrics...")
        
        metrics = {}
        metrics['PSNR'] = compute_psnr(pred, target)
        metrics['SSIM'] = compute_ssim(pred, target)
        try:
            metrics['LPIPS'] = compute_lpips(pred, target)
        except Exception as e:
            logger.warning(f"Failed to compute LPIPS (requires internet for VGG weights): {e}")
            metrics['LPIPS'] = None
            
        metrics['SAM'] = compute_sam(pred, target)
        
        rmse_metrics = compute_rmse(pred, target)
        metrics.update(rmse_metrics)
        
        return metrics
