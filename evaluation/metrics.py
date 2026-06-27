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

def compute_ergas(pred: np.ndarray, target: np.ndarray, ratio: float = 1.0) -> float:
    """
    Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).
    Widely used in remote sensing pan-sharpening and cross-sensor image synthesis.
    
    Args:
        pred: Predicted synthesized image array [H, W, C].
        target: Reference ground truth image array [H, W, C].
        ratio: Spatial resolution ratio between high-res and low-res sensors (h/l).
               Defaults to 1.0 for same-resolution inpainting benchmarks (e.g., SEN12MS-CR 10m to 10m).
               For cross-sensor super-resolution (e.g., 10m Sentinel-2 to 5.8m LISS-IV), pass ratio=0.58 (5.8/10.0).
    """
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    
    mean_target = np.mean(target, axis=(0, 1))
    rmse_bands = np.sqrt(np.mean((pred - target)**2, axis=(0, 1)))
    
    # Avoid division by zero
    mean_target[mean_target == 0] = 1e-10
    
    sum_ratio = np.sum((rmse_bands / mean_target)**2)
    ergas = 100.0 * ratio * np.sqrt(sum_ratio / pred.shape[2])
    return float(ergas)

def compute_scc(pred: np.ndarray, target: np.ndarray) -> float:
    """Spatial Correlation Coefficient (SCC)."""
    # Apply high-pass filter (Sobel or simple Laplacian)
    import cv2
    pred_gray = cv2.cvtColor(pred.astype(np.float32), cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_RGB2GRAY)
    
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    hp_pred = cv2.filter2D(pred_gray, -1, kernel)
    hp_target = cv2.filter2D(target_gray, -1, kernel)
    
    # Compute correlation
    hp_pred_flat = hp_pred.flatten()
    hp_target_flat = hp_target.flatten()
    
    correlation = np.corrcoef(hp_pred_flat, hp_target_flat)[0, 1]
    return float(correlation)

def compute_ndvi_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Computes RMSE of the NDVI index.
    Assumes bands are Green (0), Red (1), NIR (2) for LISS-IV.
    """
    if pred.shape[2] < 3:
        return 0.0
        
    def get_ndvi(img):
        img_f = img.astype(np.float32)
        red = img_f[:, :, 1]
        nir = img_f[:, :, 2]
        denominator = (nir + red)
        denominator[denominator == 0] = 1e-10
        return (nir - red) / denominator
        
    pred_ndvi = get_ndvi(pred)
    target_ndvi = get_ndvi(target)
    
    return float(np.sqrt(np.mean((pred_ndvi - target_ndvi)**2)))

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
    def evaluate(self, pred: np.ndarray, target: np.ndarray, ergas_ratio: float = 1.0) -> Dict[str, Any]:
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
        metrics['MAE'] = float(np.mean(np.abs(pred.astype(np.float32) - target.astype(np.float32))))
        metrics['ERGAS'] = compute_ergas(pred, target, ratio=ergas_ratio)
        
        try:
            metrics['SCC'] = compute_scc(pred, target)
        except Exception:
            metrics['SCC'] = 0.0
            
        metrics['NDVI_RMSE'] = compute_ndvi_rmse(pred, target)
        
        rmse_metrics = compute_rmse(pred, target)
        metrics.update(rmse_metrics)
        
        return metrics
