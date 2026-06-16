import pytest
import numpy as np
from evaluation.metrics import compute_psnr, compute_ssim, compute_rmse

def test_psnr():
    img1 = np.ones((256, 256, 3), dtype=np.uint8) * 128
    img2 = np.ones((256, 256, 3), dtype=np.uint8) * 128
    
    psnr = compute_psnr(img1, img2)
    assert psnr > 40.0 # Identical images have high PSNR (theoretically inf, but torchmetrics caps or returns high val)

def test_ssim():
    img1 = np.ones((256, 256, 3), dtype=np.uint8) * 128
    img2 = np.ones((256, 256, 3), dtype=np.uint8) * 128
    
    ssim = compute_ssim(img1, img2)
    assert np.isclose(ssim, 1.0) # Identical images have SSIM of 1

def test_rmse():
    img1 = np.ones((256, 256, 3), dtype=np.uint8) * 10
    img2 = np.ones((256, 256, 3), dtype=np.uint8) * 20
    
    metrics = compute_rmse(img1, img2)
    assert metrics['RMSE_Total'] == 10.0
