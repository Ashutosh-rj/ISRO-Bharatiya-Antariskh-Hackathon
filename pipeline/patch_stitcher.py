import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class GaussianPatchStitcher:
    """
    Reassembles overlapping image patches back into a full-resolution image
    using Gaussian blending to remove seam lines.
    """
    def __init__(self, image_shape: Tuple[int, int, int], patch_size: int = 256, stride: int = 64):
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.stride = stride
        
        # Create Gaussian weighting window
        self.weight_window = self._create_gaussian_window(patch_size)
        
        # Accumulators
        self.reconstructed = np.zeros(image_shape, dtype=np.float32)
        self.weight_sum = np.zeros(image_shape[:2] + (1,), dtype=np.float32)

    def _create_gaussian_window(self, size: int) -> np.ndarray:
        """Create a 2D Gaussian window for blending."""
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        x, y = np.meshgrid(x, y)
        d = np.sqrt(x*x + y*y)
        sigma, mu = 0.5, 0.0
        g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
        # Normalize to max 1
        g = g / np.max(g)
        # Add channel dimension
        return np.expand_dims(g, axis=-1)

    def add_patch(self, patch: np.ndarray, y: int, x: int):
        """
        Adds a predicted patch to the accumulators.
        patch shape should be (H, W, C)
        """
        h, w = patch.shape[:2]
        
        # Ensure patch fits within the image bounds
        y_end = min(y + h, self.image_shape[0])
        x_end = min(x + w, self.image_shape[1])
        
        patch_h = y_end - y
        patch_w = x_end - x
        
        p = patch[:patch_h, :patch_w]
        wnd = self.weight_window[:patch_h, :patch_w]
        
        self.reconstructed[y:y_end, x:x_end] += p * wnd
        self.weight_sum[y:y_end, x:x_end] += wnd

    def get_result(self) -> np.ndarray:
        """
        Finalizes the blending and returns the reconstructed image.
        """
        # Avoid division by zero
        eps = 1e-8
        result = self.reconstructed / (self.weight_sum + eps)
        
        # Clip to valid range and cast
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
