import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Tuple

logger = logging.getLogger(__name__)

class BaseCloudDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect clouds in the given image. 
        Returns (cloud_mask, shadow_mask, opacity_map).
        """
        pass

class OtsuCloudDetector(BaseCloudDetector):
    """
    Cloud detection using Otsu's thresholding on the NIR band.
    Fast and suitable for a baseline CPU implementation.
    """
    def __init__(self, nir_band_index: int = 2):
        # Default assumes LISS-IV band order: Green (0), Red (1), NIR (2)
        self.nir_band_index = nir_band_index

    def detect(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) != 3 or image.shape[2] <= self.nir_band_index:
            logger.error("Image does not have the required NIR band.")
            raise ValueError("Invalid image dimensions for OtsuCloudDetector.")
        
        nir_band = image[:, :, self.nir_band_index]
        
        # Normalize to 8-bit for OpenCV Otsu
        if nir_band.dtype != np.uint8:
            nir_band_norm = cv2.normalize(nir_band, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            nir_band_norm = nir_band

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(nir_band_norm, (5, 5), 0)
        
        # Otsu's thresholding
        _, cloud_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Estimate opacity (Thin cloud detection)
        # Scale blurred NIR to [0, 1] range representing opacity
        opacity_map = (blurred.astype(np.float32) / 255.0)
        opacity_map[opacity_map < 0.2] = 0.0 # Clear regions
        
        # Cloud Shadow Detection (Phase 6.2)
        # Shadows are extremely dark in NIR.
        _, shadow_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # We need a different threshold for shadows, typically the lower tail of the histogram.
        # For simplicity, we define shadows as NIR < 30 (assuming 8-bit normalized).
        shadow_mask = (blurred < 30).astype(np.uint8) * 255
        
        # Morphological cleanup for shadow
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Filter out shadows that aren't near clouds (Shadow projection heuristic)
        cloud_dilated = cv2.dilate((cloud_mask > 0).astype(np.uint8), kernel, iterations=10)
        shadow_mask = cv2.bitwise_and(shadow_mask, shadow_mask, mask=cloud_dilated)
        
        return (cloud_mask > 0).astype(np.uint8), (shadow_mask > 0).astype(np.uint8), opacity_map

class FmaskCloudDetector(BaseCloudDetector):
    """
    Simplified Fmask (Function of mask) algorithm wrapper.
    In a full implementation, this would integrate with the Fmask python package.
    """
    def detect(self, image: np.ndarray) -> np.ndarray:
        logger.info("Fmask detection invoked. Using simplified fallback for this demo.")
        # Fallback to Otsu since true Fmask requires TOA reflectance and thermal bands
        # which LISS-IV lacks (LISS-IV only has Green, Red, NIR).
        # We will emulate an advanced detector using adaptive thresholding.
        
        nir_band = image[:, :, 2] if image.shape[2] >= 3 else image[:, :, 0]
        if nir_band.dtype != np.uint8:
            nir_band = cv2.normalize(nir_band, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
        # Adaptive thresholding
        cloud_mask = cv2.adaptiveThreshold(nir_band, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, -2)
        opacity_map = (nir_band.astype(np.float32) / 255.0)
        shadow_mask = np.zeros_like(cloud_mask) # Simplified
        
        return (cloud_mask > 0).astype(np.uint8), shadow_mask, opacity_map

class HazeRemovalModule:
    """
    Phase 6.3: Haze Removal using Dark Channel Prior
    """
    def __init__(self, window_size=15, omega=0.95):
        self.window_size = window_size
        self.omega = omega

    def get_dark_channel(self, img):
        b, g, r = cv2.split(img)
        min_img = cv2.min(cv2.min(b, g), r)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.window_size, self.window_size))
        dark = cv2.erode(min_img, kernel)
        return dark

    def remove_haze(self, img: np.ndarray) -> np.ndarray:
        # Standard Dark Channel Prior implementation
        # For remote sensing, haze is mostly in the visible bands.
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
        dark = self.get_dark_channel(img)
        # Simplified atmospheric light estimation
        A = np.percentile(dark, 99.9)
        
        # Transmission map
        transmission = 1 - self.omega * self.get_dark_channel(img / max(A, 1))
        transmission = np.clip(transmission, 0.1, 1.0)
        
        # Recover
        J = np.empty_like(img, dtype=np.float32)
        for i in range(img.shape[2]):
            J[:,:,i] = (img[:,:,i] - A) / transmission + A
            
        return np.clip(J, 0, 255).astype(np.uint8)

class CloudDetectionPipeline:
    def __init__(self, method: str = 'otsu'):
        if method.lower() == 'otsu':
            self.detector = OtsuCloudDetector()
        elif method.lower() == 'fmask':
            self.detector = FmaskCloudDetector()
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Process the image and return the cloud mask, shadow mask, opacity map, and cloud coverage percentage.
        """
        cloud_mask, shadow_mask, opacity_map = self.detector.detect(image)
        cloud_percentage = (np.sum(cloud_mask) / cloud_mask.size) * 100.0
        return cloud_mask, shadow_mask, opacity_map, cloud_percentage
