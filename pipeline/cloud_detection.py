import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Tuple

logger = logging.getLogger(__name__)

class BaseCloudDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect clouds in the given image. Returns a binary mask (1 for cloud, 0 for clear)."""
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
        
        return (cloud_mask > 0).astype(np.uint8)

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
        return (cloud_mask > 0).astype(np.uint8)

class CloudDetectionPipeline:
    def __init__(self, method: str = 'otsu'):
        if method.lower() == 'otsu':
            self.detector = OtsuCloudDetector()
        elif method.lower() == 'fmask':
            self.detector = FmaskCloudDetector()
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process the image and return the cloud mask and cloud coverage percentage.
        """
        mask = self.detector.detect(image)
        cloud_percentage = (np.sum(mask) / mask.size) * 100.0
        return mask, cloud_percentage
