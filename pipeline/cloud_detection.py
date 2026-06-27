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
    Cloud detection using multi-band Haze-Optimized Transformation (HOT) index and NDSI masking.
    Combines atmospheric scattering response with NDSI vegetation suppression before Otsu thresholding.
    Prevents bright sand, rooftops, and NER India tropical vegetation from being flagged as clouds.
    """
    def __init__(self, nir_band_index: int = 2):
        # Default assumes LISS-IV band order: Green (0), Red (1), NIR (2)
        self.nir_band_index = nir_band_index

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(image.shape) != 3 or image.shape[2] < 3:
            logger.error("Image does not have the required 3 multispectral bands.")
            raise ValueError("Invalid image dimensions for OtsuCloudDetector.")
        
        # Multi-band brightness composite across Green (0), Red (1), NIR (2)
        if image.dtype != np.uint8:
            img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = image
            
        img_f = img_uint8.astype(np.float32) / 255.0
        green = img_f[:, :, 0]
        red = img_f[:, :, 1]
        nir = img_f[:, :, 2]
        
        # 1. HOT (Haze-Optimized Transformation) Index: captures atmospheric scattering
        # Clear sky line slope is typically around ~0.5 between red and green
        hot_index = green - 0.5 * red
        
        # 2. NDSI (Normalized Difference Spectral Index) (G - NIR)/(G + NIR)
        # Separates achromatic clouds/haze from highly reflective NIR vegetation
        ndsi = (green - nir) / (green + nir + 1e-8)
        
        # Combine HOT index and brightness while suppressing vegetation false positives
        cloud_response = np.clip((hot_index * 0.6 + green * 0.4) * 255.0, 0, 255).astype(np.uint8)
        
        # Suppress tropical vegetation and false positives (where NDSI is very negative)
        cloud_response[ndsi < -0.25] = 0

        # Apply Gaussian Blur to reduce high-frequency noise
        blurred = cv2.GaussianBlur(cloud_response, (5, 5), 0)
        
        # Multi-band Otsu's thresholding on HOT/NDSI composite
        _, cloud_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Estimate opacity gradient (0-1)
        opacity_map = (blurred.astype(np.float32) / 255.0)
        opacity_map[cloud_mask == 0] = 0.0 # Clear regions
        
        # Cloud Shadow Detection via NIR lower tail thresholding
        nir_band = img_uint8[:, :, min(2, img_uint8.shape[2]-1)]
        shadow_mask = (nir_band < 40).astype(np.uint8) * 255
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Filter shadows near cloud projections
        cloud_dilated = cv2.dilate((cloud_mask > 0).astype(np.uint8), kernel, iterations=10)
        shadow_mask = cv2.bitwise_and(shadow_mask, shadow_mask, mask=cloud_dilated)
        
        return (cloud_mask > 0).astype(np.uint8), (shadow_mask > 0).astype(np.uint8), opacity_map

class AdaptiveThresholdDetector(BaseCloudDetector):
    """
    Adaptive thresholding based cloud detector leveraging spatial illumination variance across visible + NIR bands.
    """
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logger.info("Adaptive Multi-band Threshold detection invoked.")
        if image.dtype != np.uint8:
            img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = image
            
        composite = np.mean(img_uint8[:, :, :3], axis=2).astype(np.uint8)
        cloud_mask = cv2.adaptiveThreshold(composite, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 15, -3)
        opacity_map = (composite.astype(np.float32) / 255.0)
        opacity_map[cloud_mask == 0] = 0.0
        shadow_mask = np.zeros_like(cloud_mask)
        
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
        elif method.lower() == 'adaptive':
            self.detector = AdaptiveThresholdDetector()
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Process the image and return the cloud mask, shadow mask, opacity map, and cloud coverage percentage.
        """
        cloud_mask, shadow_mask, opacity_map = self.detector.detect(image)
        cloud_percentage = (np.sum(cloud_mask) / cloud_mask.size) * 100.0
        return cloud_mask, shadow_mask, opacity_map, cloud_percentage
