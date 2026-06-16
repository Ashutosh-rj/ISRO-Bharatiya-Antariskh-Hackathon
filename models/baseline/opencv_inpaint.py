import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class OpenCVInpainter:
    """
    Baseline cloud removal using traditional computer vision inpainting algorithms.
    Supports Telea (fast marching) and Navier-Stokes based methods.
    Requires no training and runs completely on CPU.
    """
    def __init__(self, method: str = 'ns', inpaint_radius: int = 5):
        """
        method: 'ns' for Navier-Stokes, 'telea' for Fast Marching
        inpaint_radius: Neighborhood size to consider for inpainting
        """
        if method.lower() == 'ns':
            self.method = cv2.INPAINT_NS
        elif method.lower() == 'telea':
            self.method = cv2.INPAINT_TELEA
        else:
            raise ValueError(f"Unknown inpaint method: {method}")
            
        self.inpaint_radius = inpaint_radius

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint the regions in the image where mask == 1.
        image: (H, W, C) uint8 or float32
        mask: (H, W) uint8 (binary 0/1 or 0/255)
        """
        if mask.max() == 1:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.astype(np.uint8)

        # OpenCV inpainting works on 8-bit, 1-channel or 3-channel images.
        is_float = False
        if image.dtype != np.uint8:
            is_float = True
            image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            image_uint8 = image

        logger.info(f"Running OpenCV Inpainting with radius {self.inpaint_radius}")
        
        # Expand mask slightly to cover cloud borders perfectly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)

        result = cv2.inpaint(image_uint8, dilated_mask, self.inpaint_radius, self.method)

        if is_float:
            # Convert back to float (assuming original was 0-1)
            result = result.astype(np.float32) / 255.0
            
        return result
