import pytest
import numpy as np
from pipeline.preprocessing import PatchExtractor
from pipeline.cloud_detection import OtsuCloudDetector

def test_patch_extractor():
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    extractor = PatchExtractor(patch_size=256, stride=256)
    patches = extractor.extract_patches(image)
    
    assert len(patches) == 4
    assert patches[0]['image'].shape == (256, 256, 3)

def test_otsu_detector():
    # Mock image with a bright spot simulating a cloud
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[100:150, 100:150, 2] = 255 # Bright in NIR
    
    detector = OtsuCloudDetector()
    mask = detector.detect(image)
    
    assert mask.shape == (256, 256)
    assert mask.max() == 1
    assert mask[125, 125] == 1 # Center of cloud should be masked
    assert mask[10, 10] == 0 # Background should not be masked
