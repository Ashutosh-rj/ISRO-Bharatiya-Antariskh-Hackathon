import unittest
import numpy as np
from pipeline.preprocessing import PatchExtractor
from pipeline.cloud_detection import OtsuCloudDetector

class TestPreprocessing(unittest.TestCase):
    def test_patch_extractor(self):
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        extractor = PatchExtractor(patch_size=256, stride=256)
        patches = extractor.extract_patches(image)
        
        self.assertEqual(len(patches), 4)
        self.assertEqual(patches[0]['image'].shape, (256, 256, 3))

    def test_otsu_detector(self):
        # Mock image with a bright spot simulating a cloud (bright across Green, Red, NIR)
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[100:150, 100:150, :] = 240 # Bright in all bands (achromatic cloud scattering)
        
        detector = OtsuCloudDetector()
        mask, shadow, opacity = detector.detect(image)
        
        self.assertEqual(mask.shape, (256, 256))
        self.assertEqual(mask.max(), 1)
        self.assertEqual(mask[125, 125], 1) # Center of cloud should be masked
        self.assertEqual(mask[10, 10], 0) # Background should not be masked

if __name__ == '__main__':
    unittest.main()
