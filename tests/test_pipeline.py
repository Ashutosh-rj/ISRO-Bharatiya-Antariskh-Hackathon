import unittest
import numpy as np
from scripts.generate_dataset import generate_fractal_noise

class TestDatasetGeneration(unittest.TestCase):
    
    def test_fractal_noise_generation(self):
        shape = (128, 128)
        noise = generate_fractal_noise(shape, octaves=3, persistence=0.5)
        
        # Check shape
        self.assertEqual(noise.shape, shape)
        
        # Check normalization (should be roughly between 0 and 1)
        self.assertTrue(np.max(noise) <= 1.0 + 1e-5)
        self.assertTrue(np.min(noise) >= 0.0 - 1e-5)
        
        # Check that it's not just a flat array
        self.assertTrue(np.std(noise) > 0.1)

if __name__ == '__main__':
    unittest.main()
