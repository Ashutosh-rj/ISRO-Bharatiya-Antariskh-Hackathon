import unittest
import torch
import numpy as np
from models.lama.model import LaMaGenerator, ResidualFFCBlock
from models.sar_fusion.model import SARFusionUNet, CrossAttentionFusion
from models.losses import CombinedLoss

class TestModelArchitectures(unittest.TestCase):
    
    def setUp(self):
        self.device = torch.device('cpu')
        
    def test_lama_forward_pass(self):
        # 4 channels: RGB + Mask
        model = LaMaGenerator(in_channels=4, out_channels=3).to(self.device)
        
        # Batch size 2, 4 channels, 64x64 image
        dummy_img = torch.rand(2, 3, 64, 64).to(self.device)
        dummy_mask = torch.zeros(2, 1, 64, 64).to(self.device)
        
        # Forward pass
        output = model(dummy_img, dummy_mask)
        
        # Check output shape: (B, 3, H, W)
        self.assertEqual(output.shape, (2, 3, 64, 64))
        # Check output range
        self.assertTrue(torch.all((output >= 0) & (output <= 1)))
        
    def test_sar_fusion_forward_pass(self):
        model = SARFusionUNet(optical_channels=4, sar_channels=2, out_channels=3, base_filters=16, attention_heads=2).to(self.device)
        
        dummy_opt = torch.rand(2, 4, 64, 64).to(self.device) # RGB + Mask
        dummy_mask = torch.zeros(2, 1, 64, 64).to(self.device)
        dummy_sar = torch.rand(2, 2, 64, 64).to(self.device) # VV, VH
        
        output = model(dummy_opt, dummy_mask, dummy_sar)
        
        self.assertEqual(output.shape, (2, 3, 64, 64))
        self.assertTrue(torch.all((output >= 0) & (output <= 1)))
        
    def test_cross_attention_fusion_shapes(self):
        # Channels must be divisible by attention_heads * 8 for the query/key projections
        channels = 64
        fusion = CrossAttentionFusion(channels=channels, attention_heads=4).to(self.device)
        
        dummy_opt_feat = torch.rand(2, channels, 16, 16).to(self.device)
        dummy_sar_feat = torch.rand(2, channels, 16, 16).to(self.device)
        
        output = fusion(dummy_opt_feat, dummy_sar_feat)
        
        self.assertEqual(output.shape, (2, channels, 16, 16))

class TestLossFunctions(unittest.TestCase):
    
    def test_combined_loss(self):
        # LPIPS might fail offline, so we set perceptual_weight to 0 or it falls back to 0.0 internally
        loss_fn = CombinedLoss(l1_weight=1.0, sam_weight=1.0, perceptual_weight=1.0)
        
        pred = torch.rand(2, 3, 32, 32)
        target = torch.rand(2, 3, 32, 32)
        mask = torch.ones(2, 1, 32, 32) # All pixels clouded
        
        total_loss, metrics = loss_fn(pred, target, mask)
        
        self.assertTrue(torch.is_tensor(total_loss))
        self.assertTrue(total_loss.requires_grad)
        self.assertIn('L1', metrics)
        self.assertIn('SAM', metrics)

if __name__ == '__main__':
    unittest.main()
