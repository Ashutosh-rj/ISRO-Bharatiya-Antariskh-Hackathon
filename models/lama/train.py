import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
import yaml
from models.lama.model import LaMaGenerator
from pipeline.preprocessing import LISSIV_Dataset
# from evaluation.metrics import compute_lpips # Could be used in loss

logger = logging.getLogger(__name__)

class LaMaTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cpu') # Forced CPU for hackathon constraint
        logger.info(f"Initializing LaMaTrainer on {self.device}")
        
        self.model = LaMaGenerator(
            in_channels=self.config['model']['in_channels'],
            out_channels=self.config['model']['out_channels']
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.l1_loss = nn.L1Loss()
        # Perceptual and FFT loss stubs (in a full impl, load VGG features here)
        
        self.weights_dir = "models/lama/weights"
        os.makedirs(self.weights_dir, exist_ok=True)

    def train(self, npz_paths: list):
        dataset = LISSIV_Dataset(npz_paths, augment=True)
        dataloader = DataLoader(dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        
        epochs = self.config['training']['epochs']
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                cloudy = batch['cloudy'].to(self.device)
                mask = batch['mask'].to(self.device)
                target = batch['cloud_free'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                pred = self.model(cloudy, mask)
                
                # Compute loss
                loss_l1 = self.l1_loss(pred * mask, target * mask) # Only penalize masked region
                # loss_perceptual = compute_lpips(pred, target, self.device)
                
                loss = loss_l1 * self.config['training']['losses']['l1_weight']
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"lama_epoch_{epoch+1}.pth")
                
        # Final save
        self.save_checkpoint("lama_big.pth")
        self.export_onnx("lama_big.onnx")

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.weights_dir, filename)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved checkpoint to {path}")

    def export_onnx(self, filename: str):
        """Export model to ONNX for 3-5x CPU speedup during inference."""
        self.model.eval()
        path = os.path.join(self.weights_dir, filename)
        
        # Dummy inputs
        dummy_image = torch.randn(1, 3, 256, 256).to(self.device)
        dummy_mask = torch.randn(1, 1, 256, 256).to(self.device)
        
        torch.onnx.export(
            self.model,
            (dummy_image, dummy_mask),
            path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['image', 'mask'],
            output_names=['output'],
            dynamic_axes={
                'image': {0: 'batch_size', 2: 'height', 3: 'width'},
                'mask': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        logger.info(f"Exported model to ONNX at {path}")
