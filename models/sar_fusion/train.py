import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
import yaml
from models.sar_fusion.model import SARFusionUNet
from pipeline.preprocessing import LISSIV_Dataset
# from evaluation.metrics import compute_sam_tensor # For SAM loss

logger = logging.getLogger(__name__)

class SARFusionTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cpu')
        
        self.model = SARFusionUNet(
            optical_channels=self.config['model']['optical_channels'],
            sar_channels=self.config['model']['sar_channels'],
            out_channels=self.config['model']['out_channels'],
            base_filters=self.config['model']['architecture']['base_filters']
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.l1_loss = nn.L1Loss()
        
        self.weights_dir = "models/sar_fusion/weights"
        os.makedirs(self.weights_dir, exist_ok=True)

    def train(self, npz_paths: list):
        dataset = LISSIV_Dataset(npz_paths, augment=True)
        dataloader = DataLoader(dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        
        epochs = self.config['training']['epochs']
        logger.info(f"Starting SAR-Fusion training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                cloudy = batch['cloudy'].to(self.device)
                mask = batch['mask'].to(self.device)
                sar = batch['sar'].to(self.device)
                target = batch['cloud_free'].to(self.device)
                
                self.optimizer.zero_grad()
                
                pred = self.model(cloudy, mask, sar)
                
                loss_l1 = self.l1_loss(pred * mask, target * mask)
                # SAM loss would be added here
                
                loss = loss_l1 * self.config['training']['losses']['l1_weight']
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
            
        self.save_checkpoint("sar_fusion.pth")
        self.export_onnx("sar_fusion.onnx")

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.weights_dir, filename)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved checkpoint to {path}")

    def export_onnx(self, filename: str):
        self.model.eval()
        path = os.path.join(self.weights_dir, filename)
        
        dummy_optical = torch.randn(1, 3, 256, 256).to(self.device)
        dummy_mask = torch.randn(1, 1, 256, 256).to(self.device)
        dummy_sar = torch.randn(1, 2, 256, 256).to(self.device)
        
        torch.onnx.export(
            self.model,
            (dummy_optical, dummy_mask, dummy_sar),
            path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['optical', 'mask', 'sar'],
            output_names=['output'],
            dynamic_axes={
                'optical': {0: 'batch_size', 2: 'height', 3: 'width'},
                'mask': {0: 'batch_size', 2: 'height', 3: 'width'},
                'sar': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        logger.info(f"Exported model to ONNX at {path}")
