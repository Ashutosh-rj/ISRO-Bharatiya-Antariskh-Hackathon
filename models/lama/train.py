import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
import os
import yaml
from models.lama.model import LaMaGenerator
from pipeline.preprocessing import LISSIV_Dataset
from models.losses import CombinedLoss

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
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config['training']['epochs'])
        
        self.loss_fn = CombinedLoss(
            l1_weight=self.config['training']['losses'].get('l1_weight', 10.0),
            sam_weight=self.config['training']['losses'].get('sam_weight', 0.0), # LaMa doesn't typically use SAM but we could
            perceptual_weight=self.config['training']['losses'].get('perceptual_weight', 1.0),
            device=self.device
        )
        
        self.weights_dir = "models/lama/weights"
        os.makedirs(self.weights_dir, exist_ok=True)

    def train(self, npz_paths: list):
        dataset = LISSIV_Dataset(npz_paths, augment=True)
        
        # 80/20 train/val split
        val_size = max(1, int(0.2 * len(dataset)))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False)
        
        epochs = self.config['training']['epochs']
        logger.info(f"Starting training for {epochs} epochs (Train: {train_size}, Val: {val_size})")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                cloudy = batch['cloudy'].to(self.device)
                mask = batch['mask'].to(self.device)
                target = batch['cloud_free'].to(self.device)
                
                self.optimizer.zero_grad()
                
                pred = self.model(cloudy, mask)
                loss, metrics = self.loss_fn(pred, target, mask)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            self.scheduler.step()
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Validation step
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    cloudy = batch['cloudy'].to(self.device)
                    mask = batch['mask'].to(self.device)
                    target = batch['cloud_free'].to(self.device)
                    
                    pred = self.model(cloudy, mask)
                    loss, _ = self.loss_fn(pred, target, mask)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
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
