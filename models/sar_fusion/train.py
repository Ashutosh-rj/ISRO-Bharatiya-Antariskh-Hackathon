import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
import os
import yaml
import wandb
from torch.cuda.amp import autocast, GradScaler
from models.sar_fusion.model import SARFusionUNet, PatchGANDiscriminator
from pipeline.preprocessing import LISSIV_Dataset
from models.losses import CombinedLoss, GANLoss

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
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config['training']['epochs'])
        
        # Discriminator Setup
        self.discriminator = PatchGANDiscriminator(in_channels=7).to(self.device)
        self.opt_d = optim.Adam(
            self.discriminator.parameters(), 
            lr=self.config['training']['learning_rate'], 
            betas=(0.5, 0.999)
        )
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(self.opt_d, T_max=self.config['training']['epochs'])
        self.gan_loss_fn = GANLoss().to(self.device)
        
        self.loss_fn = CombinedLoss(
            l1_weight=self.config['training']['losses'].get('l1_weight', 10.0),
            sam_weight=self.config['training']['losses'].get('sam_weight', 2.0),
            perceptual_weight=self.config['training']['losses'].get('perceptual_weight', 1.0),
            adv_weight=self.config['training']['losses'].get('adv_weight', 0.1),
            device=self.device
        )
        
        self.weights_dir = "models/sar_fusion/weights"
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Experiment Tracking
        self.use_wandb = self.config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(project="LISS4-Cloud-Removal", config=self.config)
            
        if self.config.get('debug', False):
            torch.autograd.set_detect_anomaly(True)
            
        self.best_val_loss = float('inf')
        self.patience = self.config['training'].get('patience', 10)
        self.epochs_without_improvement = 0
        self.history = []

    def train(self, train_paths: list, val_paths: list):
        train_dataset = LISSIV_Dataset(train_paths, augment=True)
        val_dataset = LISSIV_Dataset(val_paths, augment=False)
        
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False)
        
        epochs = self.config['training']['epochs']
        logger.info(f"Starting SAR-Fusion training for {epochs} epochs (Train: {train_size}, Val: {val_size})")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                cloudy = batch['cloudy'].to(self.device)
                mask = batch['mask'].to(self.device)
                sar = batch['sar'].to(self.device)
                target = batch['cloud_free'].to(self.device)
                
                # ---------------------
                # Train Discriminator
                # ---------------------
                self.opt_d.zero_grad()
                
                # Forward pass Generator
                pred, _ = self.model(cloudy, mask, sar)
                
                # Real Input to D: [cloudy, mask, target] -> target is real
                real_input_d = torch.cat([cloudy, mask, target], dim=1)
                pred_real_d = self.discriminator(real_input_d)
                loss_d_real = self.gan_loss_fn(pred_real_d, target_is_real=True)
                
                # Fake Input to D: [cloudy, mask, pred] -> pred is fake
                fake_input_d = torch.cat([cloudy, mask, pred.detach()], dim=1)
                pred_fake_d = self.discriminator(fake_input_d)
                loss_d_fake = self.gan_loss_fn(pred_fake_d, target_is_real=False)
                
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                    
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.opt_d.step()
                
                # ---------------------
                # Train Generator
                # ---------------------
                self.optimizer.zero_grad()
                
                # D evaluates the generated image
                fake_input_g = torch.cat([cloudy, mask, pred], dim=1)
                pred_fake_g = self.discriminator(fake_input_g)
                
                loss_g, metrics = self.loss_fn(pred, target, mask, pred_fake_logits=pred_fake_g)
                
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss_g.item()
                
            self.scheduler.step()
            self.scheduler_d.step()
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Validation step
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    cloudy = batch['cloudy'].to(self.device)
                    mask = batch['mask'].to(self.device)
                    sar = batch['sar'].to(self.device)
                    target = batch['cloud_free'].to(self.device)
                    
                    pred, _ = self.model(cloudy, mask, sar)
                    loss, _ = self.loss_fn(pred, target, mask)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            self.history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": self.scheduler.get_last_lr()[0]
            })
            
            if self.use_wandb:
                wandb.log({
                    "Train Loss": avg_train_loss,
                    "Val Loss": avg_val_loss,
                    "LR": self.scheduler.get_last_lr()[0]
                })
                
            # Early Stopping and Checkpointing
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint("best_val_loss.pth")
                logger.info(f"New best validation loss: {avg_val_loss:.4f}. Saved best model.")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break
            
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"sar_fusion_epoch_{epoch+1}.pth")
            
        self.save_checkpoint("sar_fusion_final.pth")
        
        # Save training history CSV
        import pandas as pd
        res_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
        os.makedirs(res_dir, exist_ok=True)
        hist_csv = os.path.join(res_dir, "sar_training_history.csv")
        pd.DataFrame(self.history).to_csv(hist_csv, index=False)
        logger.info(f"Saved training history CSV to {hist_csv}")
        
        try:
            self.export_onnx("sar_fusion.onnx")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.weights_dir, filename)
        torch.save(self.model.state_dict(), path)
        
        # Also save discriminator
        d_filename = filename.replace('sar_fusion', 'discriminator')
        d_path = os.path.join(self.weights_dir, d_filename)
        torch.save(self.discriminator.state_dict(), d_path)
        
        logger.info(f"Saved checkpoint to {path} and {d_path}")

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
