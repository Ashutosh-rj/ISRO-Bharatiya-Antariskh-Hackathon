"""
Training & Validation Loss Curve Visualization Script.

Generates plots of training and validation loss trajectories across epochs
to empirically demonstrate model convergence trends and diagnostic stability.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(project_root, "results")
os.makedirs(results_dir, exist_ok=True)

def plot_curves():
    sar_csv = os.path.join(results_dir, "sar_training_history.csv")
    lama_csv = os.path.join(results_dir, "lama_training_history.csv")
    
    plt.figure(figsize=(12, 5))
    
    # 1. SAR-Fusion U-Net Curves
    if os.path.exists(sar_csv):
        df_sar = pd.read_csv(sar_csv)
        plt.subplot(1, 2, 1)
        plt.plot(df_sar['epoch'], df_sar['train_loss'], marker='o', label='Train Loss', color='#1f77b4', linewidth=2)
        plt.plot(df_sar['epoch'], df_sar['val_loss'], marker='s', label='Val Loss', color='#ff7f0e', linewidth=2)
        plt.title('SAR-Fusion U-Net Loss Trajectory (10 Epochs)', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Combined Loss (L1 + SAM + LPIPS)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
    # 2. LaMa Inpainting Curves
    if os.path.exists(lama_csv):
        df_lama = pd.read_csv(lama_csv)
        plt.subplot(1, 2, 2)
        plt.plot(df_lama['epoch'], df_lama['train_loss'], marker='o', label='Train Loss', color='#2ca02c', linewidth=2)
        plt.plot(df_lama['epoch'], df_lama['val_loss'], marker='s', label='Val Loss', color='#d62728', linewidth=2)
        plt.title('LaMa cGAN Loss Trajectory (10 Epochs)', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Combined Loss (L1 + Perceptual)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
    plt.tight_layout()
    out_img = os.path.join(results_dir, "training_loss_curves.png")
    plt.savefig(out_img, dpi=300)
    print(f"Successfully generated loss curve plots at {out_img}")

if __name__ == '__main__':
    plot_curves()
