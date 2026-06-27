import os
import sys
import logging
import argparse
import random

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.lama.train import LaMaTrainer
from models.sar_fusion.train import SARFusionTrainer
# from scripts.generate_dataset import main as generate_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainingOrchestrator")

def get_npz_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-data', action='store_true', help='Skip data generation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()

    # 1. Generate Dataset
    if not args.skip_data:
        logger.info("Step 1: Generating Dataset via Planetary Computer STAC...")
        try:
            generate_data()
        except Exception as e:
            logger.error(f"Data generation failed: {e}")
            logger.warning("If you are offline, ensure data/processed/train contains .npz files.")

    train_dir = os.path.join(project_root, "datasets", "SEN12MS-CR_subset", "train")
    val_dir = os.path.join(project_root, "datasets", "SEN12MS-CR_subset", "val")
    if not os.path.exists(train_dir):
        logger.error(f"No training data found in {train_dir}. Exiting.")
        sys.exit(1)

    train_files = sorted(get_npz_files(train_dir))[:30] # 30 patches
    val_split = sorted(get_npz_files(val_dir))[:10]    # 10 val patches
    logger.info(f"Loaded {len(train_files)} training patches and {len(val_split)} validation patches.")
    train_split = train_files

    # 2. Train LaMa (1 Epoch fast adaptation)
    logger.info("Step 2: Training LaMa Generator (1 Epoch)...")
    lama_config = os.path.join(project_root, "configs", "lama_config.yaml")
    lama_trainer = LaMaTrainer(lama_config)
    lama_trainer.config['training']['epochs'] = 1
    lama_trainer.train(train_split, val_split)

    # 3. Train SAR-Fusion U-Net (3 Epochs flagship convergence trend run)
    logger.info("Step 3: Training SAR-Fusion U-Net (3 Epochs)...")
    sar_config = os.path.join(project_root, "configs", "sar_fusion_config.yaml")
    sar_trainer = SARFusionTrainer(sar_config)
    sar_trainer.config['training']['epochs'] = 3
    sar_trainer.train(train_split, val_split)

    logger.info("Training complete! Weights saved to models/*/weights/")

if __name__ == "__main__":
    main()
