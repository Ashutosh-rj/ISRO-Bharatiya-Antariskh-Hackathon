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

    train_dir = os.path.join(project_root, "data", "processed", "train")
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        logger.error(f"No training data found in {train_dir}. Exiting.")
        sys.exit(1)

    train_files = get_npz_files(train_dir)
    train_files.sort() # Ensure stable sorting
    logger.info(f"Found {len(train_files)} training patches.")
    
    # Generate identical train/val split for all models
    random.seed(42)
    random.shuffle(train_files)
    val_size = max(1, int(0.2 * len(train_files)))
    train_split = train_files[:-val_size]
    val_split = train_files[-val_size:]

    # 2. Train LaMa
    logger.info("Step 2: Training LaMa Generator...")
    lama_config = os.path.join(project_root, "configs", "lama_config.yaml")
    lama_trainer = LaMaTrainer(lama_config)
    # Override epochs from args
    lama_trainer.config['training']['epochs'] = args.epochs
    lama_trainer.train(train_split, val_split)

    # 3. Train SAR-Fusion
    logger.info("Step 3: Training SAR-Fusion U-Net...")
    sar_config = os.path.join(project_root, "configs", "sar_fusion_config.yaml")
    sar_trainer = SARFusionTrainer(sar_config)
    # Override epochs from args
    sar_trainer.config['training']['epochs'] = args.epochs
    sar_trainer.train(train_split, val_split)

    logger.info("Training complete! Weights saved to models/*/weights/")

if __name__ == "__main__":
    main()
