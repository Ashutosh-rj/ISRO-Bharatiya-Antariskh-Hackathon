import os
import glob
import random
import json
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DatasetManager")

class DatasetManager:
    """Manages dataset splits (Train/Val/Test) for the PINNACLE Grade LISS-IV architecture."""
    
    def __init__(self, raw_data_dir: str, split_dir: str):
        self.raw_data_dir = raw_data_dir
        self.split_dir = split_dir
        os.makedirs(self.split_dir, exist_ok=True)
        self.split_file = os.path.join(self.split_dir, "dataset_splits.json")

    def generate_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> Dict[str, List[str]]:
        """Scans the raw data directory and generates consistent train/val/test splits."""
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
        
        random.seed(seed)
        
        # Discover all valid patches/scenes (Assuming standard directory structure or .tif/.npz files)
        # For SEN12MS-CR, directories are grouped by ROI and season
        all_files = []
        for ext in ('*.npz', '*.tif', '*.tiff'):
            all_files.extend(glob.glob(os.path.join(self.raw_data_dir, '**', ext), recursive=True))
            
        all_files = sorted(list(set(all_files))) # Ensure deterministic order
        
        if not all_files:
            logger.warning(f"No valid dataset files found in {self.raw_data_dir}. Cannot generate splits.")
            return {"train": [], "val": [], "test": []}
            
        random.shuffle(all_files)
        
        n = len(all_files)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        splits = {
            "train": all_files[:train_idx],
            "val": all_files[train_idx:val_idx],
            "test": all_files[val_idx:]
        }
        
        self._save_splits(splits)
        logger.info(f"Generated splits: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test.")
        return splits
        
    def _save_splits(self, splits: Dict[str, List[str]]):
        with open(self.split_file, 'w') as f:
            json.dump(splits, f, indent=4)
        logger.info(f"Saved splits to {self.split_file}")
        
    def load_splits(self) -> Dict[str, List[str]]:
        """Loads existing splits or fails if they don't exist."""
        if not os.path.exists(self.split_file):
            logger.error(f"Split file not found at {self.split_file}. Run generate_splits() first.")
            return {}
            
        with open(self.split_file, 'r') as f:
            splits = json.load(f)
            
        logger.info(f"Loaded splits: {len(splits.get('train', []))} train, {len(splits.get('val', []))} val, {len(splits.get('test', []))} test.")
        return splits

if __name__ == "__main__":
    manager = DatasetManager(raw_data_dir="data/raw", split_dir="data/splits")
    manager.generate_splits()
