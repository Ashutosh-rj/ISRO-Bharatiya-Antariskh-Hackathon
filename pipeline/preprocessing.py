import os
import numpy as np
import cv2
import albumentations as A
import torch
from torch.utils.data import Dataset
import logging
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

class PatchExtractor:
    def __init__(self, patch_size: int = 256, stride: int = 64):
        self.patch_size = patch_size
        self.stride = stride

    def extract_patches(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Dict[str, np.ndarray]]:
        """
        Extracts overlapping patches from an image.
        Returns a list of dicts: {'image': patch, 'mask': mask_patch, 'coord': (y, x)}
        """
        h, w = image.shape[:2]
        patches = []
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patch_dict = {'image': img_patch, 'coord': (y, x)}
                if mask is not None:
                    patch_dict['mask'] = mask[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch_dict)
                
        # Handle edges
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            # Add logic for right and bottom edges if strict coverage is needed
            pass
            
        return patches

class DataAugmentor:
    def __init__(self, crop_size=256):
        self.crop_size = crop_size
        self.transform = A.Compose([
            A.RandomCrop(width=crop_size, height=crop_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2), # Atmospheric noise simulation
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3) # Cloud opacity variance
        ], additional_targets={'mask': 'mask', 'cloud_free': 'image', 'sar': 'image'})

    def augment(self, cloudy: np.ndarray, mask: np.ndarray, cloud_free: np.ndarray = None, sar: np.ndarray = None) -> Dict[str, np.ndarray]:
        # Handle dictionary-based augmentation for multiple modalities
        targets = {'image': cloudy, 'mask': mask}
        if cloud_free is not None:
            targets['cloud_free'] = cloud_free
        if sar is not None:
            targets['sar'] = sar
            
        augmented = self.transform(**targets)
        return augmented

class LISSIV_Dataset(Dataset):
    """
    PyTorch Dataset for LISS-IV + SAR + Sentinel-2 fusion.
    Loads data from pre-extracted .npz files.
    """
    def __init__(self, npz_paths: List[str], augment: bool = False):
        self.npz_paths = npz_paths
        self.augmentor = DataAugmentor() if augment else None

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, idx):
        path = self.npz_paths[idx]
        data = np.load(path)
        
        cloudy = data['cloudy']
        mask = data['mask']
        cloud_free = data.get('cloud_free', np.zeros_like(cloudy)) # For training target
        sar = data.get('sar', np.zeros((cloudy.shape[0], cloudy.shape[1], 2), dtype=cloudy.dtype))
        
        if self.augmentor:
            augmented = self.augmentor.augment(cloudy=cloudy, mask=mask, cloud_free=cloud_free, sar=sar)
            cloudy = augmented['image']
            mask = augmented['mask']
            cloud_free = augmented['cloud_free']
            sar = augmented['sar']
            
        # Convert to float32 tensors, channel first [C, H, W]
        cloudy_t = torch.from_numpy(cloudy).float().permute(2, 0, 1) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0) # [1, H, W]
        cloud_free_t = torch.from_numpy(cloud_free).float().permute(2, 0, 1) / 255.0
        sar_t = torch.from_numpy(sar).float().permute(2, 0, 1) / 255.0
        
        return {
            'cloudy': cloudy_t,
            'mask': mask_t,
            'cloud_free': cloud_free_t,
            'sar': sar_t
        }

def save_patches_npz(patches: List[Dict], output_dir: str, prefix: str):
    """Save a list of patches to individual .npz files."""
    os.makedirs(output_dir, exist_ok=True)
    for i, patch in enumerate(patches):
        filename = os.path.join(output_dir, f"{prefix}_patch_{i}.npz")
        np.savez_compressed(filename, **patch)

def load_patches_npz(npz_paths: List[str]) -> List[Dict]:
    """Load patches from .npz files."""
    return [dict(np.load(p)) for p in npz_paths]
