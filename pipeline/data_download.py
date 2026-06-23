import os
import logging
from typing import Optional, List, Dict
import requests
from dotenv import load_dotenv
import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import Window
import numpy as np

load_dotenv()
logger = logging.getLogger(__name__)

class PlanetaryComputerDownloader:
    """
    Downloads Sentinel-1 (SAR) and Sentinel-2 (Optical) data via Microsoft Planetary Computer STAC API.
    Does not require authentication for public datasets.
    """
    def __init__(self):
        self.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        logger.info("Initialized Planetary Computer STAC Client.")

    def search_sentinel2(self, bbox: List[float], start_date: str, end_date: str, max_cloud_cover: float = 30.0):
        """Finds Sentinel-2 L2A items matching criteria."""
        search = self.catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        )
        items = list(search.items())
        logger.info(f"Found {len(items)} Sentinel-2 items.")
        return items

    def search_sentinel1(self, bbox: List[float], start_date: str, end_date: str):
        """Finds Sentinel-1 GRD items matching criteria."""
        search = self.catalog.search(
            collections=["sentinel-1-grd"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
        )
        items = list(search.items())
        logger.info(f"Found {len(items)} Sentinel-1 items.")
        return items

    def download_s2_patch(self, item, bbox: List[float], output_dir: str):
        """
        Downloads a windowed patch of Sentinel-2 (B03=Green, B04=Red, B08=NIR) + SCL (Scene Classification).
        """
        os.makedirs(output_dir, exist_ok=True)
        # Note: Planetary Computer provides COG, so we can do windowed reads easily.
        # However, for a hackathon, let's just use requests to download the specific assets.
        # Alternatively, using rasterio to read the exact bbox window.
        
        bands = ['B03', 'B04', 'B08', 'SCL']
        patch_data = {}
        
        try:
            for band in bands:
                href = item.assets[band].href
                with rasterio.open(href) as src:
                    # In a full implementation, we'd calculate the exact window from bbox.
                    # For simplicity, we just read the center 1024x1024.
                    width, height = src.width, src.height
                    w = 1024
                    h = 1024
                    window = Window(width // 2 - w // 2, height // 2 - h // 2, w, h)
                    data = src.read(1, window=window)
                    patch_data[band] = data
            
            # Stack Green, Red, NIR
            rgbn = np.stack([patch_data['B03'], patch_data['B04'], patch_data['B08']], axis=-1)
            # Normalize to 0-255 uint8 roughly for LISS-IV approximation
            # Sentinel-2 data is roughly 0-10000 reflectance
            rgbn_norm = np.clip((rgbn / 4000.0) * 255, 0, 255).astype(np.uint8)
            
            scl = patch_data['SCL'] # Scene classification map (clouds, shadows, etc.)
            
            np.savez_compressed(
                os.path.join(output_dir, f"s2_{item.id}.npz"),
                optical=rgbn_norm,
                scl=scl
            )
            logger.info(f"Saved Sentinel-2 patch {item.id}")
            return True
        except Exception as e:
            logger.error(f"Error downloading S2 patch: {e}")
            return False

    def download_s1_patch(self, item, output_dir: str):
        """Downloads a windowed patch of Sentinel-1 (VV, VH)."""
        os.makedirs(output_dir, exist_ok=True)
        bands = ['vv', 'vh']
        patch_data = {}
        
        try:
            for band in bands:
                if band in item.assets:
                    href = item.assets[band].href
                    with rasterio.open(href) as src:
                        w, h = 1024, 1024
                        window = Window(src.width // 2 - w // 2, src.height // 2 - h // 2, w, h)
                        data = src.read(1, window=window)
                        patch_data[band] = data
            
            if 'vv' in patch_data and 'vh' in patch_data:
                sar = np.stack([patch_data['vv'], patch_data['vh']], axis=-1)
                
                # Simple normalization for SAR backscatter to uint8 representation
                sar_db = 10 * np.log10(np.clip(sar, 1e-5, None))
                sar_norm = np.clip((sar_db + 25) / 30 * 255, 0, 255).astype(np.uint8)
                
                np.savez_compressed(
                    os.path.join(output_dir, f"s1_{item.id}.npz"),
                    sar=sar_norm
                )
                logger.info(f"Saved Sentinel-1 patch {item.id}")
                return True
        except Exception as e:
            logger.error(f"Error downloading S1 patch: {e}")
            return False

class BhuvanDownloader:
    """
    Bhuvan integration requires actual ISRO API credentials and WCS access.
    This public version does not contain proprietary data scraping logic.
    """
    def __init__(self):
        self.username = os.getenv('BHUVAN_USERNAME')
        if not self.username or self.username == 'your_username':
            logger.warning("Bhuvan credentials not found.")

    def download_region(self, *args, **kwargs):
        raise NotImplementedError("Bhuvan API access requires authorized ISRO credentials and is disabled in the public demo.")
