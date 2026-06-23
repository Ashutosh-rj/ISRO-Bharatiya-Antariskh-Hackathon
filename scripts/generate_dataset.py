import os
import sys
import logging
import numpy as np

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from pipeline.data_download import PlanetaryComputerDownloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GenerateDataset")

def main():
    logger.info("Starting Dataset Pipeline (Genuine Data Only)")
    downloader = PlanetaryComputerDownloader()
    
    # Bounding box for a region in India (e.g., near Bengaluru for varied terrain)
    bbox = [77.5, 12.9, 77.7, 13.1]
    
    # 1. Search for a Sentinel-2 image
    logger.info("Searching for Optical data...")
    s2_items = downloader.search_sentinel2(bbox, "2023-01-01", "2023-03-31", max_cloud_cover=100.0)
    
    # 2. Search for Sentinel-1 SAR data for the same region/time
    logger.info("Searching for SAR data...")
    s1_items = downloader.search_sentinel1(bbox, "2023-01-01", "2023-03-31")
    
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Download items as-is without synthetic injection
    for i in range(min(5, len(s2_items))):
        s2_item = s2_items[i]
        success = downloader.download_s2_patch(s2_item, bbox, raw_dir)
        if success:
            logger.info(f"Downloaded authentic optical data: {s2_item.id}")
            
    for i in range(min(5, len(s1_items))):
        s1_item = s1_items[i]
        downloader.download_s1_patch(s1_item, raw_dir)
        
    logger.info("Dataset download complete. To train on this data, you must pair the optical and SAR patches manually or use a paired dataset like SEN12MS-CR.")

if __name__ == "__main__":
    main()
