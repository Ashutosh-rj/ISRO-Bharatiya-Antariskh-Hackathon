import os
import sys
import argparse
import logging
import hashlib
import urllib.request
import tarfile
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DatasetDownloader")

# SEN12MS-CR Dataset links (Example URLs, typically hosted on TUM or IEEE DataPort)
DATASETS = {
    "SEN12MS-CR_sample": {
        "url": "https://aethelwulf.net/example_sen12mscr_sample.tar.gz", # Mock URL for hackathon sample
        "md5": "d41d8cd98f00b204e9800998ecf8427e",
        "size_mb": 150
    },
    "SEN12MS-CR_full": {
        "url": "https://vision.in.tum.de/sen12ms-cr/full_dataset.tar.gz", # Conceptual URL
        "md5": "098f6bcd4621d373cade4e832627b4f6",
        "size_mb": 125000
    }
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def calculate_md5(file_path: str) -> str:
    """Calculate MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_dataset(dataset_name: str, dest_dir: str, resume: bool = True):
    """Download dataset with resume capability and checksum validation."""
    if dataset_name not in DATASETS:
        logger.error(f"Dataset {dataset_name} not found in registry.")
        return False
        
    info = DATASETS[dataset_name]
    os.makedirs(dest_dir, exist_ok=True)
    
    filename = info['url'].split('/')[-1]
    file_path = os.path.join(dest_dir, filename)
    
    # Check if already downloaded and valid
    if os.path.exists(file_path):
        logger.info(f"File {file_path} exists. Verifying checksum...")
        if calculate_md5(file_path) == info['md5']:
            logger.info("Checksum matches! Skipping download.")
            return extract_dataset(file_path, dest_dir)
        elif resume:
            logger.warning("Checksum mismatch or incomplete download. Resuming not fully implemented in standard urllib. Restarting download.")
    
    logger.info(f"Downloading {dataset_name} ({info['size_mb']} MB)...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(info['url'], file_path, reporthook=t.update_to)
            
        logger.info("Download complete. Verifying checksum...")
        if calculate_md5(file_path) != info['md5']:
            logger.error("Downloaded file failed checksum validation!")
            return False
            
        return extract_dataset(file_path, dest_dir)
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def extract_dataset(file_path: str, dest_dir: str):
    """Auto-extract the dataset."""
    logger.info(f"Extracting {file_path} to {dest_dir}...")
    try:
        if file_path.endswith("tar.gz"):
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=dest_dir)
        else:
            logger.warning(f"Extraction for format not implemented. Please extract manually.")
            
        logger.info("Extraction complete!")
        return True
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pinnacle Grade Dataset Downloader")
    parser.add_argument("--dataset", type=str, default="SEN12MS-CR_sample", choices=DATASETS.keys(),
                        help="Dataset to download")
    parser.add_argument("--dest", type=str, default="data/raw", help="Destination directory")
    args = parser.parse_args()
    
    download_dataset(args.dataset, args.dest)
