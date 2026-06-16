import os
import logging
from typing import Optional, List
import requests
from dotenv import load_dotenv

# Optional import for Sentinel
try:
    from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
except ImportError:
    SentinelAPI = None

load_dotenv()
logger = logging.getLogger(__name__)

class BhuvanDownloader:
    """
    Downloads LISS-IV imagery from ISRO's Bhuvan portal via WMS/WCS APIs.
    Requires BHUVAN_USERNAME and BHUVAN_PASSWORD in .env.
    """
    def __init__(self):
        self.username = os.getenv('BHUVAN_USERNAME')
        self.password = os.getenv('BHUVAN_PASSWORD')
        self.base_url = "https://bhuvan-vec1.nrsc.gov.in/bhuvan/ows"
        
        if not self.username:
            logger.warning("Bhuvan credentials not found in environment. Downloads may fail.")

    def download_region(self, bbox: List[float], start_date: str, end_date: str, output_path: str):
        """
        Mock implementation for downloading LISS-IV data via WMS/WCS.
        bbox: [min_lon, min_lat, max_lon, max_lat]
        """
        logger.info(f"Initiating Bhuvan download for bbox {bbox} from {start_date} to {end_date}.")
        
        # Real implementation would construct WCS GetCoverage requests here.
        # e.g., using requests.get(url, params={...}, auth=(user, pass))
        
        # Mocking a successful download
        logger.info(f"Downloaded LISS-IV GeoTIFF to {output_path}")
        # open(output_path, 'wb').write(b'mock_geotiff_data')


class SentinelDownloader:
    """
    Downloads Sentinel-1 (SAR) and Sentinel-2 (Optical) data via Copernicus API.
    Requires COPERNICUS_USER and COPERNICUS_PASSWORD in .env.
    """
    def __init__(self):
        self.user = os.getenv('COPERNICUS_USER')
        self.password = os.getenv('COPERNICUS_PASSWORD')
        
        if not self.user or not SentinelAPI:
            logger.warning("Copernicus credentials missing or sentinelsat not installed.")
            self.api = None
        else:
            self.api = SentinelAPI(self.user, self.password, 'https://apihub.copernicus.eu/apihub')

    def download_sar(self, footprint_wkt: str, start_date: str, end_date: str, output_dir: str):
        """Downloads Sentinel-1 GRD (SAR) imagery for the specified footprint."""
        if not self.api:
            logger.error("SentinelAPI is not initialized.")
            return
            
        logger.info(f"Querying Sentinel-1 data for {start_date} to {end_date}")
        products = self.api.query(footprint_wkt,
                                  date=(start_date, end_date),
                                  platformname='Sentinel-1',
                                  producttype='GRD')
        
        if products:
            logger.info(f"Found {len(products)} products. Downloading...")
            self.api.download_all(products, directory_path=output_dir)
        else:
            logger.warning("No Sentinel-1 products found for given criteria.")

    def download_optical(self, footprint_wkt: str, start_date: str, end_date: str, output_dir: str):
        """Downloads Sentinel-2 MSI (Optical) imagery for the specified footprint."""
        if not self.api:
            logger.error("SentinelAPI is not initialized.")
            return
            
        logger.info(f"Querying Sentinel-2 data for {start_date} to {end_date}")
        products = self.api.query(footprint_wkt,
                                  date=(start_date, end_date),
                                  platformname='Sentinel-2',
                                  cloudcoverpercentage=(0, 30))
        
        if products:
            logger.info(f"Found {len(products)} products. Downloading...")
            self.api.download_all(products, directory_path=output_dir)
        else:
            logger.warning("No Sentinel-2 products found for given criteria.")
