import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import DatasetReader
import numpy as np
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def read_geotiff(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reads a GeoTIFF file.
    Returns: image array (H, W, C), and the rasterio profile (metadata).
    """
    with rasterio.open(file_path) as src:
        # Read all bands and transpose to (H, W, C)
        image = src.read()
        image = np.transpose(image, (1, 2, 0))
        profile = src.profile
    return image, profile

def write_geotiff(output_path: str, image: np.ndarray, profile: Dict[str, Any]):
    """
    Writes a numpy array (H, W, C) to a GeoTIFF file preserving metadata.
    """
    # Transpose back to (C, H, W) for rasterio
    if len(image.shape) == 3:
        image_to_write = np.transpose(image, (2, 0, 1))
    else:
        # 2D mask
        image_to_write = np.expand_dims(image, axis=0)

    # Update profile with correct shape and type
    profile.update(
        dtype=image_to_write.dtype,
        count=image_to_write.shape[0],
        width=image_to_write.shape[2],
        height=image_to_write.shape[1]
    )

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(image_to_write)
    logger.info(f"Saved GeoTIFF to {output_path}")

def reproject_to_epsg(src_path: str, dst_path: str, dst_crs: str = 'EPSG:32644'):
    """
    Reprojects a GeoTIFF to a target EPSG code (default: WGS84 UTM Zone 44N for NER India).
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def generate_kml_overlay(geotiff_path: str, kml_output_path: str):
    """
    Generates a KML overlay for the given GeoTIFF for Google Earth viewing.
    (Placeholder implementation)
    """
    # In a full implementation, this uses simplekml and GDAL's gdal2tiles
    logger.info(f"Generating KML overlay for {geotiff_path} -> {kml_output_path}")
    with open(kml_output_path, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
        f.write('  <Document>\n    <name>LISS-IV Overlay</name>\n  </Document>\n')
        f.write('</kml>\n')

def validate_radiometric_consistency(original: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray) -> float:
    """
    Checks the RMSE at the boundary of the cloud mask to ensure smooth blending.
    Returns the RMSE value.
    """
    import cv2
    # Dilate mask to get boundary region
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    boundary = dilated - mask
    
    if np.sum(boundary) == 0:
        return 0.0
        
    orig_boundary = original[boundary == 1]
    recon_boundary = reconstructed[boundary == 1]
    
    rmse = np.sqrt(np.mean((orig_boundary - recon_boundary) ** 2))
    return rmse
