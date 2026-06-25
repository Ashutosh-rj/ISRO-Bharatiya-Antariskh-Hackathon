import os
import numpy as np
import pystac_client
import planetary_computer
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from pyproj import CRS

def get_aligned_array(href, bbox, width, height):
    with rasterio.open(href) as src:
        # We want to warp the source to EPSG:4326 with exact bbox
        vrt_options = {
            'resampling': Resampling.bilinear,
            'crs': CRS.from_epsg(4326),
            'transform': rasterio.transform.from_bounds(*bbox, width, height),
            'height': height,
            'width': width,
        }
        with WarpedVRT(src, **vrt_options) as vrt:
            return vrt.read(1)

catalog = pystac_client.Client.open(
    'https://planetarycomputer.microsoft.com/api/stac/v1',
    modifier=planetary_computer.sign_inplace,
)

bbox = [77.5, 12.9, 77.7, 13.1]

search_s2 = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=bbox,
    datetime='2023-01-01/2023-03-31',
)
s2_items = list(search_s2.items())
print(f"Found {len(s2_items)} S2 items")
s2_item = s2_items[0]

search_s1 = catalog.search(
    collections=['sentinel-1-grd'],
    bbox=bbox,
    datetime='2023-01-01/2023-03-31',
)
s1_items = list(search_s1.items())
print(f"Found {len(s1_items)} S1 items")
s1_item = s1_items[0]

# Try to read B04 from S2 and vv from S1 into perfectly aligned 1024x1024 arrays
width, height = 1024, 1024

try:
    print("Reading S2...")
    s2_b04 = get_aligned_array(s2_item.assets['B04'].href, bbox, width, height)
    print(f"S2 shape: {s2_b04.shape}, min: {s2_b04.min()}, max: {s2_b04.max()}")
    
    print("Reading S1...")
    s1_vv = get_aligned_array(s1_item.assets['vv'].href, bbox, width, height)
    print(f"S1 shape: {s1_vv.shape}, min: {s1_vv.min()}, max: {s1_vv.max()}")
    print("Alignment successful!")
except Exception as e:
    print(f"Error: {e}")
