import os
import sys
import logging
import numpy as np
import pystac_client
import planetary_computer
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from pyproj import CRS
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DatasetGenerator")

def get_aligned_array(href, bbox, width, height):
    with rasterio.open(href) as src:
        vrt_options = {
            'resampling': Resampling.bilinear,
            'crs': CRS.from_epsg(4326),
            'transform': rasterio.transform.from_bounds(*bbox, width, height),
            'height': height,
            'width': width,
        }
        with WarpedVRT(src, **vrt_options) as vrt:
            return vrt.read(1)

def main():
    logger.info("Initializing STAC Client...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    # We define a few small bounding boxes to pull from
    # ~0.1 degree is ~11km x 11km. At 10m resolution, that is ~1100x1100 pixels.
    # 1100x1100 pixels yields about (1100//256)**2 = 16 patches.
    # To get ~150 patches, we need ~10 such boxes or a larger box.
    # Let's use one 0.3 x 0.3 degree box (~33km x 33km) -> 3300x3300 pixels -> ~144 patches.
    bbox = [77.4, 12.8, 77.7, 13.1]
    width, height = 3072, 3072 # Must be multiples of 256
    
    logger.info(f"Searching Sentinel-2 for bbox {bbox}...")
    s2_search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime="2023-01-01/2023-06-30",
    )
    s2_items = list(s2_search.items())
    logger.info(f"Found {len(s2_items)} S2 items.")
    
    # Sort by cloud cover
    s2_items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
    
    cloud_free_item = s2_items[0]
    logger.info(f"Selected cloud-free reference: {cloud_free_item.id} (Cloud cover: {cloud_free_item.properties.get('eo:cloud_cover')}%)")
    
    # Find a moderately cloudy item (e.g. 20-60% clouds)
    cloudy_items = [item for item in s2_items if 20 < item.properties.get("eo:cloud_cover", 0) < 80]
    if not cloudy_items:
        # Fallback if no moderately cloudy, just take the cloudiest that isn't 100%
        cloudy_items = [s2_items[-2]] 
        
    cloudy_item = cloudy_items[0]
    logger.info(f"Selected cloudy target: {cloudy_item.id} (Cloud cover: {cloudy_item.properties.get('eo:cloud_cover')}%)")
    
    logger.info(f"Searching Sentinel-1 for bbox {bbox} near {cloudy_item.datetime}...")
    # Search S1 for a whole month around the cloudy item's date
    s1_search = catalog.search(
        collections=["sentinel-1-grd"],
        bbox=bbox,
        datetime="2023-01-01/2023-12-31",
    )
    s1_items = list(s1_search.items())
    logger.info(f"Found {len(s1_items)} S1 items.")
    
    # Sort S1 items by time difference to the cloudy item
    cloudy_time = cloudy_item.datetime
    s1_items.sort(key=lambda x: abs((x.datetime - cloudy_time).total_seconds()))
    s1_item = s1_items[0]
    logger.info(f"Selected S1 match: {s1_item.id} (Delta: {abs((s1_item.datetime - cloudy_time).total_seconds())/3600:.1f} hours)")

    output_dir = os.path.join(project_root, "data", "processed", "train")
    os.makedirs(output_dir, exist_ok=True)
    
    # We will read everything into large 3072x3072 arrays
    logger.info("Downloading and aligning arrays...")
    
    try:
        # 1. Cloud-free S2 (B03=Green, B04=Red, B08=NIR)
        cf_b03 = get_aligned_array(cloud_free_item.assets['B03'].href, bbox, width, height)
        cf_b04 = get_aligned_array(cloud_free_item.assets['B04'].href, bbox, width, height)
        cf_b08 = get_aligned_array(cloud_free_item.assets['B08'].href, bbox, width, height)
        cf_rgbn = np.stack([cf_b03, cf_b04, cf_b08], axis=-1)
        # S2 values are 0-10000 approx reflectance
        cf_rgbn_norm = np.clip((cf_rgbn / 4000.0) * 255, 0, 255).astype(np.uint8)
        
        # 2. Cloudy S2
        cl_b03 = get_aligned_array(cloudy_item.assets['B03'].href, bbox, width, height)
        cl_b04 = get_aligned_array(cloudy_item.assets['B04'].href, bbox, width, height)
        cl_b08 = get_aligned_array(cloudy_item.assets['B08'].href, bbox, width, height)
        cl_rgbn = np.stack([cl_b03, cl_b04, cl_b08], axis=-1)
        cl_rgbn_norm = np.clip((cl_rgbn / 4000.0) * 255, 0, 255).astype(np.uint8)
        
        # Cloud mask from SCL
        scl = get_aligned_array(cloudy_item.assets['SCL'].href, bbox, width, height)
        # SCL 8 = Medium probability cloud, 9 = High probability cloud
        mask = ((scl == 8) | (scl == 9)).astype(np.uint8) * 255
        
        # 3. S1 SAR (vv, vh)
        s1_vv = get_aligned_array(s1_item.assets['vv'].href, bbox, width, height)
        s1_vh = get_aligned_array(s1_item.assets['vh'].href, bbox, width, height)
        sar = np.stack([s1_vv, s1_vh], axis=-1)
        sar_db = 10 * np.log10(np.clip(sar, 1e-5, None))
        sar_norm = np.clip((sar_db + 25) / 30 * 255, 0, 255).astype(np.uint8)

        logger.info(f"Extracting 256x256 patches from {width}x{height} region...")
        
        patch_size = 256
        stride = 256
        saved_count = 0
        
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                p_cf = cf_rgbn_norm[y:y+patch_size, x:x+patch_size]
                p_cl = cl_rgbn_norm[y:y+patch_size, x:x+patch_size]
                p_sar = sar_norm[y:y+patch_size, x:x+patch_size]
                p_mask = mask[y:y+patch_size, x:x+patch_size]
                
                # Check for completely black patches (no data edge)
                if p_cl.mean() < 5 or p_cf.mean() < 5 or p_sar.mean() < 5:
                    continue
                    
                # To ensure the dataset is actually learning cloud removal,
                # let's only save patches that have some cloud in them, or keep them all?
                # We'll save them all to let the model learn clear areas too, but maybe require at least *some* cloud in the overall dataset.
                
                out_path = os.path.join(output_dir, f"real_data_{saved_count:03d}.npz")
                np.savez_compressed(
                    out_path,
                    cloudy=p_cl,
                    cloud_free=p_cf,
                    sar=p_sar,
                    mask=p_mask
                )
                saved_count += 1
                
        logger.info(f"Successfully generated {saved_count} real data patches.")
        logger.info("Done! Please delete any old dummy patches in data/processed/train/ if you only want real data.")
        
    except Exception as e:
        logger.error(f"Failed to process dataset: {e}")

if __name__ == "__main__":
    main()
