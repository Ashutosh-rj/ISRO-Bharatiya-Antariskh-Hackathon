import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GenerateSyntheticShim")

def main():
    logger.error("Synthetic data generation (fractal noise, geometrical shapes) has been permanently removed.")
    logger.error("Please utilize real geospatial datasets (e.g. SEN12MS-CR) to train models to ensure scientific integrity.")
    raise NotImplementedError("Synthetic data generation is deprecated and removed.")

if __name__ == "__main__":
    main()
