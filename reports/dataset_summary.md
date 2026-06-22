# Dataset Statistics Report

## Primary Dataset: SEN12MS-CR (Research Subset)

To facilitate the Rapid Validation Package (RVP) without requiring week-long data transfers, a 700-sample research subset of the SEN12MS-CR dataset has been extracted and pre-processed.

### Composition
- **Total Scenes**: 700 patches
- **Train Split**: 500 scenes
- **Validation Split**: 100 scenes
- **Test Split**: 100 scenes
- **Resolution**: 10m/px (Sentinel-2/LISS-IV equivalent)
- **Patch Size**: 256x256

### Spectral Bands
- **Optical (Input & Target)**: 4 Bands (Green, Red, NIR, SWIR pseudo-mapped)
- **SAR (Input)**: 2 Bands (C-Band VV, VH polarization)

### Cloud Statistics
- **Average Cloud Cover**: ~42%
- **Cloud Types**: Thick cumulus, thin cirrus, and structured haze.
