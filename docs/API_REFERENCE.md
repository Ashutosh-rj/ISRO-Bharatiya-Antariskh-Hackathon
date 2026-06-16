# API Reference

## `pipeline.cloud_detection`
Handles cloud masking algorithms.

### `OtsuCloudDetector`
Uses Otsu's thresholding on the NIR band to detect clouds.
- `__init__(nir_band_index: int = 2)`
- `detect(image: np.ndarray) -> np.ndarray`: Returns binary cloud mask.

## `pipeline.preprocessing`
Handles dataset curation and manipulation.

### `PatchExtractor`
Extracts 256x256 overlapping patches.
- `__init__(patch_size: int = 256, stride: int = 64)`
- `extract_patches(image: np.ndarray, mask: Optional[np.ndarray]) -> List[Dict]`

### `LISSIV_Dataset`
PyTorch Dataset class for loading `.npz` patches during training.
- `__init__(npz_paths: List[str], augment: bool = False)`
- `__getitem__(idx: int) -> Dict[str, torch.Tensor]`

## `models.baseline.opencv_inpaint`
### `OpenCVInpainter`
- `__init__(method: str = 'ns', inpaint_radius: int = 5)`
- `inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray`

## `models.lama.model`
### `LaMaGenerator`
Core generator using Fast Fourier Convolutions.
- `forward(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor`

## `models.sar_fusion.model`
### `SARFusionUNet`
Novel multimodal Dual-Encoder U-Net architecture.
- `forward(optical: torch.Tensor, mask: torch.Tensor, sar: torch.Tensor) -> torch.Tensor`

## `evaluation.metrics`
### `MetricsCalculator`
Computes PSNR, SSIM, LPIPS, SAM, and RMSE.
- `evaluate(pred: np.ndarray, target: np.ndarray) -> Dict[str, Any]`
