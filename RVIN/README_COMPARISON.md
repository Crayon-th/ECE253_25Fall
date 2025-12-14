# Denoising Methods Comparison Script

This script compares 3 denoising methods on the custom dataset:

1. **Adaptive Median Filter for Random Noise** (GPU-accelerated Traditional)
2. **Decision-based Median Filter** (GPU-accelerated Traditional)
3. **MC-AWGN-RVIN Model** (ML - Deep Learning)

**Note:** All traditional methods are GPU-accelerated using PyTorch for fast processing of high-resolution images.

## Dataset

* For the own collected data, please use the following google driver's link to download.
https://drive.google.com/file/d/1ppc-3Rwv1GPH3qGyGyzH7jOEzHINv0wD/view?usp=drive_link

* For the CBSD68 dataset, please refer to this link:
https://github.com/clausmichele/CBSD68-dataset

We only provide unprocessed data. To add noise to them, use the provided scripts:

### Adding Noise to CBSD68 Dataset

For CBSD68 dataset processing with multiple noise levels:

```bash
python compare_cbsd68_multinoise.py --cbsd68_dir /path/to/CBSD68/original --noise_levels 0.1 0.3 0.5 0.95
```

This will add random-valued impulse noise at different densities (10%, 30%, 50%, 95%) to all CBSD68 images and run all denoising methods.

### Adding Noise to Custom Dataset

For custom dataset, prepare your data with:

1. **Clean images**: Place ground truth images in `dataset/selected/`
2. **Noisy images**: Place noisy images in `dataset/noised/` (with corresponding filenames)

Or use the noise generation utilities in `Utils/random_noise_generation.py` to programmatically add noise:

```python
from Utils.random_noise_generation import add_random_impulse_noise
import cv2

# Load clean image
clean_img = cv2.imread('path/to/clean/image.jpg')

# Add random impulse noise (10% density)
noisy_img = add_random_impulse_noise(clean_img, p=0.1)

# Save noisy image
cv2.imwrite('path/to/noisy/image.jpg', noisy_img)
```


## Usage

### Process All Dataset Images

```bash
cd github_upstream/ECE253_25Fall/RVIN
python compare_methods.py
```

### Process Only Sample Images

```bash
python compare_methods.py --sample_only
```

### Custom Options

```bash
python compare_methods.py \
    --dataset_dir /path/to/dataset \
    --output_dir /path/to/output \
    --gpu 0 \
    --ml_model_path /path/to/model
```

## Arguments

- `--dataset_dir`: Path to dataset directory (default: auto-detect from project root)
- `--output_dir`: Output directory for results (default: `RVIN/results/comparison`)
- `--gpu`: GPU device ID (default: auto-detect, use `-1` for CPU)
- `--ml_model_path`: Path to ML model (default: auto-detect from `ML/logs/`)
- `--sample_only`: Only process sample images (4_rand_noisy.png and 1_clean.png)

## Output

The script generates:

1. **Denoised images** for each method:
   - `{image_name}_standard_median.jpg`
   - `{image_name}_adaptive_median.jpg`
   - `{image_name}_ml_denoised.jpg`

2. **Results summary** (`results_summary.txt`) containing:
   - Average PSNR and SSIM for each method
   - Detailed per-image metrics

## Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better (measured in dB)
- **SSIM** (Structural Similarity Index): Higher is better (range 0-1)

## Requirements

- PyTorch
- OpenCV (cv2)
- NumPy
- scikit-image (for SSIM)
- Pre-trained ML model weights in `ML/logs/logs_color_MC_AWGN_RVIN/`

## Notes

- The script automatically handles image resizing if clean and noisy images have different sizes
- If ML model is not available, the script will continue with traditional methods only
- Sample images are automatically detected from common locations

