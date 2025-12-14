# Denoising Methods Comparison Script

This script compares 3 denoising methods on the custom dataset:

1. **Adaptive Median Filter for Random Noise** (GPU-accelerated Traditional)
2. **Decision-based Median Filter** (GPU-accelerated Traditional)
3. **MC-AWGN-RVIN Model** (ML - Deep Learning)

**Note:** All traditional methods are GPU-accelerated using PyTorch for fast processing of high-resolution images.

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

