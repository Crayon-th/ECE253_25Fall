"""
Compare 3 denoising methods on CBSD68 dataset with multiple noise levels:
1. Adaptive Median Filter for Random Noise (GPU-accelerated Traditional)
2. Decision-based Median Filter (GPU-accelerated Traditional)
3. MC-AWGN-RVIN Model (ML - Deep Learning)

This script:
- Generates noisy versions of CBSD68 images at different noise levels (p = 0.1, 0.3, 0.5, 0.95)
- Runs all 3 denoising methods on each noise level
- Saves results and analyzes performance across noise levels
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from datetime import datetime
import time

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "TR"))
sys.path.insert(0, str(SCRIPT_DIR / "Utils"))
sys.path.insert(0, str(SCRIPT_DIR / "ML"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import GPU-accelerated traditional methods
from TR.traditional_denoising_gpu import (
    adaptive_median_filter_random_noise_gpu,
    decision_based_median_filter_gpu,
)

# Import metrics
from image_metrics import compute_psnr, compute_ssim

# Import noise generation
from Utils.random_noise_generation import add_random_impulse_noise

# Import ML utilities
from ML.models import DnCNN_c, Estimation_direct
from ML.utils import (
    np2ts,
    visual_va2np,
    pixelshuffle,
)

# Define img_normalize locally
def img_normalize(data):
    """Normalize image data from [0, 255] to [0, 1]."""
    return data / 255.0

# Limit set for noise level normalization
limit_set = [[0, 75], [0, 80]]


def load_ml_model(device, mode="MC", color=1, num_layers=20, model_path=None):
    """Load the ML denoising model."""
    print(f"Loading ML model (mode={mode}, color={color})...")
    
    # Determine model path
    if model_path is None:
        if color == 1:
            model_path = SCRIPT_DIR / "ML" / "logs" / "logs_color_MC_AWGN_RVIN"
        else:
            model_path = SCRIPT_DIR / "ML" / "logs" / "logs_gray_MC_AWGN_RVIN"
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    c = 3 if color == 1 else 1
    
    # Build models
    if mode == "MC":
        net = DnCNN_c(channels=c, num_of_layers=num_layers, num_of_est=2 * c)
        est_net = Estimation_direct(c, 2 * c)
    else:
        raise ValueError(f"Mode {mode} not supported. Use 'MC' for MC-AWGN-RVIN.")
    
    # Load weights
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        return new_state_dict
    
    if device.type == 'cuda':
        model = nn.DataParallel(net).to(device)
        model_est = nn.DataParallel(est_net).to(device)
    else:
        model = net.to(device)
        model_est = est_net.to(device)
    
    # Load state dicts
    net_path = model_path / "net.pth"
    est_path = model_path / "est_net.pth"
    
    if not net_path.exists() or not est_path.exists():
        raise FileNotFoundError(f"Model weights not found in {model_path}")
    
    state_dict = torch.load(net_path, map_location=device)
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    
    est_state_dict = torch.load(est_path, map_location=device)
    est_state_dict = remove_module_prefix(est_state_dict)
    model_est.load_state_dict(est_state_dict)
    model_est.eval()
    
    print(f"✓ ML model loaded from {model_path}")
    return model, model_est


def denoise_ml(
    img: np.ndarray,
    model,
    model_est,
    device,
    cond=1,
    ps=0,
    ps_scale=2,
    color=1,
    scale=1.0,
    rescale=1,
):
    """Denoise image using ML model."""
    # Store original dimensions
    orig_h, orig_w = img.shape[:2]
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    # Resize if needed
    if scale != 1.0:
        img_rgb = cv2.resize(img_rgb, (0, 0), fx=scale, fy=scale)
        h, w = img_rgb.shape[:2]
    
    # Convert to grayscale if needed
    c = 3 if color == 1 else 1
    if color == 0:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_rgb = np.expand_dims(img_rgb, 2)
    
    # Store shape before pixel shuffle
    h_before_ps, w_before_ps = img_rgb.shape[:2]
    
    # Pixel shuffle
    pss = 1
    if ps == 1:
        pss = ps_scale
        img_rgb = pixelshuffle(img_rgb, pss)
    elif ps == 2:
        pss = ps_scale
        img_rgb = pixelshuffle(img_rgb, pss)
    
    # Normalize to [0, 1]
    img_norm = img_normalize(np.float32(img_rgb))
    
    # Convert to tensor
    INoisy = np2ts(img_norm, color)
    INoisy = torch.clamp(INoisy, 0., 1.)
    INoisy = INoisy.to(device)
    
    # Get noise map - use estimated noise map (cond=1) for blind denoising
    with torch.no_grad():
        NM_tensor = torch.clamp(model_est(INoisy), 0., 1.)
        
        # Denoise
        Res = model(INoisy, NM_tensor)
        Out = torch.clamp(INoisy - Res, 0., 1.)
    
    # Convert back to numpy
    try:
        out_numpy = visual_va2np(
            Out, color, ps, pss, 1, rescale, orig_w, orig_h, c
        )
    except Exception as e:
        # Fallback: direct conversion if visual_va2np fails
        out_numpy = Out.data.squeeze(0).cpu().numpy()
        if out_numpy.ndim == 3:
            out_numpy = np.transpose(out_numpy, (1, 2, 0))
        else:
            out_numpy = np.transpose(out_numpy, (1, 2, 0))
            if out_numpy.shape[2] == 1:
                out_numpy = np.repeat(out_numpy, 3, axis=2)
        
        # Resize to original size
        if out_numpy.shape[:2] != (orig_h, orig_w):
            out_numpy = cv2.resize(out_numpy, (orig_w, orig_h))
    
    # Ensure output matches original size
    if out_numpy.shape[:2] != (orig_h, orig_w):
        out_numpy = cv2.resize(out_numpy, (orig_w, orig_h))
    
    # Convert RGB to BGR and ensure uint8
    if out_numpy.max() <= 1.0:
        out_numpy = (out_numpy * 255.0).astype(np.uint8)
    else:
        out_numpy = out_numpy.astype(np.uint8)
    
    # Handle grayscale output
    if out_numpy.ndim == 2:
        out_numpy = cv2.cvtColor(out_numpy, cv2.COLOR_GRAY2BGR)
    elif out_numpy.ndim == 3 and out_numpy.shape[2] == 3:
        out_bgr = cv2.cvtColor(out_numpy, cv2.COLOR_RGB2BGR)
        return out_bgr
    
    return out_numpy


def process_image_with_noise_level(
    clean_path: Path,
    noise_level: float,
    methods: Dict,
    device,
    output_dir: Path,
    img_name: str,
    rng: np.random.Generator,
    verbose: bool = True,
) -> Dict:
    """Process a single image: generate noisy version and denoise with all methods."""
    start_time = time.time()
    
    # Read clean image
    clean = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
    if clean is None:
        if verbose:
            print(f"⚠ Warning: Could not read {clean_path}")
        return None
    
    # Generate noisy image
    noisy = add_random_impulse_noise(clean, p=noise_level, rng=rng)
    
    results = {
        'image': img_name,
        'noise_level': noise_level,
        'shape': clean.shape,
        'methods': {},
        'timings': {}
    }
    
    # Method 1: Adaptive Median Filter (GPU)
    t0 = time.time()
    if verbose:
        print(f"    [1/3] Adaptive Median Filter (GPU)...", end=' ', flush=True)
    denoised_1 = methods['adaptive_median'](noisy)
    t1 = time.time()
    psnr_1 = compute_psnr(clean, denoised_1)
    ssim_1 = compute_ssim(clean, denoised_1, channel_axis=2)
    results['methods']['Adaptive_Median_GPU'] = {
        'psnr': psnr_1,
        'ssim': ssim_1
    }
    results['timings']['Adaptive_Median_GPU'] = t1 - t0
    cv2.imwrite(
        str(output_dir / f"{img_name}_p{noise_level:.2f}_adaptive_median_gpu.jpg"),
        denoised_1
    )
    if verbose:
        print(f"✓ ({t1-t0:.2f}s)")
    
    # Method 2: Decision-based Median Filter (GPU)
    t0 = time.time()
    if verbose:
        print(f"    [2/3] Decision-based Median Filter (GPU)...", end=' ', flush=True)
    denoised_2 = methods['decision_based'](noisy)
    t1 = time.time()
    psnr_2 = compute_psnr(clean, denoised_2)
    ssim_2 = compute_ssim(clean, denoised_2, channel_axis=2)
    results['methods']['Decision_Based_GPU'] = {
        'psnr': psnr_2,
        'ssim': ssim_2
    }
    results['timings']['Decision_Based_GPU'] = t1 - t0
    cv2.imwrite(
        str(output_dir / f"{img_name}_p{noise_level:.2f}_decision_based_gpu.jpg"),
        denoised_2
    )
    if verbose:
        print(f"✓ ({t1-t0:.2f}s)")
    
    # Method 3: ML Model
    t0 = time.time()
    if verbose:
        print(f"    [3/3] ML Model...", end=' ', flush=True)
    try:
        if methods.get('ml_denoise') is not None:
            denoised_3 = methods['ml_denoise'](noisy)
            t1 = time.time()
            psnr_3 = compute_psnr(clean, denoised_3)
            ssim_3 = compute_ssim(clean, denoised_3, channel_axis=2)
            results['methods']['ML_MC_AWGN_RVIN'] = {
                'psnr': psnr_3,
                'ssim': ssim_3
            }
            results['timings']['ML_MC_AWGN_RVIN'] = t1 - t0
            cv2.imwrite(
                str(output_dir / f"{img_name}_p{noise_level:.2f}_ml_denoised.jpg"),
                denoised_3
            )
            if verbose:
                print(f"✓ ({t1-t0:.2f}s)")
        else:
            if verbose:
                print("⚠ (ML model not available)")
            results['methods']['ML_MC_AWGN_RVIN'] = {
                'psnr': None,
                'ssim': None,
                'error': 'ML model not loaded'
            }
    except Exception as e:
        t1 = time.time()
        if verbose:
            print(f"⚠ Error: {str(e)[:50]}")
        results['methods']['ML_MC_AWGN_RVIN'] = {
            'psnr': None,
            'ssim': None,
            'error': str(e)
        }
        results['timings']['ML_MC_AWGN_RVIN'] = t1 - t0
    
    # Save noisy image for reference
    cv2.imwrite(
        str(output_dir / f"{img_name}_p{noise_level:.2f}_noisy.jpg"),
        noisy
    )
    
    total_time = time.time() - start_time
    results['timings']['total'] = total_time
    
    if verbose:
        print(f"    Total time: {total_time:.2f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare 3 denoising methods on CBSD68 dataset with multiple noise levels"
    )
    parser.add_argument(
        "--cbsd68_dir",
        type=str,
        default=None,
        help="Path to CBSD68 dataset directory (default: auto-detect)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: RVIN/results/cbsd68_multinoise)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID (default: auto-detect, use -1 for CPU)"
    )
    parser.add_argument(
        "--ml_model_path",
        type=str,
        default=None,
        help="Path to ML model (default: auto-detect)"
    )
    parser.add_argument(
        "--noise_levels",
        type=float,
        nargs='+',
        default=[0.1, 0.3, 0.5, 0.95],
        help="Noise levels to test (default: 0.1 0.3 0.5 0.95)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise generation (default: 42)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all)"
    )
    args = parser.parse_args()
    
    # Setup paths
    if args.cbsd68_dir is None:
        cbsd68_dir = PROJECT_ROOT / "ext" / "SeConvNet" / "data" / "Test" / "CBSD68"
    else:
        cbsd68_dir = Path(args.cbsd68_dir)
    
    if not cbsd68_dir.exists():
        print(f"⚠ Error: CBSD68 directory not found: {cbsd68_dir}")
        return
    
    if args.output_dir is None:
        output_dir = SCRIPT_DIR / "results" / "cbsd68_multinoise"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if args.gpu == -1:
        device = torch.device('cpu')
        print("Using CPU")
    elif args.gpu is not None:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu}')
            print(f"Using GPU {args.gpu}")
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    
    # Load ML model
    try:
        model, model_est = load_ml_model(device, mode="MC", color=1, model_path=args.ml_model_path)
    except Exception as e:
        print(f"⚠ Warning: Could not load ML model: {e}")
        print("  Continuing with traditional methods only...")
        model = None
        model_est = None
    
    # Setup methods
    def gpu_adaptive_median(img):
        return adaptive_median_filter_random_noise_gpu(
            img, device=device, max_window=7, threshold=40.0
        )
    
    def gpu_decision_based(img):
        return decision_based_median_filter_gpu(
            img, device=device, window=3, threshold=40.0
        )
    
    methods = {
        'adaptive_median': gpu_adaptive_median,
        'decision_based': gpu_decision_based,
    }
    
    if model is not None:
        methods['ml_denoise'] = lambda img: denoise_ml(
            img, model, model_est, device, cond=1, ps=0, color=1
        )
    else:
        methods['ml_denoise'] = None
    
    # Get all images
    image_files = sorted(cbsd68_dir.glob("*.png"))
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"\nFound {len(image_files)} images in CBSD68 dataset")
    print(f"Noise levels to test: {args.noise_levels}")
    print(f"Output directory: {output_dir}\n")
    
    # Process all images with all noise levels
    all_results = []
    noise_levels = args.noise_levels
    rng = np.random.default_rng(args.seed)
    
    for img_idx, img_path in enumerate(image_files, 1):
        img_name = img_path.stem
        
        print(f"[{img_idx}/{len(image_files)}] Processing {img_name}...")
        
        for noise_level in noise_levels:
            print(f"  Noise level p={noise_level:.2f}:")
            
            result = process_image_with_noise_level(
                img_path,
                noise_level,
                methods,
                device,
                output_dir,
                img_name,
                rng,
                verbose=True
            )
            
            if result:
                all_results.append(result)
        
        print()
    
    # Analyze results
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    if not all_results:
        print("No results to display.")
        return
    
    # Organize results by noise level
    methods_list = ['Adaptive_Median_GPU', 'Decision_Based_GPU', 'ML_MC_AWGN_RVIN']
    
    # Compute statistics for each noise level
    for noise_level in noise_levels:
        print(f"\n{'='*80}")
        print(f"NOISE LEVEL: p = {noise_level:.2f}")
        print(f"{'='*80}")
        
        # Filter results for this noise level
        level_results = [r for r in all_results if r['noise_level'] == noise_level]
        
        if not level_results:
            continue
        
        # Compute averages
        avg_psnr = {m: [] for m in methods_list}
        avg_ssim = {m: [] for m in methods_list}
        avg_timings = {m: [] for m in methods_list}
        
        for result in level_results:
            for method in methods_list:
                if method in result['methods']:
                    m_result = result['methods'][method]
                    if m_result.get('psnr') is not None:
                        avg_psnr[method].append(m_result['psnr'])
                        avg_ssim[method].append(m_result['ssim'])
                    if method in result['timings']:
                        avg_timings[method].append(result['timings'][method])
        
        # Print table
        print(f"\n{'Method':<25} {'Avg PSNR (dB)':<15} {'Avg SSIM':<15} {'Avg Time (s)':<15} {'Count':<10}")
        print("-" * 80)
        
        for method in methods_list:
            if avg_psnr[method]:
                mean_psnr = np.mean(avg_psnr[method])
                mean_ssim = np.mean(avg_ssim[method])
                count = len(avg_psnr[method])
                mean_time = np.mean(avg_timings[method]) if avg_timings[method] else 0.0
                print(f"{method:<25} {mean_psnr:>10.2f}      {mean_ssim:>10.4f}      {mean_time:>10.2f}      {count:>5}")
            else:
                print(f"{method:<25} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'0':<10}")
    
    # Save detailed results to file
    results_file = output_dir / "results_summary.txt"
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CBSD68 MULTI-NOISE LEVEL DENOISING COMPARISON RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total images processed: {len(image_files)}\n")
        f.write(f"Noise levels tested: {noise_levels}\n")
        f.write(f"Total results: {len(all_results)}\n\n")
        
        # Write results by noise level
        for noise_level in noise_levels:
            f.write(f"\n{'='*80}\n")
            f.write(f"NOISE LEVEL: p = {noise_level:.2f}\n")
            f.write(f"{'='*80}\n")
            
            level_results = [r for r in all_results if r['noise_level'] == noise_level]
            
            if not level_results:
                continue
            
            # Compute averages
            avg_psnr = {m: [] for m in methods_list}
            avg_ssim = {m: [] for m in methods_list}
            avg_timings = {m: [] for m in methods_list}
            
            for result in level_results:
                for method in methods_list:
                    if method in result['methods']:
                        m_result = result['methods'][method]
                        if m_result.get('psnr') is not None:
                            avg_psnr[method].append(m_result['psnr'])
                            avg_ssim[method].append(m_result['ssim'])
                        if method in result['timings']:
                            avg_timings[method].append(result['timings'][method])
            
            f.write(f"\n{'Method':<25} {'Avg PSNR (dB)':<15} {'Avg SSIM':<15} {'Avg Time (s)':<15} {'Count':<10}\n")
            f.write("-" * 80 + "\n")
            
            for method in methods_list:
                if avg_psnr[method]:
                    mean_psnr = np.mean(avg_psnr[method])
                    mean_ssim = np.mean(avg_ssim[method])
                    count = len(avg_psnr[method])
                    mean_time = np.mean(avg_timings[method]) if avg_timings[method] else 0.0
                    f.write(f"{method:<25} {mean_psnr:>10.2f}      {mean_ssim:>10.4f}      {mean_time:>10.2f}      {count:>5}\n")
                else:
                    f.write(f"{method:<25} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'0':<10}\n")
            
            # Detailed per-image results
            f.write(f"\nDetailed Results:\n")
            f.write("-" * 80 + "\n")
            for result in level_results:
                f.write(f"\nImage: {result['image']}\n")
                f.write(f"Shape: {result['shape']}\n")
                f.write(f"Total Processing Time: {result['timings'].get('total', 0):.2f}s\n")
                for method, metrics in result['methods'].items():
                    if metrics.get('psnr') is not None:
                        method_time = result['timings'].get(method, 0)
                        f.write(f"  {method}:\n")
                        f.write(f"    PSNR: {metrics['psnr']:.2f} dB\n")
                        f.write(f"    SSIM: {metrics['ssim']:.4f}\n")
                        f.write(f"    Time: {method_time:.2f}s\n")
                    else:
                        method_time = result['timings'].get(method, 0)
                        f.write(f"  {method}: Error - {metrics.get('error', 'Unknown')}\n")
                        f.write(f"    Time: {method_time:.2f}s\n")
    
    print(f"\n✓ Detailed results saved to: {results_file}")
    print(f"✓ All denoised images saved to: {output_dir}")


if __name__ == "__main__":
    main()

