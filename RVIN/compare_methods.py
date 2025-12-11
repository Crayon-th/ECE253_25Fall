"""
Compare 3 denoising methods on the custom dataset:
1. Adaptive Median Filter for Random Noise (GPU-accelerated Traditional)
2. Decision-based Median Filter (GPU-accelerated Traditional)
3. MC-AWGN-RVIN Model (ML - Deep Learning)

This script processes all images in dataset/noised and compares with dataset/selected (ground truth).
All traditional methods are GPU-accelerated using PyTorch for fast processing of high-resolution images.
Also includes special processing for sample images.
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

# Import GPU-accelerated traditional methods
from TR.traditional_denoising_gpu import (
    adaptive_median_filter_random_noise_gpu,
    decision_based_median_filter_gpu,
)

# Import metrics
from image_metrics import compute_psnr, compute_ssim

# Import ML utilities
from ML.models import DnCNN_c, Estimation_direct
from ML.utils import (
    np2ts,
    visual_va2np,
    pixelshuffle,
)

# Define img_normalize locally (it's not in utils.py but used in other ML scripts)
def img_normalize(data):
    """Normalize image data from [0, 255] to [0, 1]."""
    return data / 255.0

# Limit set for noise level normalization
limit_set = [[0, 75], [0, 80]]  # [AWGN, RVIN]


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
    """
    Denoise image using ML model.
    
    Args:
        img: Input noisy image (BGR, uint8, shape HxWxC)
        model: Denoising model
        model_est: Noise estimation model
        device: torch device
        cond: Condition mode (0=ground truth, 1=estimated, 2=external)
        ps: Pixel shuffle mode (0=none, 1=adaptive, 2=fixed)
        ps_scale: Pixel shuffle scale if ps=2
        color: 0=gray, 1=color
        scale: Resize scale
        rescale: Whether to rescale back
    
    Returns:
        Denoised image (BGR, uint8) with same size as input
    """
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
        # Adaptive pixel shuffle (simplified - use fixed for now)
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
    # Use torch.no_grad() for inference (modern PyTorch)
    with torch.no_grad():
        NM_tensor = torch.clamp(model_est(INoisy), 0., 1.)
        
        # Denoise
        Res = model(INoisy, NM_tensor)
        Out = torch.clamp(INoisy - Res, 0., 1.)
    
    # Convert back to numpy
    # Use original dimensions to ensure output matches input
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


def process_image(
    noisy_path: Path,
    clean_path: Path,
    methods: Dict,
    device,
    output_dir: Path,
    img_name: str,
    verbose: bool = True,
) -> Dict:
    """Process a single image with all methods and compute metrics."""
    start_time = time.time()
    
    # Read images
    noisy = cv2.imread(str(noisy_path), cv2.IMREAD_COLOR)
    clean = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
    
    if noisy is None:
        if verbose:
            print(f"⚠ Warning: Could not read {noisy_path}")
        return None
    if clean is None:
        if verbose:
            print(f"⚠ Warning: Could not read {clean_path}")
        return None
    
    # Resize clean to match noisy if needed
    if noisy.shape[:2] != clean.shape[:2]:
        clean = cv2.resize(clean, (noisy.shape[1], noisy.shape[0]))
    
    results = {
        'image': img_name,
        'shape': noisy.shape,
        'methods': {},
        'timings': {}
    }
    
    # Method 1: Adaptive Median Filter (GPU)
    t0 = time.time()
    if verbose:
        print(f"  [1/3] Adaptive Median Filter (GPU)...", end=' ', flush=True)
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
        str(output_dir / f"{img_name}_adaptive_median_gpu.jpg"),
        denoised_1
    )
    if verbose:
        print(f"✓ ({t1-t0:.2f}s)")
    
    # Method 2: Decision-based Median Filter (GPU)
    t0 = time.time()
    if verbose:
        print(f"  [2/3] Decision-based Median Filter (GPU)...", end=' ', flush=True)
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
        str(output_dir / f"{img_name}_decision_based_gpu.jpg"),
        denoised_2
    )
    if verbose:
        print(f"✓ ({t1-t0:.2f}s)")
    
    # Method 3: ML Model
    t0 = time.time()
    if verbose:
        print(f"  [3/3] ML Model...", end=' ', flush=True)
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
                str(output_dir / f"{img_name}_ml_denoised.jpg"),
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
    
    total_time = time.time() - start_time
    results['timings']['total'] = total_time
    
    if verbose:
        print(f"  Total time: {total_time:.2f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare 3 denoising methods on custom dataset"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to dataset directory (default: auto-detect from project root)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: RVIN/results/comparison)"
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
        "--sample_only",
        action="store_true",
        help="Only process sample images (4_rand_noisy.png and 1_clean.png)"
    )
    args = parser.parse_args()
    
    # Setup paths
    if args.dataset_dir is None:
        dataset_dir = PROJECT_ROOT / "dataset"
    else:
        dataset_dir = Path(args.dataset_dir)
    
    if args.output_dir is None:
        output_dir = SCRIPT_DIR / "results" / "comparison"
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
    
    # Setup GPU-accelerated methods
    # Method 1: Adaptive Median Filter (GPU)
    def gpu_adaptive_median(img):
        """GPU-accelerated adaptive median filter."""
        return adaptive_median_filter_random_noise_gpu(
            img, device=device, max_window=7, threshold=40.0
        )
    
    # Method 2: Decision-based Median Filter (GPU)
    def gpu_decision_based(img):
        """GPU-accelerated decision-based median filter."""
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
    
    # Process images
    all_results = []
    
    if args.sample_only:
        # Process sample images
        print("\n" + "="*80)
        print("PROCESSING SAMPLE IMAGES")
        print("="*80)
        
        # Find sample images
        sample_noisy = None
        sample_clean = None
        
        # Try multiple possible locations
        possible_noisy = [
            SCRIPT_DIR / "ML" / "data" / "single_test" / "4_rand_noisy.png",
            PROJECT_ROOT / "outputs" / "quick_demo" / "4_rand_noisy.png",
        ]
        possible_clean = [
            PROJECT_ROOT / "outputs" / "quick_demo" / "1_clean.png",
        ]
        
        for path in possible_noisy:
            if path.exists():
                sample_noisy = path
                break
        
        for path in possible_clean:
            if path.exists():
                sample_clean = path
                break
        
        if sample_noisy is None or sample_clean is None:
            print("⚠ Warning: Sample images not found!")
            print(f"  Looking for: 4_rand_noisy.png and 1_clean.png")
            return
        
        print(f"Sample noisy image: {sample_noisy}")
        print(f"Sample clean image: {sample_clean}")
        
        result = process_image(
            sample_noisy,
            sample_clean,
            methods,
            device,
            output_dir,
            "sample_4_rand_noisy",
            verbose=True
        )
        if result:
            all_results.append(result)
            print(f"\n✓ Sample image processed")
            if 'Adaptive_Median_GPU' in result['methods']:
                am_result = result['methods']['Adaptive_Median_GPU']
                am_time = result['timings'].get('Adaptive_Median_GPU', 0)
                print(f"  Adaptive Median (GPU): PSNR={am_result['psnr']:.2f} dB, "
                      f"SSIM={am_result['ssim']:.4f}, Time={am_time:.2f}s")
            if 'Decision_Based_GPU' in result['methods']:
                db_result = result['methods']['Decision_Based_GPU']
                db_time = result['timings'].get('Decision_Based_GPU', 0)
                print(f"  Decision-based (GPU): PSNR={db_result['psnr']:.2f} dB, "
                      f"SSIM={db_result['ssim']:.4f}, Time={db_time:.2f}s")
            if 'ML_MC_AWGN_RVIN' in result['methods']:
                ml_result = result['methods']['ML_MC_AWGN_RVIN']
                ml_time = result['timings'].get('ML_MC_AWGN_RVIN', 0)
                if ml_result.get('psnr') is not None:
                    print(f"  ML Model: PSNR={ml_result['psnr']:.2f} dB, "
                          f"SSIM={ml_result['ssim']:.4f}, Time={ml_time:.2f}s")
    else:
        # Process all images in dataset
        print("\n" + "="*80)
        print("PROCESSING DATASET IMAGES")
        print("="*80)
        
        noised_dir = dataset_dir / "noised"
        selected_dir = dataset_dir / "selected"
        
        if not noised_dir.exists():
            print(f"⚠ Error: {noised_dir} does not exist!")
            return
        if not selected_dir.exists():
            print(f"⚠ Error: {selected_dir} does not exist!")
            return
        
        # Get all images
        noisy_images = sorted(noised_dir.glob("*.jpg"), key=lambda x: int(x.stem))
        
        print(f"Found {len(noisy_images)} images to process")
        print(f"Output directory: {output_dir}\n")
        
        for i, noisy_path in enumerate(noisy_images, 1):
            img_name = noisy_path.stem
            clean_path = selected_dir / f"{img_name}.jpg"
            
            if not clean_path.exists():
                print(f"⚠ Warning: Clean image not found: {clean_path}")
                continue
            
            print(f"[{i}/{len(noisy_images)}] Processing {img_name}...")
            
            result = process_image(
                noisy_path,
                clean_path,
                methods,
                device,
                output_dir,
                img_name,
                verbose=True
            )
            
            if result:
                all_results.append(result)
                print(f"  ✓ Completed")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    
    if not all_results:
        print("No results to display.")
        return
    
    # Compute averages
    methods_list = ['Adaptive_Median_GPU', 'Decision_Based_GPU', 'ML_MC_AWGN_RVIN']
    avg_psnr = {m: [] for m in methods_list}
    avg_ssim = {m: [] for m in methods_list}
    
    for result in all_results:
        for method in methods_list:
            if method in result['methods']:
                m_result = result['methods'][method]
                if m_result.get('psnr') is not None:
                    avg_psnr[method].append(m_result['psnr'])
                if m_result.get('ssim') is not None:
                    avg_ssim[method].append(m_result['ssim'])
    
    # Compute average timings
    avg_timings = {m: [] for m in methods_list}
    for result in all_results:
        if 'timings' in result:
            for method in methods_list:
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
        f.write("DENOISING METHODS COMPARISON RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total images processed: {len(all_results)}\n\n")
        
        f.write("AVERAGE METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<25} {'Avg PSNR (dB)':<15} {'Avg SSIM':<15} {'Avg Time (s)':<15} {'Count':<10}\n")
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
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS:\n")
        f.write("="*80 + "\n\n")
        
        for result in all_results:
            f.write(f"Image: {result['image']}\n")
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
            f.write("\n")
    
    print(f"\n✓ Detailed results saved to: {results_file}")
    print(f"✓ All denoised images saved to: {output_dir}")


if __name__ == "__main__":
    main()

