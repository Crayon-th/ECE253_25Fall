"""
GPU-accelerated traditional denoising methods for impulse noise removal.

This module provides GPU-accelerated versions of traditional filters using PyTorch.
Designed for high-resolution images where CPU methods are too slow.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


def adaptive_median_filter_random_noise_gpu(
    img: np.ndarray,
    device: torch.device,
    max_window: int = 7,
    threshold: float = 40.0,
) -> np.ndarray:
    """
    GPU-accelerated adaptive median filter for random-valued impulse noise.
    
    Args:
        img: uint8 image, shape (H, W) or (H, W, C), range [0, 255].
        device: torch device (cuda or cpu).
        max_window: maximum window size (odd integer).
        threshold: noise detection threshold.
    
    Returns:
        Filtered uint8 image.
    """
    if img.dtype != np.uint8:
        raise ValueError("adaptive_median_filter_random_noise_gpu expects uint8 image.")
    if max_window % 2 == 0 or max_window < 3:
        raise ValueError("max_window must be an odd integer >= 3.")
    
    if img.ndim == 2:
        return _adaptive_median_random_gray_gpu(img, device, max_window, threshold)
    elif img.ndim == 3:
        channels = []
        for c in range(img.shape[2]):
            channels.append(_adaptive_median_random_gray_gpu(
                img[:, :, c], device, max_window, threshold
            ))
        return np.stack(channels, axis=2)
    else:
        raise ValueError("Unsupported image shape, expected 2D or 3D array.")


def _adaptive_median_random_gray_gpu(
    img: np.ndarray,
    device: torch.device,
    max_window: int,
    threshold: float,
) -> np.ndarray:
    """
    Fully GPU-vectorized adaptive median filter for grayscale images.
    Uses tensor operations to avoid CPU loops.
    """
    h, w = img.shape
    
    # Convert to tensor and move to device
    img_tensor = torch.from_numpy(img.astype(np.float32)).to(device)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    img_flat = img_tensor.squeeze()  # (H, W)
    
    # Compute 3x3 median for all pixels at once
    median_3x3 = _median_filter_2d_gpu(img_tensor, 3).squeeze()  # (H, W)
    
    # Vectorized noise detection
    diff = torch.abs(img_flat - median_3x3)
    noisy_mask = diff > threshold
    
    # Initialize output
    out = img_flat.clone()
    
    # If no noisy pixels, return early
    if not torch.any(noisy_mask):
        result = torch.clamp(out, 0, 255).cpu().numpy().astype(np.uint8)
        return result
    
    # Pre-compute medians for all window sizes (fully on GPU)
    medians_by_window = {}
    for win in range(3, max_window + 1, 2):
        medians_by_window[win] = _median_filter_2d_gpu(img_tensor, win).squeeze()  # (H, W)
    
    # Fully vectorized adaptive selection
    # For each window size, compute if it should be used
    # We'll use a vectorized approach to select the appropriate window
    
    # Stack all medians: (num_windows, H, W)
    window_sizes = list(range(3, max_window + 1, 2))
    num_windows = len(window_sizes)
    all_medians = torch.stack([medians_by_window[win] for win in window_sizes], dim=0)  # (num_windows, H, W)
    
    # Compute differences for all windows at once
    # Expand img_flat to (1, H, W) for broadcasting
    img_expanded = img_flat.unsqueeze(0)  # (1, H, W)
    diff_all = torch.abs(all_medians - img_expanded)  # (num_windows, H, W)
    
    # Threshold check: (num_windows, H, W)
    valid_mask = diff_all > (threshold * 0.5)  # (num_windows, H, W)
    
    # Only consider noisy pixels
    noisy_expanded = noisy_mask.unsqueeze(0)  # (1, H, W)
    valid_mask = valid_mask & noisy_expanded  # (num_windows, H, W)
    
    # Find first valid window for each pixel (first True in each column)
    # Create indices for windows
    window_indices = torch.arange(num_windows, device=device).view(-1, 1, 1)  # (num_windows, 1, 1)
    
    # For each pixel, find the first window that satisfies the condition
    # Use argmax to find first True (or use a custom approach)
    # We'll use a more direct approach: create a mask and select
    
    # Initialize output with largest window median (fallback)
    out[noisy_mask] = medians_by_window[max_window][noisy_mask]
    
    # For each window size (from smallest to largest), update pixels that satisfy condition
    # This ensures we use the smallest valid window
    for win_idx, win in enumerate(window_sizes):
        # Get pixels that are noisy and satisfy this window's condition
        update_mask = valid_mask[win_idx] & noisy_mask  # (H, W)
        if torch.any(update_mask):
            out[update_mask] = all_medians[win_idx][update_mask]
    
    result = torch.clamp(out, 0, 255).cpu().numpy().astype(np.uint8)
    return result


def decision_based_median_filter_gpu(
    img: np.ndarray,
    device: torch.device,
    window: int = 3,
    threshold: float = 40.0,
) -> np.ndarray:
    """
    GPU-accelerated decision-based median filter using statistical detection.
    
    This filter detects potential noisy pixels by comparing them with
    the local median. Only detected noisy pixels are replaced.
    
    Args:
        img: uint8 image, shape (H, W) or (H, W, C), range [0, 255].
        device: torch device (cuda or cpu).
        window: window size for median computation (odd integer).
        threshold: detection threshold. Higher = less aggressive filtering.
    
    Returns:
        Filtered uint8 image.
    """
    if img.dtype != np.uint8:
        raise ValueError("decision_based_median_filter_gpu expects uint8 image.")
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be an odd integer >= 3.")
    
    if img.ndim == 2:
        return _decision_based_median_gray_gpu(img, device, window, threshold)
    elif img.ndim == 3:
        channels = []
        for c in range(img.shape[2]):
            channels.append(_decision_based_median_gray_gpu(
                img[:, :, c], device, window, threshold
            ))
        return np.stack(channels, axis=2)
    else:
        raise ValueError("Unsupported image shape, expected 2D or 3D array.")


def _decision_based_median_gray_gpu(
    img: np.ndarray,
    device: torch.device,
    window: int,
    threshold: float,
) -> np.ndarray:
    """GPU-accelerated decision-based median filter for grayscale images."""
    h, w = img.shape
    
    # Convert to tensor and move to device
    img_tensor = torch.from_numpy(img.astype(np.float32)).to(device)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # Compute median for all pixels at once
    median_map = _median_filter_2d_gpu(img_tensor, window)
    median_map = median_map.squeeze()  # (H, W)
    
    # Vectorized decision: replace pixels where |pixel - median| > threshold
    diff = torch.abs(img_tensor.squeeze() - median_map)
    mask = diff > threshold
    
    # Replace noisy pixels with median
    out = img_tensor.squeeze().clone()
    out[mask] = median_map[mask]
    
    result = torch.clamp(out, 0, 255).cpu().numpy().astype(np.uint8)
    return result


def _median_filter_2d_gpu(
    img_tensor: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """
    Efficient GPU-accelerated 2D median filter using unfold.
    
    Args:
        img_tensor: Input tensor of shape (B, C, H, W).
        window_size: Window size (must be odd).
    
    Returns:
        Median filtered tensor of same shape.
    """
    B, C, H, W = img_tensor.shape
    pad = window_size // 2
    
    # Pad the image
    padded = F.pad(img_tensor, (pad, pad, pad, pad), mode='replicate')
    
    # Unfold to get patches: (B, C, H, W, window_size*window_size)
    patches = F.unfold(
        padded,
        kernel_size=window_size,
        stride=1,
        padding=0
    )  # (B, C*window_size*window_size, H*W)
    
    # Reshape to (B, C, window_size*window_size, H, W)
    patches = patches.view(B, C, window_size * window_size, H, W)
    
    # Compute median along the patch dimension
    median, _ = torch.median(patches, dim=2, keepdim=False)
    
    return median


__all__ = [
    "adaptive_median_filter_random_noise_gpu",
    "decision_based_median_filter_gpu",
]

