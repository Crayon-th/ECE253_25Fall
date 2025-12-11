"""
Traditional denoising methods for impulse noise removal.

This module contains classical image denoising filters that work well
for both salt-and-pepper noise and random-valued impulse noise.
"""

import numpy as np
from typing import Tuple


def standard_median_filter(
    img: np.ndarray,
    window: int = 3,
) -> np.ndarray:
    """
    Standard median filter - applies median filtering to all pixels.
    
    Simple but effective for low to moderate noise densities.
    Works for both salt-and-pepper and random-valued impulse noise.
    
    Args:
        img: uint8 image, shape (H, W) or (H, W, C).
        window: window size (odd integer, e.g., 3, 5, 7).
    
    Returns:
        Filtered uint8 image.
    """
    if img.dtype != np.uint8:
        raise ValueError("standard_median_filter expects uint8 image.")
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be an odd integer >= 3.")
    
    if img.ndim == 2:
        return _median_filter_gray(img, window)
    elif img.ndim == 3:
        channels = []
        for c in range(img.shape[2]):
            channels.append(_median_filter_gray(img[:, :, c], window))
        return np.stack(channels, axis=2)
    else:
        raise ValueError("Unsupported image shape, expected 2D or 3D array.")


def _median_filter_gray(img: np.ndarray, window: int) -> np.ndarray:
    """Internal helper for single-channel median filtering."""
    h, w = img.shape
    pad = window // 2
    padded = np.pad(img, pad, mode='edge')
    out = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+window, j:j+window]
            out[i, j] = np.median(patch)
    
    return out.astype(np.uint8)


def decision_based_median_filter(
    img: np.ndarray,
    window: int = 3,
    threshold: float = 40.0,
) -> np.ndarray:
    """
    Decision-based median filter using statistical detection.
    
    This filter detects potential noisy pixels by comparing them with
    the local median. Only detected noisy pixels are replaced.
    
    Algorithm:
    1. Compute local median for each pixel
    2. If |pixel - median| > threshold, mark as noisy
    3. Replace noisy pixels with the local median
    
    Args:
        img: uint8 image, shape (H, W) or (H, W, C).
        window: window size for median computation (odd integer).
        threshold: detection threshold. Higher = less aggressive filtering.
    
    Returns:
        Filtered uint8 image.
    """
    if img.dtype != np.uint8:
        raise ValueError("decision_based_median_filter expects uint8 image.")
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be an odd integer >= 3.")
    
    if img.ndim == 2:
        return _decision_based_median_gray(img, window, threshold)
    elif img.ndim == 3:
        channels = []
        for c in range(img.shape[2]):
            channels.append(_decision_based_median_gray(img[:, :, c], window, threshold))
        return np.stack(channels, axis=2)
    else:
        raise ValueError("Unsupported image shape, expected 2D or 3D array.")


def _decision_based_median_gray(
    img: np.ndarray,
    window: int,
    threshold: float,
) -> np.ndarray:
    """Internal helper for decision-based median filtering."""
    h, w = img.shape
    pad = window // 2
    padded = np.pad(img, pad, mode='edge')
    out = img.copy().astype(np.float32)
    
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+window, j:j+window].astype(np.float32)
            median_val = np.median(patch)
            pixel_val = img[i, j]
            
            # Decision: is this pixel noisy?
            if abs(pixel_val - median_val) > threshold:
                # Replace with median
                out[i, j] = median_val
            # else: keep original value
    
    return np.clip(out, 0, 255).astype(np.uint8)


def adaptive_median_filter_random_noise(
    img: np.ndarray,
    max_window: int = 7,
    threshold: float = 40.0,
) -> np.ndarray:
    """
    Adaptive median filter for random-valued impulse noise.
    
    Combines statistical noise detection with adaptive window sizing.
    More robust than fixed-window filters for varying noise densities.
    
    Algorithm:
    1. For each pixel, compute median in a 3x3 window
    2. If |pixel - median| > threshold, mark as potentially noisy
    3. For noisy pixels, try increasing window sizes
    4. Replace with median from the first window that gives stable result
    
    Args:
        img: uint8 image, shape (H, W) or (H, W, C).
        max_window: maximum window size (odd integer).
        threshold: noise detection threshold.
    
    Returns:
        Filtered uint8 image.
    """
    if img.dtype != np.uint8:
        raise ValueError("adaptive_median_filter_random_noise expects uint8 image.")
    if max_window % 2 == 0 or max_window < 3:
        raise ValueError("max_window must be an odd integer >= 3.")
    
    if img.ndim == 2:
        return _adaptive_median_random_gray(img, max_window, threshold)
    elif img.ndim == 3:
        channels = []
        for c in range(img.shape[2]):
            channels.append(_adaptive_median_random_gray(img[:, :, c], max_window, threshold))
        return np.stack(channels, axis=2)
    else:
        raise ValueError("Unsupported image shape, expected 2D or 3D array.")


def _adaptive_median_random_gray(
    img: np.ndarray,
    max_window: int,
    threshold: float,
) -> np.ndarray:
    """Internal helper for adaptive median filtering with random noise."""
    h, w = img.shape
    pad = max_window // 2
    padded = np.pad(img, pad, mode='edge')
    out = img.copy().astype(np.float32)
    
    for i in range(h):
        for j in range(w):
            ci = i + pad
            cj = j + pad
            pixel_val = img[i, j]
            
            # First check with 3x3 window
            patch_3x3 = padded[ci-1:ci+2, cj-1:cj+2].astype(np.float32)
            median_3x3 = np.median(patch_3x3)
            
            # Noise detection
            if abs(pixel_val - median_3x3) <= threshold:
                # Likely clean pixel, keep original
                continue
            
            # Noisy pixel detected, use adaptive window
            for win in range(3, max_window + 1, 2):
                r = win // 2
                patch = padded[ci-r:ci+r+1, cj-r:cj+r+1].astype(np.float32)
                median_val = np.median(patch)
                
                # Use this median if it's significantly different from pixel
                # (i.e., the median is stable and different from noisy value)
                if abs(median_val - pixel_val) > threshold * 0.5:
                    out[i, j] = median_val
                    break
            else:
                # Fallback: use median from largest window
                r = max_window // 2
                patch = padded[ci-r:ci+r+1, cj-r:cj+r+1].astype(np.float32)
                out[i, j] = np.median(patch)
    
    return np.clip(out, 0, 255).astype(np.uint8)


def adaptive_median_decision_filter(
    img: np.ndarray,
    max_window: int = 7,
    min_clean: int = 3,
) -> np.ndarray:
    """
    Adaptive median / decision-based filter for salt-and-pepper impulse noise.

    Basic logic:
      - If pixel is not 0 or 255 -> keep as is.
      - If pixel is 0 or 255:
          * search in increasing window sizes (3x3, 5x5, ..., max_window x max_window)
          * collect non-impulse pixels (values in (0, 255))
          * if there are at least `min_clean` pixels, replace by their median
          * otherwise, keep original value.

    Args:
        img: uint8 image, shape (H, W) or (H, W, C).
        max_window: maximum window size (odd, e.g. 7).
        min_clean: minimum number of non-impulse pixels required to replace.

    Returns:
        Denoised uint8 image.
    """
    if img.dtype != np.uint8:
        raise ValueError("adaptive_median_decision_filter expects uint8 image.")
    if max_window % 2 == 0 or max_window < 3:
        raise ValueError("max_window must be an odd integer >= 3.")

    if img.ndim == 2:
        return _adaptive_median_decision_filter_gray(img, max_window, min_clean)
    elif img.ndim == 3:
        # Process per-channel independently
        channels = []
        for c in range(img.shape[2]):
            channels.append(_adaptive_median_decision_filter_gray(img[:, :, c], max_window, min_clean))
        return np.stack(channels, axis=2)
    else:
        raise ValueError("Unsupported image shape, expected 2D or 3D array.")


def _adaptive_median_decision_filter_gray(
    img: np.ndarray,
    max_window: int,
    min_clean: int,
) -> np.ndarray:
    """Internal helper for single-channel uint8 images."""
    h, w = img.shape
    pad = max_window // 2
    # edge padding to simplify boundary handling
    padded = np.pad(img, pad, mode="edge")
    out = img.copy().astype(np.float32)

    for i in range(h):
        for j in range(w):
            val = img[i, j]
            # non-impulse pixel, keep as is
            if val != 0 and val != 255:
                continue

            # impulse candidate, try increasing windows
            replaced = False
            ci = i + pad
            cj = j + pad
            for win in range(3, max_window + 1, 2):
                r = win // 2
                r0, r1 = ci - r, ci + r + 1
                c0, c1 = cj - r, cj + r + 1
                patch = padded[r0:r1, c0:c1]
                valid = patch[(patch > 0) & (patch < 255)]
                if valid.size >= min_clean:
                    out[i, j] = np.median(valid)
                    replaced = True
                    break
            if not replaced:
                # keep original 0/255 if no enough clean pixels found
                out[i, j] = val

    return np.clip(out, 0, 255).astype(np.uint8)


def progressive_switching_median_filter(
    img: np.ndarray,
    window: int = 3,
    threshold: float = 40.0,
    iterations: int = 3,
) -> np.ndarray:
    """
    Progressive Switching Median Filter (PSMF).
    
    Iteratively detects and corrects noisy pixels. In each iteration,
    detected noisy pixels are replaced, and the process repeats.
    This allows the filter to progressively improve the image quality.
    
    Args:
        img: uint8 image, shape (H, W) or (H, W, C).
        window: window size for median computation.
        threshold: noise detection threshold.
        iterations: number of filtering iterations.
    
    Returns:
        Filtered uint8 image.
    """
    if img.dtype != np.uint8:
        raise ValueError("progressive_switching_median_filter expects uint8 image.")
    
    result = img.copy()
    
    for _ in range(iterations):
        result = decision_based_median_filter(result, window, threshold)
    
    return result


__all__ = [
    "standard_median_filter",
    "decision_based_median_filter",
    "adaptive_median_filter_random_noise",
    "adaptive_median_decision_filter",
    "progressive_switching_median_filter",
]

