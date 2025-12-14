"""
Image quality metrics computation.

This module provides functions for computing image quality metrics
such as PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).
"""

import numpy as np
from typing import Optional

try:
    from skimage.metrics import structural_similarity as ssim_skimage
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def compute_psnr(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: Optional[float] = None
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    PSNR is defined as:
        PSNR = 10 * log10(MAX^2 / MSE)
    
    where MAX is the maximum possible pixel value (255 for uint8 images),
    and MSE is the Mean Squared Error between the two images.
    
    Args:
        img1: First image, uint8 array, shape (H, W) or (H, W, C).
        img2: Second image, uint8 array, shape (H, W) or (H, W, C).
        data_range: Maximum value of the image data range. If None, 
                   automatically determined from image dtype (255 for uint8).
    
    Returns:
        PSNR value in dB. Returns inf if images are identical (MSE = 0).
    
    Raises:
        ValueError: If images have different shapes or unsupported dtype.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape. Got {img1.shape} and {img2.shape}")
    
    # Determine data range
    if data_range is None:
        if img1.dtype == np.uint8:
            data_range = 255.0
        elif img1.dtype in (np.float32, np.float64):
            # Assume normalized [0, 1] range
            if img1.max() <= 1.0 and img2.max() <= 1.0:
                data_range = 1.0
            else:
                data_range = 255.0
        else:
            data_range = float(np.iinfo(img1.dtype).max)
    
    # Convert to float for computation
    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)
    
    # Compute MSE
    mse = np.mean((img1_float - img2_float) ** 2)
    
    # Handle perfect match
    if mse == 0:
        return float('inf')
    
    # Compute PSNR
    psnr = 10 * np.log10(data_range ** 2 / mse)
    
    return float(psnr)


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: Optional[float] = None,
    multichannel: Optional[bool] = None,
    channel_axis: Optional[int] = None,
    win_size: Optional[int] = None,
    use_sample_covariance: bool = True
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    SSIM measures the structural similarity between two images, considering
    luminance, contrast, and structure. Values range from -1 to 1, with 1
    indicating perfect similarity.
    
    This function uses scikit-image's implementation if available, otherwise
    falls back to a basic implementation.
    
    Args:
        img1: First image, uint8 array, shape (H, W) or (H, W, C).
        img2: Second image, uint8 array, shape (H, W) or (H, W, C).
        data_range: Maximum value of the image data range. If None, 
                   automatically determined from image dtype (255 for uint8).
        multichannel: Deprecated. Use channel_axis instead.
        channel_axis: If None, the image is assumed to be grayscale.
                     If int, indicates which axis of the array corresponds to channels.
                     For 3D arrays (H, W, C), use channel_axis=2.
        win_size: Size of the sliding window used for SSIM computation.
                 Default is 11 for images > 11 pixels, or image size otherwise.
        use_sample_covariance: If True, use sample covariance. If False, use population covariance.
    
    Returns:
        SSIM value between -1 and 1. Higher values indicate better similarity.
    
    Raises:
        ValueError: If images have different shapes or unsupported dtype.
        ImportError: If scikit-image is not available and images are multichannel.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape. Got {img1.shape} and {img2.shape}")
    
    # Determine data range
    if data_range is None:
        if img1.dtype == np.uint8:
            data_range = 255.0
        elif img1.dtype in (np.float32, np.float64):
            # Assume normalized [0, 1] range
            if img1.max() <= 1.0 and img2.max() <= 1.0:
                data_range = 1.0
            else:
                data_range = 255.0
        else:
            data_range = float(np.iinfo(img1.dtype).max)
    
    # Determine if multichannel
    if channel_axis is None and multichannel is None:
        # Auto-detect: if 3D and last dimension is 3 or 4, assume it's channels
        if img1.ndim == 3 and img1.shape[2] in (3, 4):
            channel_axis = 2
        else:
            channel_axis = None
    
    # Use scikit-image if available
    if SKIMAGE_AVAILABLE:
        # Handle deprecated multichannel parameter
        if multichannel is not None:
            channel_axis = 2 if multichannel else None
        
        try:
            ssim_value = ssim_skimage(
                img1,
                img2,
                data_range=data_range,
                channel_axis=channel_axis,
                win_size=win_size,
                use_sample_covariance=use_sample_covariance
            )
            return float(ssim_value)
        except Exception as e:
            # Fallback to basic implementation if scikit-image fails
            if img1.ndim == 3 and channel_axis == 2:
                # Compute SSIM for each channel and average
                ssim_values = []
                for c in range(img1.shape[2]):
                    ssim_val = _compute_ssim_basic(
                        img1[:, :, c], img2[:, :, c], data_range
                    )
                    ssim_values.append(ssim_val)
                return float(np.mean(ssim_values))
            else:
                return _compute_ssim_basic(img1, img2, data_range)
    else:
        # Fallback to basic implementation
        if img1.ndim == 3 and channel_axis == 2:
            # Compute SSIM for each channel and average
            ssim_values = []
            for c in range(img1.shape[2]):
                ssim_val = _compute_ssim_basic(
                    img1[:, :, c], img2[:, :, c], data_range
                )
                ssim_values.append(ssim_val)
            return float(np.mean(ssim_values))
        else:
            return _compute_ssim_basic(img1, img2, data_range)


def _compute_ssim_basic(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: float
) -> float:
    """
    Basic SSIM implementation using numpy.
    
    This is a simplified SSIM computation that uses global statistics
    rather than local windows. For more accurate results, use scikit-image.
    
    Args:
        img1: First image, 2D array.
        img2: Second image, 2D array.
        data_range: Maximum value of the image data range.
    
    Returns:
        SSIM value between -1 and 1.
    """
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Constants
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Compute means
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Compute variances and covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    # Compute SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = numerator / denominator
    
    return float(ssim)


__all__ = [
    "compute_psnr",
    "compute_ssim",
]

