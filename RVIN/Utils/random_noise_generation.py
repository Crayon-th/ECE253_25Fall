"""
Random value noise generation functions.

This module contains functions for generating various types of impulse noise,
including salt-and-pepper noise and random-valued impulse noise.
"""

import numpy as np
from typing import Optional


def add_random_impulse_noise(
    img: np.ndarray, 
    p: float = 0.1, 
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Add random-valued impulse noise to a uint8 image.
    
    Unlike salt-and-pepper noise, corrupted pixels are replaced with
    random values in [0, 255] instead of just 0 or 255.
    
    Args:
        img: uint8 image, shape (H, W) or (H, W, C), range [0, 255].
        p: noise density in [0, 1]. Fraction of pixels to corrupt.
        rng: optional numpy Generator for reproducibility.
    
    Returns:
        Noisy uint8 image with the same shape as input.
    """
    if img.dtype != np.uint8:
        raise ValueError("add_random_impulse_noise expects uint8 image.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1].")
    
    if rng is None:
        rng = np.random.default_rng()
    
    noisy = img.copy()
    
    if img.ndim == 2:
        h, w = img.shape
        # Generate noise mask
        mask = rng.random((h, w)) < p
        # Replace corrupted pixels with random values
        num_corrupted = mask.sum()
        noisy[mask] = rng.integers(0, 256, size=num_corrupted, dtype=np.uint8)
    elif img.ndim == 3:
        h, w, c = img.shape
        # Generate noise mask (same for all channels)
        mask = rng.random((h, w)) < p
        # Replace corrupted pixels with random values
        num_corrupted = mask.sum()
        for ch in range(c):
            noisy[mask, ch] = rng.integers(0, 256, size=num_corrupted, dtype=np.uint8)
    else:
        raise ValueError("Unsupported image shape, expected 2D or 3D array.")
    
    return noisy


def add_impulse_noise(
    img: np.ndarray, 
    p: float = 0.1, 
    noise_type: str = "salt_pepper",
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Add impulse noise to a uint8 image.

    Args:
        img: uint8 image, shape (H, W) or (H, W, C), range [0, 255].
        p: noise density in [0, 1]. Fraction of pixels to corrupt.
        noise_type: Type of impulse noise to add:
            - "salt_pepper": Salt-and-pepper noise (pixels set to 0 or 255)
            - "random": Random-valued impulse noise (pixels set to random values in [0, 255])
        rng: optional numpy Generator for reproducibility.

    Returns:
        Noisy uint8 image with the same shape as input.
    """
    if img.dtype != np.uint8:
        raise ValueError("add_impulse_noise expects uint8 image.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1].")
    if noise_type not in ("salt_pepper", "random"):
        raise ValueError(f"noise_type must be 'salt_pepper' or 'random', got '{noise_type}'")

    if rng is None:
        rng = np.random.default_rng()

    noisy = img.copy()
    
    if noise_type == "salt_pepper":
        # Salt-and-pepper noise: pixels set to 0 or 255
        if img.ndim == 2:
            h, w = img.shape
            # generate mask for salt vs pepper
            mask = rng.random((h, w))
            noisy[mask < p / 2] = 0
            noisy[(mask >= p / 2) & (mask < p)] = 255
        elif img.ndim == 3:
            h, w, c = img.shape
            mask = rng.random((h, w))
            # broadcast mask to channels
            pepper = mask < p / 2
            salt = (mask >= p / 2) & (mask < p)
            noisy[pepper, :] = 0
            noisy[salt, :] = 255
        else:
            raise ValueError("Unsupported image shape, expected 2D or 3D array.")
    
    elif noise_type == "random":
        # Random-valued impulse noise: pixels set to random values
        if img.ndim == 2:
            h, w = img.shape
            mask = rng.random((h, w)) < p
            random_values = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
            noisy[mask] = random_values[mask]
        elif img.ndim == 3:
            h, w, c = img.shape
            mask = rng.random((h, w)) < p
            # Generate random values for each channel
            random_values = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
            noisy[mask, :] = random_values[mask, :]
        else:
            raise ValueError("Unsupported image shape, expected 2D or 3D array.")
    
    return noisy


def add_salt_pepper_noise(
    img: np.ndarray,
    p: float = 0.1,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Add salt-and-pepper noise to a uint8 image.
    
    Convenience wrapper for add_impulse_noise with noise_type="salt_pepper".
    
    Args:
        img: uint8 image, shape (H, W) or (H, W, C), range [0, 255].
        p: noise density in [0, 1]. Fraction of pixels to corrupt.
        rng: optional numpy Generator for reproducibility.
    
    Returns:
        Noisy uint8 image with the same shape as input.
    """
    return add_impulse_noise(img, p=p, noise_type="salt_pepper", rng=rng)


__all__ = [
    "add_random_impulse_noise",
    "add_impulse_noise",
    "add_salt_pepper_noise",
]

