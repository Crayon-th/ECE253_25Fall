"""
Traditional denoising methods module.

This package contains classical image denoising filters for impulse noise removal.
Includes both CPU and GPU-accelerated versions.
"""

from .traditional_denoising import (
    standard_median_filter,
    decision_based_median_filter,
    adaptive_median_filter_random_noise,
    adaptive_median_decision_filter,
    progressive_switching_median_filter,
)

from .traditional_denoising_gpu import (
    adaptive_median_filter_random_noise_gpu,
    decision_based_median_filter_gpu,
)

__all__ = [
    "standard_median_filter",
    "decision_based_median_filter",
    "adaptive_median_filter_random_noise",
    "adaptive_median_decision_filter",
    "progressive_switching_median_filter",
    # GPU versions
    "adaptive_median_filter_random_noise_gpu",
    "decision_based_median_filter_gpu",
]

